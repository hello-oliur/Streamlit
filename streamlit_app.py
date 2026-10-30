"""Food Security & Malnutrition Risk Prediction (SDG 3) - Streamlit app.

Features
 1. Food Security Assessment (single household form)   -> "Assessment"
 2. AI Malnutrition Risk Prediction + Risk Score        -> "Risk Prediction"
 3. Risk Factor Analysis (SHAP)                         -> "Risk Factors"
 4. Personalised Food Recommendation                    -> "Food Recommendations"
 5. Budget-Based Meal Planning                          -> "Meal Planner"
 6. Risk Trend Tracking                                 -> "Trends"
 7. Upload a CSV of many households -> cleaned with the training data,
    scored by the trained model                         -> "Upload Data"
 8. Dashboard & Report (PDF)                            -> "Dashboard", "Report"

Files this app reads from the repository root (created by the Colab notebook):
 * model.pkl        trained model bundle: {"model", "columns", "label_map"}   (required)
 * foods.csv        food prices and nutrition for recommendations / meal plan (required)
 * ml_training.db   SQLite training data, used to clean uploaded data and to
                    compare it with the training population                   (recommended)
"""
import hashlib
import hmac
import html
import io
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

try:
    from fpdf import FPDF

    HAVE_FPDF = True
except Exception:  # fpdf2 not installed -> PDF buttons are disabled
    FPDF = object
    HAVE_FPDF = False

st.set_page_config(page_title="Food Security Risk Assessment", page_icon="🍚", layout="wide")

# ----------------------------------------------------------------- settings
DB_PATH = os.environ.get("FS_DB_PATH", "foodsecurity.db")  # accounts, history, uploads (created automatically)
MODEL_PATH = "model.pkl"
FOODS_CSV = "foods.csv"
TRAIN_DB = "ml_training.db"
MAX_UPLOAD_ROWS = 5000

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SEC_QUESTION = "What is your favourite food?"
DISCLAIMER = "Screening guidance only. This is not a medical diagnosis."

LEVELS = ["Low", "Moderate", "High"]
COLORS = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#dc2626"}
LEVEL_STYLE = {
    "Low": ("✅", "linear-gradient(135deg,#15803d,#22c55e)", "Your household shows a low food-security risk. Keep it up!"),
    "Moderate": ("⚠️", "linear-gradient(135deg,#d97706,#fbbf24)", "Some risk factors need attention. See the suggestions below."),
    "High": ("🚨", "linear-gradient(135deg,#b91c1c,#f87171)", "Several risk factors are present. Please act on the recommendations."),
}

# ---------------------------------------------------------------- indicators
FACTORS = [
    "Income per person",
    "Meals per day",
    "Dietary diversity",
    "Difficulty accessing food",
    "Employment",
    "Child under 5",
    "Food spending share",
]
FACTOR_ICON = {
    "Income per person": "💰",
    "Meals per day": "🍽️",
    "Dietary diversity": "🥦",
    "Difficulty accessing food": "🛒",
    "Employment": "💼",
    "Child under 5": "🧒",
    "Food spending share": "🧾",
}
SHORT = {
    "Income per person": "Income",
    "Meals per day": "Meals",
    "Dietary diversity": "Diet variety",
    "Difficulty accessing food": "Food access",
    "Employment": "Employment",
    "Child under 5": "Child under 5",
    "Food spending share": "Food spending",
}
MAX_RAW = {  # maximum risk points of each indicator (used for the 0-100 risk score)
    "Income per person": 3.0,
    "Meals per day": 3.0,
    "Dietary diversity": 2.8,
    "Difficulty accessing food": 2.4,
    "Employment": 2.0,
    "Child under 5": 0.5,
    "Food spending share": 1.0,
}
BASELINE = {  # typical household points, only used if SHAP is unavailable
    "Income per person": 1.2,
    "Meals per day": 0.5,
    "Dietary diversity": 0.5,
    "Difficulty accessing food": 1.2,
    "Employment": 0.7,
    "Child under 5": 0.25,
    "Food spending share": 0.5,
}
ADVICE = {
    "Income per person": "Look into local social-support or food-assistance programs and community kitchens.",
    "Meals per day": "Aim for at least 3 meals a day. Batch-cooked dishes like khichuri stretch a small budget.",
    "Dietary diversity": "Add low-cost variety: lentils, eggs, seasonal vegetables, leafy greens and small fish.",
    "Difficulty accessing food": "Check nearby markets, buy staples in bulk with neighbours, or ask about community food banks.",
    "Employment": "Ask about local skills-training and job-support programs for a more stable income.",
    "Child under 5": "Keep up regular child health check-ups and include eggs, milk and lentils in the child's diet.",
    "Food spending share": "A large share of income goes to food. Plan weekly menus and buy staples in bulk to cut costs.",
}
FACTOR_NEEDS = {  # what each risk factor means for food choices
    "Income per person": {"tags": ["protein", "energy"], "value": True},
    "Meals per day": {"tags": ["energy"], "value": False},
    "Dietary diversity": {"tags": ["vitaminA", "vitaminC", "iron", "calcium"], "value": False},
    "Difficulty accessing food": {"tags": ["local"], "value": False},
    "Employment": {"tags": [], "value": True},
    "Child under 5": {"tags": ["protein", "iron", "calcium", "vitaminA"], "value": False},
    "Food spending share": {"tags": [], "value": True},
}
TAG_LABEL = {
    "energy": "energy-dense",
    "protein": "high protein",
    "iron": "iron-rich",
    "calcium": "calcium source",
    "vitaminA": "vitamin A",
    "vitaminC": "vitamin C",
    "local": "easy to find locally",
}
ACCESS_LABEL = {1: "1 – Easy", 2: "2", 3: "3", 4: "4", 5: "5 – Very hard"}

FOOD_COLS = ["Food", "Category", "Price (BDT/kg)", "kcal/100g", "Protein g/100g", "Max g/person/day", "Tags"]
CATEGORY_EMOJI = {"Staple": "🍚", "Pulse": "🫘", "Protein": "🥚", "Dairy": "🥛", "Vegetable": "🥬", "Fruit": "🍌", "Fat": "🫒"}
MEAL_SPLIT = {  # share of each food category eaten at (breakfast, lunch, dinner)
    "Staple": (0.25, 0.40, 0.35),
    "Pulse": (0.10, 0.45, 0.45),
    "Protein": (0.25, 0.35, 0.40),
    "Dairy": (0.55, 0.20, 0.25),
    "Vegetable": (0.00, 0.50, 0.50),
    "Fruit": (0.50, 0.25, 0.25),
    "Fat": (0.20, 0.40, 0.40),
}
MEALS = [("Breakfast", "🌅"), ("Lunch", "☀️"), ("Dinner", "🌙")]

# ------------------------------------------------------------ upload columns
CANON = [
    "household_size", "monthly_income", "food_expenditure", "meals_per_day",
    "dietary_diversity", "employment", "child_under5", "food_access",
]
FEATURE_NUM = ["household_size", "monthly_income", "food_expenditure", "meals_per_day", "dietary_diversity", "food_access"]
CRITICAL = ["household_size", "monthly_income", "food_expenditure"]   # rows missing these are rejected
IMPUTABLE = ["meals_per_day", "dietary_diversity", "food_access", "employment", "child_under5"]
VALID_RANGE = {"meals_per_day": (1, 4), "dietary_diversity": (0, 12), "food_access": (1, 5)}
COLUMN_GUIDE = pd.DataFrame(
    [
        ("household_id", "Optional", "Any text or number that identifies the household"),
        ("household_size", "Required", "Number of people (1 or more)"),
        ("monthly_income", "Required", "Total monthly household income in BDT"),
        ("food_expenditure", "Required", "Monthly spending on food in BDT"),
        ("meals_per_day", "Required", "1, 2, 3 or 4"),
        ("dietary_diversity", "Required", "Food groups eaten, 0 to 12"),
        ("employment", "Required", "Regular, Daily wage or Unemployed"),
        ("child_under5", "Required", "1 / 0 or Yes / No"),
        ("food_access", "Required", "1 (easy) to 5 (very hard)"),
        ("risk_level", "Optional", "Low, Moderate or High. If present, the model is checked against it"),
    ],
    columns=["Column", "Required?", "Allowed values"],
)
ALIASES = {
    "id": "household_id", "hh_id": "household_id", "hhid": "household_id",
    "hh_size": "household_size", "family_size": "household_size", "household_members": "household_size", "members": "household_size",
    "income": "monthly_income", "household_income": "monthly_income", "monthly_household_income": "monthly_income",
    "food_exp": "food_expenditure", "food_spending": "food_expenditure", "monthly_food_expenditure": "food_expenditure",
    "food_expense": "food_expenditure", "monthly_food_spending": "food_expenditure",
    "meals": "meals_per_day", "number_of_meals": "meals_per_day", "meals_a_day": "meals_per_day",
    "diet_diversity": "dietary_diversity", "dietary_diversity_score": "dietary_diversity", "hdds": "dietary_diversity", "dds": "dietary_diversity",
    "employment_status": "employment", "job_type": "employment",
    "child_under_5": "child_under5", "children_under_5": "child_under5", "has_child_under5": "child_under5",
    "child_under_five": "child_under5", "child": "child_under5",
    "food_access_difficulty": "food_access", "access": "food_access", "food_availability_difficulty": "food_access",
    "risk": "risk_level", "actual_risk": "risk_level",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1180px;}

.hero {background: linear-gradient(135deg,#0f766e 0%,#16a34a 100%); color:#fff; padding:30px 34px;
       border-radius:22px; margin-bottom:1.2rem;}
.hero .h-title {font-size:2rem; font-weight:800; line-height:1.2;}
.hero .h-sub {opacity:.93; margin-top:.45rem; font-size:1.02rem;}

.auth-hero {background: linear-gradient(145deg,#0f766e 0%,#15803d 55%,#22c55e 100%); color:#fff;
            padding:42px 38px; border-radius:26px; min-height:520px;}
.auth-hero .logo {font-size:2.6rem; background:rgba(255,255,255,.18); width:68px; height:68px;
                  display:flex; align-items:center; justify-content:center; border-radius:18px;}
.auth-hero .a-title {font-size:2.3rem; font-weight:800; line-height:1.15; margin:1.2rem 0 .6rem 0;}
.auth-hero .a-sub {opacity:.93; font-size:1.02rem; margin-bottom:1.4rem;}
.auth-hero ul {list-style:none; padding:0; margin:0 0 1.6rem 0;}
.auth-hero li {margin:.7rem 0; font-size:1.0rem;}
.auth-hero .tag {display:inline-block; background:rgba(255,255,255,.2); padding:.35rem .8rem;
                 border-radius:999px; font-size:.82rem; font-weight:600;}

.result-hero {display:flex; align-items:center; gap:20px; color:#fff; padding:24px 28px;
              border-radius:22px; margin-bottom:1rem;}
.result-hero .rh-icon {font-size:2.8rem;}
.result-hero .rh-title {font-size:2rem; font-weight:800; line-height:1.1;}
.result-hero .rh-sub {opacity:.95; margin-top:.2rem;}

.brand {display:flex; align-items:center; gap:12px; margin-bottom:.8rem;}
.brand .b-logo {font-size:1.6rem; background:linear-gradient(135deg,#0f766e,#22c55e); width:46px; height:46px;
                display:flex; align-items:center; justify-content:center; border-radius:14px;}
.brand .b-name {font-weight:800; font-size:1.05rem; line-height:1.15;}
.brand .b-sub {font-size:.75rem; opacity:.65;}
.user-row {display:flex; align-items:center; gap:12px;}
.avatar {background:linear-gradient(135deg,#0f766e,#22c55e); color:#fff; font-weight:700; width:42px; height:42px;
         display:flex; align-items:center; justify-content:center; border-radius:50%; flex-shrink:0;}
.user-row .u-name {font-weight:700; line-height:1.2;}
.user-row .u-mail {font-size:.78rem; opacity:.7; word-break:break-all;}
.sec-label {font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; opacity:.55; font-weight:700;
            margin:.9rem 0 .2rem .3rem;}
.feature-icon {font-size:1.9rem; line-height:1.1;}
.f-title {font-weight:700; font-size:1.0rem; margin:.15rem 0 .1rem 0;}
.f-desc {font-size:.85rem; opacity:.72; min-height:4.1em; margin-bottom:.6rem;}
.chip {display:inline-block; background:#e6f6ee; color:#0b6b3a; padding:.14rem .6rem; border-radius:999px;
       font-size:.74rem; font-weight:600; margin:0 .3rem .3rem 0;}
.chip.warn {background:#fff3dc; color:#9a5b00;}
.chip.red {background:#fde8e8; color:#b42318;}
.food-emoji {font-size:2.1rem; line-height:1;}
.food-name {font-weight:700; font-size:1.05rem;}
.food-meta {font-size:.82rem; opacity:.75; margin:.15rem 0 .45rem 0;}
.meal-title {font-weight:800; font-size:1.08rem; margin-bottom:.2rem;}
.meal-kcal {font-size:.78rem; opacity:.7; margin-bottom:.5rem;}
.meal-item {display:flex; justify-content:space-between; padding:.22rem 0; border-bottom:1px dashed #e5e7eb; font-size:.92rem;}
.step-row {display:flex; gap:10px; flex-wrap:wrap; margin:.2rem 0 .8rem 0;}
.step {background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:.55rem .9rem; font-size:.88rem; font-weight:600;}
.step b {color:#0f9d58;}

[data-testid="stSidebar"] [data-testid="stButton"] {width:100%;}
[data-testid="stSidebar"] [data-testid="stButton"] > button {width:100%; justify-content:flex-start;
        border-radius:12px; padding:.5rem .9rem; font-weight:600;}
[data-testid="stFormSubmitButton"] > button {width:100%;}
button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
    background:#0f9d58; border-color:#0f9d58; color:#fff;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {background:#0b7a45; border-color:#0b7a45; color:#fff;}
[data-testid="stMetricValue"] {font-weight:800;}
</style>
"""


# ================================================================== database
def q(sql, params=(), one=False, write=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, params)
        if write:
            conn.commit()
            return cur.lastrowid
        rows = cur.fetchall()
        if one:
            return rows[0] if rows else None
        return rows
    finally:
        conn.close()


def init_db():
    q(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            sec_hash TEXT NOT NULL,
            created_at TEXT NOT NULL)""",
        write=True,
    )
    q(
        """CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            score INTEGER NOT NULL,
            level TEXT NOT NULL,
            data TEXT NOT NULL)""",
        write=True,
    )
    q(
        """CREATE TABLE IF NOT EXISTS batch_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            n_rows INTEGER NOT NULL,
            meta TEXT NOT NULL,
            results_csv TEXT NOT NULL)""",
        write=True,
    )


# ================================================================== security
def make_hash(secret):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, 120_000)
    return salt.hex() + "$" + digest.hex()


def check_hash(secret, stored):
    try:
        salt_hex, digest_hex = stored.split("$")
        test = hashlib.pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt_hex), 120_000).hex()
        return hmac.compare_digest(test, digest_hex)
    except ValueError:
        return False


# ===================================================================== users
def create_user(name, email, password, answer):
    name, email = name.strip(), email.strip().lower()
    if len(name) < 2:
        return False, "Please enter your full name."
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not answer.strip():
        return False, "Please answer the security question."
    if q("SELECT id FROM users WHERE email=?", (email,), one=True):
        return False, "An account with this email already exists. Please log in."
    uid = q(
        "INSERT INTO users (name, email, pw_hash, sec_hash, created_at) VALUES (?,?,?,?,?)",
        (name, email, make_hash(password), make_hash(answer.strip().lower()), date.today().isoformat()),
        write=True,
    )
    return True, uid


def authenticate(email, password):
    row = q("SELECT * FROM users WHERE email=?", (email.strip().lower(),), one=True)
    if row and check_hash(password, row["pw_hash"]):
        return row
    return None


def reset_password(email, answer, new_password):
    row = q("SELECT * FROM users WHERE email=?", (email.strip().lower(),), one=True)
    if not row or not check_hash(answer.strip().lower(), row["sec_hash"]):
        return False, "Email or security answer is incorrect."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."
    q("UPDATE users SET pw_hash=? WHERE id=?", (make_hash(new_password), row["id"]), write=True)
    return True, "Password updated. You can log in now."


def login_user(row):
    uid = int(row["id"])
    st.session_state["user"] = {
        "id": uid,
        "name": row["name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }
    st.session_state["last"] = latest_result(uid)
    st.session_state["batch"] = load_latest_batch(uid)
    st.session_state["page"] = "Home"


def logout():
    st.session_state["user"] = None
    st.session_state["last"] = None
    st.session_state["batch"] = None
    st.session_state["page"] = "Home"


# ===================================================== trained model + SHAP
@st.cache_resource
def load_model_bundle():
    """Loads model.pkl (made in Colab). Returns None if missing or unreadable."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib

        bundle = joblib.load(MODEL_PATH)
        if all(k in bundle for k in ("model", "columns", "label_map")):
            return bundle
    except Exception:
        pass
    return None


@st.cache_resource
def get_explainer():
    import shap

    return shap.TreeExplainer(load_model_bundle()["model"])


def feature_frame(df, bundle):
    """Canonical household columns -> the exact feature matrix the model was trained on."""
    X = pd.DataFrame(
        {
            "household_size": df["household_size"].astype(float),
            "monthly_income": df["monthly_income"].astype(float),
            "food_expenditure": df["food_expenditure"].astype(float),
            "meals_per_day": df["meals_per_day"].astype(float),
            "dietary_diversity": df["dietary_diversity"].astype(float),
            "child_under5": df["child_under5"].astype(int),
            "food_access": df["food_access"].astype(float),
            "employment_Daily wage": (df["employment"] == "Daily wage").astype(int),
            "employment_Regular": (df["employment"] == "Regular").astype(int),
            "employment_Unemployed": (df["employment"] == "Unemployed").astype(int),
        },
        index=df.index,
    )
    return X.reindex(columns=bundle["columns"], fill_value=0)


def feature_factor(name):
    if name.startswith("employment"):
        return "Employment"
    return {
        "household_size": "Income per person",
        "monthly_income": "Income per person",
        "food_expenditure": "Food spending share",
        "meals_per_day": "Meals per day",
        "dietary_diversity": "Dietary diversity",
        "food_access": "Difficulty accessing food",
        "child_under5": "Child under 5",
    }.get(name)


def shap_factor_matrix(bundle, X):
    """SHAP values toward the High-risk class, grouped into the 7 factors shown to users."""
    model = bundle["model"]
    high_idx = list(model.classes_).index(bundle["label_map"]["High"])
    sv = get_explainer().shap_values(X)
    if isinstance(sv, list):
        vals = np.array(sv[high_idx])
    else:
        arr = np.array(sv)
        vals = arr[:, :, high_idx] if arr.ndim == 3 else arr
    out = pd.DataFrame(0.0, index=X.index, columns=FACTORS)
    for j, name in enumerate(X.columns):
        f = feature_factor(name)
        if f:
            out[f] += vals[:, j]
    return out


def predict_proba_frame(bundle, X):
    inv = {v: k for k, v in bundle["label_map"].items()}
    model = bundle["model"]
    P = pd.DataFrame(model.predict_proba(X), index=X.index, columns=[inv[int(c)] for c in model.classes_])
    if set(P.columns) != set(LEVELS):
        raise ValueError("The model must predict exactly Low, Moderate and High.")
    return P[LEVELS]


# ------------------------------------------------ indicators (risk score)
def rule_contributions(d):
    """Risk points (0 = no risk) of each indicator. They form the 0-100 Food Security Risk Score."""
    people = max(d["size"], 1)
    pc_income = d["income"] / people
    c = {}
    c["Income per person"] = 3 if pc_income < 3000 else 2 if pc_income < 6000 else 1 if pc_income < 10000 else 0
    c["Meals per day"] = {1: 3, 2: 1.5, 3: 0, 4: 0}[int(d["meals"])]
    c["Dietary diversity"] = round(max(0, 7 - d["diversity"]) * 0.4, 2)
    c["Difficulty accessing food"] = round((d["access"] - 1) * 0.6, 2)
    c["Employment"] = {"Regular": 0, "Daily wage": 1, "Unemployed": 2}[d["employment"]]
    c["Child under 5"] = 0.5 if d["child"] else 0
    share = d["food_exp"] / d["income"] if d["income"] > 0 else 1
    c["Food spending share"] = 1 if share > 0.6 else 0.5 if share > 0.5 else 0
    return {k: float(v) for k, v in c.items()}


def composite_score(raw):
    return int(min(100, round(sum(raw.values()) / 14 * 100)))


def d_to_frame(d):
    return pd.DataFrame(
        [{
            "household_size": d["size"], "monthly_income": d["income"], "food_expenditure": d["food_exp"],
            "meals_per_day": d["meals"], "dietary_diversity": d["diversity"], "employment": d["employment"],
            "child_under5": int(d["child"]), "food_access": d["access"],
        }]
    )


def row_to_d(r):
    return {
        "size": int(r["household_size"]), "income": float(r["monthly_income"]), "food_exp": float(r["food_expenditure"]),
        "meals": int(r["meals_per_day"]), "diversity": float(r["dietary_diversity"]), "employment": r["employment"],
        "child": bool(int(r["child_under5"])), "access": int(r["food_access"]),
    }


def predict_household(d):
    """Predicts one household with the trained model and explains it with SHAP."""
    bundle = load_model_bundle()
    X = feature_frame(d_to_frame(d), bundle)
    probs = {k: float(v) for k, v in predict_proba_frame(bundle, X).iloc[0].items()}
    raw = rule_contributions(d)
    try:
        contrib = {k: float(v) for k, v in shap_factor_matrix(bundle, X).iloc[0].items()}
        explain = "SHAP"
    except Exception:  # SHAP not installed -> approximate explanation from indicator points
        contrib = {k: raw[k] - BASELINE[k] for k in FACTORS}
        explain = "rules"
    return {
        "level": max(probs, key=probs.get),
        "score": composite_score(raw),
        "probs": probs,
        "contributions": contrib,
        "indicators": raw,
        "inputs": d,
        "source": "model",
        "explain": explain,
    }


def risk_drivers(contrib, n=3):
    """Factors that push the risk up the most (relative threshold)."""
    mx = max((abs(v) for v in contrib.values()), default=0) or 1
    pos = [(k, v) for k, v in contrib.items() if v > 0 and v >= 0.1 * mx]
    return [k for k, _ in sorted(pos, key=lambda kv: -kv[1])][:n]


def describe_input(d, factor):
    people = max(d["size"], 1)
    share = d["food_exp"] / d["income"] * 100 if d["income"] > 0 else 100
    return {
        "Income per person": f"{d['income'] / people:,.0f} BDT per person / month",
        "Meals per day": f"{int(d['meals'])} per day",
        "Dietary diversity": f"{d['diversity']:.0f} of 12 food groups",
        "Difficulty accessing food": f"{d['access']} out of 5",
        "Employment": d["employment"],
        "Child under 5": "Yes" if d["child"] else "No",
        "Food spending share": f"{share:.0f}% of income",
    }[factor]


# ===================================================== training-data reference
@st.cache_data
def training_reference():
    """Training households from ml_training.db (None if the file is not in the repository)."""
    if not os.path.exists(TRAIN_DB):
        return None
    try:
        con = sqlite3.connect(TRAIN_DB)
        df = pd.read_sql(
            "SELECT household_size, monthly_income, food_expenditure, meals_per_day, dietary_diversity, "
            "employment, child_under5, food_access, risk_level FROM households",
            con,
        )
        con.close()
    except Exception:
        return None
    if df.empty:
        return None
    ipp = df["monthly_income"] / df["household_size"]
    return {
        "df": df,
        "n": int(len(df)),
        "risk_pct": (df["risk_level"].value_counts(normalize=True) * 100).reindex(LEVELS).fillna(0).to_dict(),
        "medians": {c: float(df[c].median()) for c in FEATURE_NUM},
        "modes": {"employment": str(df["employment"].mode()[0]), "child_under5": int(df["child_under5"].mode()[0])},
        "ipp_range": (float(ipp.quantile(0.01)), float(ipp.quantile(0.99))),
    }


@st.cache_data
def model_runs():
    if not os.path.exists(TRAIN_DB):
        return None
    try:
        con = sqlite3.connect(TRAIN_DB)
        df = pd.read_sql(
            "SELECT model_name AS Model, ROUND(accuracy,3) AS Accuracy, ROUND(macro_f1,3) AS 'Macro F1', "
            "ROUND(high_risk_recall,3) AS 'High-risk recall', ROUND(cv_macro_f1,3) AS 'CV macro F1' "
            "FROM model_runs ORDER BY macro_f1 DESC",
            con,
        )
        con.close()
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data
def global_importance():
    """Average absolute SHAP effect of each factor over the training households."""
    ref = training_reference()
    bundle = load_model_bundle()
    if ref is None or bundle is None:
        return None
    try:
        sample = ref["df"].sample(min(800, len(ref["df"])), random_state=1)
        X = feature_frame(sample, bundle)
        M = shap_factor_matrix(bundle, X)
        return pd.DataFrame({"Factor": FACTORS, "Mean |effect|": [float(M[f].abs().mean()) for f in FACTORS]})
    except Exception:
        return None


# ============================================================== assessments
def save_assessment(user_id, res, when):
    payload = {
        "probs": res["probs"],
        "contributions": res["contributions"],
        "inputs": res["inputs"],
        "source": res.get("source", "model"),
        "explain": res.get("explain", "SHAP"),
    }
    q(
        "INSERT INTO assessments (user_id, created_at, score, level, data) VALUES (?,?,?,?,?)",
        (user_id, when.isoformat(), res["score"], res["level"], json.dumps(payload, default=float)),
        write=True,
    )


def result_from_row(row):
    d = json.loads(row["data"])
    inputs = d["inputs"]
    return {
        "id": row["id"],
        "level": row["level"],
        "score": row["score"],
        "probs": d["probs"],
        "contributions": d["contributions"],
        "indicators": rule_contributions(inputs),
        "inputs": inputs,
        "date": date.fromisoformat(row["created_at"]),
        "source": d.get("source", "model"),
        "explain": d.get("explain", "SHAP"),
    }


def latest_result(user_id):
    row = q(
        "SELECT * FROM assessments WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
        one=True,
    )
    return result_from_row(row) if row else None


def load_assessments(user_id):
    rows = q("SELECT * FROM assessments WHERE user_id=? ORDER BY created_at ASC, id ASC", (user_id,))
    return [result_from_row(r) for r in rows]


def history_df(assessments):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime([a["date"] for a in assessments]),
            "Risk score": [a["score"] for a in assessments],
            "Risk level": [a["level"] for a in assessments],
        }
    )


def run_assessment():
    """Form callback: validates the form, predicts with the trained model and saves the result."""
    s = st.session_state
    fields = [
        ("in_size", "household size"), ("in_income", "monthly income"), ("in_food", "monthly food expenditure"),
        ("in_emp", "employment"), ("in_meals", "meals per day"), ("in_div", "dietary diversity score"),
        ("in_access", "difficulty accessing food"), ("in_child", "child under 5"),
    ]
    missing = [label for key, label in fields if s.get(key) is None]
    if missing:
        s["form_error"] = "Please fill in: " + ", ".join(missing) + "."
        return
    data = {
        "size": int(s["in_size"]),
        "income": float(s["in_income"]),
        "food_exp": float(s["in_food"]),
        "meals": int(s["in_meals"]),
        "diversity": float(s["in_div"]),
        "employment": s["in_emp"],
        "child": s["in_child"] == "Yes",
        "access": int(s["in_access"]),
    }
    if data["income"] <= 0:
        s["form_error"] = "Monthly income must be above 0."
        return
    s["form_error"] = None
    res = predict_household(data)
    res["date"] = date.today()
    save_assessment(s["user"]["id"], res, res["date"])
    s["last"] = res
    s["page"] = "Risk Prediction"


def wide_button(*args, **kwargs):
    """Full-width button; falls back for older Streamlit versions."""
    try:
        return st.button(*args, width="stretch", **kwargs)
    except TypeError:
        return st.button(*args, use_container_width=True, **kwargs)


def trend_text(diff):
    """Lower risk score is better, so a negative change is good news."""
    return "No change" if diff == 0 else f"{diff:+d} pts"


def goto(page):
    st.session_state["page"] = page


# ===================================================================== foods
@st.cache_data
def load_foods():
    """Reads foods.csv. Returns None if the file is missing or has the wrong columns."""
    if not os.path.exists(FOODS_CSV):
        return None
    try:
        df = pd.read_csv(FOODS_CSV)
        if not set(FOOD_COLS).issubset(df.columns):
            return None
        df = df[FOOD_COLS].copy()
    except Exception:
        return None
    df["Tags"] = df["Tags"].fillna("").astype(str)
    df["Protein per 10 BDT (g)"] = (df["Protein g/100g"] * 10 / df["Price (BDT/kg)"] * 10).round(1)
    df["kcal per 10 BDT"] = (df["kcal/100g"] * 10 / df["Price (BDT/kg)"] * 10).round(0)
    return df


def foods_missing_notice():
    st.error(
        "The food table **foods.csv** was not found (or has the wrong columns). Upload the `foods.csv` created by the "
        "Colab notebook to the root of your GitHub repository, then reboot the app.",
        icon=":material/error:",
    )


def recommend_foods(r, n=6, max_price=None, exclude=()):
    foods = load_foods()
    drivers = risk_drivers(r["contributions"], 4) if r else []
    tags, value_w = [], 1.0
    for f in drivers:
        tags += FACTOR_NEEDS[f]["tags"]
        if FACTOR_NEEDS[f]["value"]:
            value_w = 2.0
    if r and r["inputs"]["child"]:
        tags += FACTOR_NEEDS["Child under 5"]["tags"]
    tags = list(dict.fromkeys(tags))

    df = foods[~foods["Food"].isin(exclude)]
    if max_price:
        df = df[df["Price (BDT/kg)"] <= max_price]
    if df.empty:
        return [], drivers, tags
    value = 0.5 * df["Protein per 10 BDT (g)"].rank(pct=True) + 0.5 * df["kcal per 10 BDT"].rank(pct=True)
    scored = []
    for (_, row), v in zip(df.iterrows(), value):
        have = set(row["Tags"].split("|"))
        matched = [t for t in tags if t in have]
        scored.append((len(matched) + value_w * float(v), row, matched, float(v)))
    scored.sort(key=lambda x: -x[0])

    picked, per_cat = [], {}
    for _, row, matched, v in scored:
        if per_cat.get(row["Category"], 0) >= 2:
            continue
        per_cat[row["Category"]] = per_cat.get(row["Category"], 0) + 1
        picked.append({"row": row, "matched": matched, "value": v})
        if len(picked) >= n:
            break
    return picked, drivers, tags


# ============================================================== meal planner
def compute_plan(people, kcal_pp, prot_pp, variety=True, exclude=()):
    foods = load_foods()
    cost = foods["Price (BDT/kg)"].to_numpy(float) / 1000.0  # BDT per gram
    kcal = foods["kcal/100g"].to_numpy(float) / 100.0
    prot = foods["Protein g/100g"].to_numpy(float) / 100.0
    bounds = [
        (0, 0 if f in exclude else float(m) * people)
        for f, m in zip(foods["Food"], foods["Max g/person/day"])
    ]
    rows = [-kcal, -prot, kcal]
    rhs = [-kcal_pp * people, -prot_pp * people, 1.3 * kcal_pp * people]
    if variety:
        veg = (foods["Category"] == "Vegetable").to_numpy(float)
        fruit = (foods["Category"] == "Fruit").to_numpy(float)
        rows += [-veg, -fruit]
        rhs += [-150.0 * people, -50.0 * people]
    res = linprog(cost, A_ub=np.vstack(rows), b_ub=np.array(rhs, float), bounds=bounds, method="highs")
    if not res.success:
        return None
    grams = np.where(res.x < 1e-6, 0, res.x)
    plan = foods[["Food", "Category"]].copy()
    plan["Grams per day"] = grams.round(0).astype(int)
    plan["Cost per day (BDT)"] = (grams * cost).round(1)
    plan = plan[plan["Grams per day"] > 0].reset_index(drop=True)

    meals = {m: [] for m, _ in MEALS}
    meal_kcal = {m: 0.0 for m, _ in MEALS}
    for i, row in foods.iterrows():
        if grams[i] <= 0:
            continue
        split = MEAL_SPLIT.get(row["Category"], (0.3, 0.35, 0.35))
        for (m, _), share in zip(MEALS, split):
            g_pp = grams[i] * share / people
            if g_pp >= 5:
                meals[m].append((row["Food"], int(round(g_pp))))
                meal_kcal[m] += g_pp * kcal[i]
    return {
        "plan": plan,
        "total": float(res.fun),
        "kcal_pp": float((grams * kcal).sum()) / people,
        "protein_pp": float((grams * prot).sum()) / people,
        "meals": meals,
        "meal_kcal": meal_kcal,
        "people": people,
    }


# ================================================== CSV upload: cleaning
def normalize_columns(df):
    new = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        new[c] = ALIASES.get(key, key)
    out = df.rename(columns=new)
    return out.loc[:, ~out.columns.duplicated()]


def _norm_emp(v):
    if pd.isna(v):
        return np.nan
    s = re.sub(r"[^a-z]+", " ", str(v).lower()).strip()
    if s in {"regular", "regular employment", "permanent", "salaried", "formal", "employed", "full time", "fulltime", "service"}:
        return "Regular"
    if s in {"daily wage", "daily wages", "day labour", "day laborer", "day labourer", "daily", "wage", "casual",
             "wage labour", "wage labor", "daily wage worker"}:
        return "Daily wage"
    if s in {"unemployed", "jobless", "no job", "none", "not working", "no work"}:
        return "Unemployed"
    return np.nan


def _norm_child(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "t", "present"}:
        return 1
    if s in {"no", "n", "false", "f", "none", "absent"}:
        return 0
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return 1 if x >= 1 else 0 if x == 0 else np.nan


def clean_upload(raw, ref):
    """Validates and cleans an uploaded table using rules and (if available) the training data.

    * rows without household size, income or food spending are rejected
    * other missing values are filled with the training-data median / most common value
    * out-of-range values are adjusted to the valid range
    * households far outside the training population are flagged (kept)
    """
    df = normalize_columns(raw).reset_index(drop=True)
    missing_cols = [c for c in CANON if c not in df.columns]
    if missing_cols:
        return {"error": missing_cols, "found": list(df.columns)}
    n = len(df)
    idx = df.index
    work = pd.DataFrame(index=idx)
    work["row_number"] = np.arange(n) + 2  # row number in the spreadsheet (row 1 is the header)

    if "household_id" in df.columns:
        ids = df["household_id"].astype(str).str.strip()
        id_given = ids.ne("") & ids.ne("nan")
        work["household_id"] = np.where(id_given, ids, (np.arange(n) + 1).astype(str))
    else:
        id_given = pd.Series(False, index=idx)
        work["household_id"] = (np.arange(n) + 1).astype(str)

    for c in FEATURE_NUM:
        work[c] = pd.to_numeric(df[c], errors="coerce")
    work["employment"] = df["employment"].map(_norm_emp)
    work["child_under5"] = df["child_under5"].map(_norm_child)
    if "risk_level" in df.columns:
        lab = df["risk_level"].astype(str).str.strip().str.title()
        work["actual_risk"] = lab.where(lab.isin(LEVELS))
    extras = [c for c in df.columns if c not in CANON and c not in ("household_id", "risk_level")][:5]
    for e in extras:
        work[e] = df[e]

    reasons = [[] for _ in range(n)]

    def mark(mask, text):
        for i in np.where(np.asarray(mask, dtype=bool))[0]:
            reasons[i].append(text)

    for c in CRITICAL:
        mark(work[c].isna(), f"missing or non-numeric {c}")

    imp = pd.DataFrame(False, index=idx, columns=IMPUTABLE)
    for c in IMPUTABLE:
        m = work[c].isna()
        if ref is None:
            mark(m, f"missing {c}")
        else:
            imp[c] = m
            fill = ref["modes"][c] if c in ref["modes"] else ref["medians"][c]
            work.loc[m, c] = fill

    for c in ("household_size", "meals_per_day", "dietary_diversity", "food_access"):
        work[c] = work[c].round()

    mark(work["household_size"].notna() & (work["household_size"] < 1), "household_size below 1")
    mark(work["monthly_income"].notna() & (work["monthly_income"] <= 0), "monthly_income must be above 0")
    mark(work["food_expenditure"].notna() & (work["food_expenditure"] < 0), "negative food_expenditure")

    clip = pd.DataFrame(False, index=idx, columns=["household_size"] + list(VALID_RANGE))
    m = work["household_size"].notna() & (work["household_size"] > 30)
    clip.loc[m, "household_size"] = True
    work.loc[m, "household_size"] = 30
    for c, (lo, hi) in VALID_RANGE.items():
        m = work[c].notna() & ((work[c] < lo) | (work[c] > hi))
        clip[c] = m
        work[c] = work[c].clip(lo, hi)

    mark(id_given & work["household_id"].duplicated(keep="first"), "duplicate household_id")

    ipp = work["monthly_income"] / work["household_size"]
    warn_unusual = pd.Series(False, index=idx)
    if ref is not None:
        lo, hi = ref["ipp_range"]
        warn_unusual = ipp.notna() & ((ipp < lo) | (ipp > hi))
    warn_share = (work["food_expenditure"] > work["monthly_income"]).fillna(False)

    rej = pd.Series([bool(r) for r in reasons], index=idx)

    def quality(i):
        parts = [f"filled {c} from training data" for c in IMPUTABLE if imp.at[i, c]]
        parts += [f"{c} adjusted to valid range" for c in clip.columns if clip.at[i, c]]
        if warn_unusual[i]:
            parts.append("outside training range")
        if warn_share[i]:
            parts.append("food spending above income")
        return "; ".join(parts) or "OK"

    keep = work[~rej].copy()
    keep["data_quality"] = [quality(i) for i in keep.index]
    for c in ("household_size", "meals_per_day", "dietary_diversity", "food_access", "child_under5"):
        keep[c] = keep[c].astype(int)

    rejected = raw.reset_index(drop=True)[rej].copy()
    rejected.insert(0, "Row number", work.loc[rej, "row_number"].to_numpy())
    rejected["Reason"] = ["; ".join(reasons[i]) for i in np.where(rej.to_numpy())[0]]

    reason_counts = Counter(r for rs in reasons for r in rs)
    kept_mask = ~rej
    return {
        "clean": keep,
        "rejected": rejected,
        "extras": extras,
        "counts": {
            "n_raw": int(n),
            "n_used": int(kept_mask.sum()),
            "n_rejected": int(rej.sum()),
            "n_filled_rows": int((imp.any(axis=1) & kept_mask).sum()),
            "n_adjusted_rows": int((clip.any(axis=1) & kept_mask).sum()),
            "n_unusual": int((warn_unusual & kept_mask).sum()),
        },
        "reason_counts": dict(reason_counts),
        "used_training_reference": ref is not None,
    }


def run_batch(clean, extras):
    """Scores every cleaned household with the trained model and explains it with SHAP."""
    bundle = load_model_bundle()
    X = feature_frame(clean, bundle)
    P = predict_proba_frame(bundle, X)
    try:
        C = shap_factor_matrix(bundle, X)
        explain = "SHAP"
    except Exception:
        explain = "rules"
        rows = []
        for _, r in clean.iterrows():
            raw = rule_contributions(row_to_d(r))
            rows.append({k: raw[k] - BASELINE[k] for k in FACTORS})
        C = pd.DataFrame(rows, index=clean.index)
    ind = pd.DataFrame([rule_contributions(row_to_d(r)) for _, r in clean.iterrows()], index=clean.index)
    score = (ind.sum(axis=1) / 14 * 100).clip(upper=100).round().astype(int)
    top = C.idxmax(axis=1).where(C.max(axis=1) > 0, "None")

    res = pd.DataFrame(
        {
            "household_id": clean["household_id"],
            "household_size": clean["household_size"],
            "monthly_income": clean["monthly_income"],
            "food_expenditure": clean["food_expenditure"],
            "meals_per_day": clean["meals_per_day"],
            "dietary_diversity": clean["dietary_diversity"],
            "employment": clean["employment"],
            "child_under5": clean["child_under5"],
            "food_access": clean["food_access"],
            "income_per_person": (clean["monthly_income"] / clean["household_size"]).round(0),
            "risk_level": P.idxmax(axis=1),
            "p_low": P["Low"].round(4),
            "p_moderate": P["Moderate"].round(4),
            "p_high": P["High"].round(4),
            "risk_score": score,
            "top_risk_driver": top,
            "data_quality": clean["data_quality"],
        }
    )
    for f in FACTORS:
        res[f"effect_{f}"] = C[f].round(4)
    if "actual_risk" in clean.columns:
        res["actual_risk"] = clean["actual_risk"]
    for e in extras:
        res[e] = clean[e]
    return res.reset_index(drop=True), explain


def evaluate_labels(res):
    """Compares model predictions with a risk_level column supplied by the user."""
    if "actual_risk" not in res.columns:
        return None
    sub = res.dropna(subset=["actual_risk"])
    if sub.empty:
        return None
    cm = pd.crosstab(sub["actual_risk"], sub["risk_level"]).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    high = cm.loc["High"].sum()
    return {
        "n": int(len(sub)),
        "accuracy": float((sub["actual_risk"] == sub["risk_level"]).mean()),
        "high_recall": float(cm.loc["High", "High"] / high) if high else None,
        "cm": cm,
    }


def result_from_batch_row(r, explain):
    d = row_to_d(r)
    return {
        "level": r["risk_level"],
        "score": int(r["risk_score"]),
        "probs": {"Low": float(r["p_low"]), "Moderate": float(r["p_moderate"]), "High": float(r["p_high"])},
        "contributions": {f: float(r[f"effect_{f}"]) for f in FACTORS},
        "indicators": rule_contributions(d),
        "inputs": d,
        "date": date.today(),
        "source": "model",
        "explain": explain,
    }


def process_upload(raw, filename):
    ref = training_reference()
    cleaned = clean_upload(raw, ref)
    if "error" in cleaned:
        return cleaned
    if cleaned["clean"].empty:
        return {"empty": True, **cleaned}
    results, explain = run_batch(cleaned["clean"], cleaned["extras"])
    ev = evaluate_labels(results)
    meta = {
        "filename": filename,
        "created": datetime.now().isoformat(timespec="seconds"),
        "explain": explain,
        "counts": cleaned["counts"],
        "reason_counts": cleaned["reason_counts"],
        "used_training_reference": cleaned["used_training_reference"],
        "eval": None if ev is None else {k: v for k, v in ev.items() if k != "cm"},
        "truncated": False,
    }
    return {"results": results, "meta": meta, "rejected": cleaned["rejected"], "eval_cm": None if ev is None else ev["cm"]}


def save_batch(user_id, batch):
    q(
        "INSERT INTO batch_uploads (user_id, created_at, filename, n_rows, meta, results_csv) VALUES (?,?,?,?,?,?)",
        (user_id, batch["meta"]["created"], batch["meta"]["filename"], int(len(batch["results"])),
         json.dumps(batch["meta"], default=float), batch["results"].to_csv(index=False)),
        write=True,
    )


def load_latest_batch(user_id):
    row = q("SELECT * FROM batch_uploads WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,), one=True)
    if not row:
        return None
    try:
        res = pd.read_csv(io.StringIO(row["results_csv"]), dtype={"household_id": str})
        meta = json.loads(row["meta"])
    except Exception:
        return None
    cm = None
    if meta.get("eval") and "actual_risk" in res.columns:
        ev = evaluate_labels(res)
        cm = None if ev is None else ev["cm"]
    return {"results": res, "meta": meta, "rejected": None, "eval_cm": cm}


# ===================================================================== PDF
def _t(s):
    s = str(s)
    for a, b in (("–", "-"), ("—", "-"), ("·", "-"), ("’", "'"), ("•", "-"), ("→", "->")):
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _t(f"{DISCLAIMER}   |   Page {self.page_no()}"), align="C")


_LV_RGB = {"Low": (22, 163, 74), "Moderate": (245, 158, 11), "High": (220, 38, 38)}


def _pdf_start(title, subtitle):
    pdf = _PDF()
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    pdf.set_fill_color(15, 118, 110)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_xy(12, 7)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, _t(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _t(subtitle), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(36)
    pdf.set_text_color(31, 41, 55)
    return pdf


def _pdf_heading(pdf, text):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 118, 110)
    pdf.cell(0, 7, _t(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 220, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "", 10)


def _pdf_bar(pdf, label, frac, rgb, text, label_w=34):
    y = pdf.get_y()
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(label_w, 6, _t(label))
    x = pdf.get_x()
    pdf.set_fill_color(230, 232, 235)
    pdf.rect(x, y + 1.6, 100, 3, "F")
    pdf.set_fill_color(*rgb)
    pdf.rect(x, y + 1.6, max(0.5, 100 * frac), 3, "F")
    pdf.set_x(x + 104)
    pdf.cell(0, 6, _t(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(user, r, assessments, recs, plan):
    L = dict(new_x="LMARGIN", new_y="NEXT")
    pdf = _pdf_start(
        "Food Security Risk Assessment Report",
        f"Prepared for {user['name']} ({user['email']})   |   {r['date']:%d %b %Y}",
    )
    pdf.set_fill_color(*_LV_RGB[r["level"]])
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 14, _t(f"   {r['level']} risk   |   Risk score: {r['score']} / 100"), fill=True, **L)
    pdf.set_text_color(31, 41, 55)

    _pdf_heading(pdf, "Prediction probabilities")
    for lvl in LEVELS:
        _pdf_bar(pdf, lvl, r["probs"][lvl], _LV_RGB[lvl], f"{r['probs'][lvl] * 100:.0f}%", 28)

    _pdf_heading(pdf, "Household information")
    i = r["inputs"]
    info = [
        ("Household size", i["size"]),
        ("Monthly income (BDT)", f"{i['income']:,.0f}"),
        ("Monthly food expenditure (BDT)", f"{i['food_exp']:,.0f}"),
        ("Meals per day", int(i["meals"])),
        ("Dietary diversity score (0-12)", f"{i['diversity']:.0f}"),
        ("Employment", i["employment"]),
        ("Child under 5", "Yes" if i["child"] else "No"),
        ("Difficulty accessing food (1-5)", i["access"]),
    ]
    for k, v in info:
        pdf.cell(75, 6, _t(k))
        pdf.cell(0, 6, _t(v), **L)

    _pdf_heading(pdf, "Risk factor analysis (red raises risk, green lowers it)")
    contrib = r["contributions"]
    mx = max((abs(v) for v in contrib.values()), default=1) or 1
    for f in sorted(FACTORS, key=lambda k: -contrib[k]):
        v = contrib[f]
        y = pdf.get_y()
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(88, 6, _t(f"{f}: {describe_input(i, f)}"))
        cx, w = 150, 45 * abs(v) / mx
        if v > 0:
            pdf.set_fill_color(220, 38, 38)
            pdf.rect(cx, y + 1.6, max(w, 0.5), 3, "F")
        else:
            pdf.set_fill_color(22, 163, 74)
            pdf.rect(cx - max(w, 0.5), y + 1.6, max(w, 0.5), 3, "F")
        pdf.set_draw_color(150, 150, 150)
        pdf.line(cx, y, cx, y + 6)
        pdf.set_y(y + 6)
    pdf.set_font("Helvetica", "", 10)

    _pdf_heading(pdf, "Recommendations")
    drivers = risk_drivers(contrib, 3)
    if drivers:
        for f in drivers:
            pdf.multi_cell(0, 5.5, _t(f"- {f}: {ADVICE[f]}"), **L)
    else:
        pdf.multi_cell(0, 5.5, "- No major risk factors found. Keep up your current food habits.", **L)

    if recs:
        _pdf_heading(pdf, "Suggested affordable foods")
        for item in recs:
            row = item["row"]
            why = ", ".join(TAG_LABEL[t] for t in item["matched"]) or "good value for money"
            pdf.multi_cell(
                0, 5.5,
                _t(f"- {row['Food']} ({row['Category']}): {row['Price (BDT/kg)']:.0f} BDT/kg, "
                   f"{row['Protein per 10 BDT (g)']:.1f} g protein per 10 BDT - {why}"),
                **L,
            )

    if plan:
        _pdf_heading(pdf, "Budget meal plan (targets: 2100 kcal and 50 g protein per person)")
        pdf.multi_cell(
            0, 5.5,
            _t(f"Household of {plan['people']}: minimum cost about {plan['total']:.0f} BDT per day "
               f"({plan['total'] * 7:.0f} BDT per week)."),
            **L,
        )
        for _, row in plan["plan"].iterrows():
            pdf.cell(0, 5.5, _t(f"- {row['Food']}: {row['Grams per day']} g/day ({row['Cost per day (BDT)']:.1f} BDT)"), **L)

    _pdf_heading(pdf, "Risk trend")
    for a in assessments[-8:]:
        pdf.cell(0, 5.5, _t(f"{a['date']:%d %b %Y}:  score {a['score']}  ({a['level']})"), **L)
    if len(assessments) >= 2:
        diff = assessments[-1]["score"] - assessments[0]["score"]
        word = "improved" if diff < 0 else "worsened" if diff > 0 else "not changed"
        pdf.ln(1)
        pdf.multi_cell(0, 5.5, _t(f"Since the first assessment the score has {word} by {abs(diff)} points."), **L)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, _t(DISCLAIMER + " Predictions come from a machine-learning model trained on the project's training data."), **L)
    return bytes(pdf.output())


def build_batch_pdf(user, batch):
    L = dict(new_x="LMARGIN", new_y="NEXT")
    res, meta = batch["results"], batch["meta"]
    c = meta["counts"]
    pdf = _pdf_start(
        "Community Food Security Screening Report",
        f"Prepared for {user['name']}   |   File: {meta['filename']}   |   {meta['created'][:10]}",
    )
    n = len(res)
    _pdf_heading(pdf, "Summary")
    pdf.multi_cell(
        0, 5.5,
        _t(f"{n} households were screened with the trained model. {c['n_rejected']} of {c['n_raw']} uploaded rows "
           f"were rejected during data checks."),
        **L,
    )
    pdf.ln(1)
    for lvl in LEVELS:
        k = int((res["risk_level"] == lvl).sum())
        _pdf_bar(pdf, lvl, k / n, _LV_RGB[lvl], f"{k} households ({k / n * 100:.0f}%)")
    pdf.cell(0, 6, _t(f"Average risk score: {res['risk_score'].mean():.0f} / 100"), **L)

    ref = training_reference()
    if ref:
        _pdf_heading(pdf, "Comparison with the training population")
        for lvl in LEVELS:
            up = (res["risk_level"] == lvl).mean() * 100
            _pdf_bar(pdf, lvl, up / 100, _LV_RGB[lvl], f"uploaded {up:.0f}%  |  training {ref['risk_pct'][lvl]:.0f}%")

    _pdf_heading(pdf, "Data quality checks")
    for label, key in (
        ("Rows uploaded", "n_raw"), ("Rows analysed", "n_used"), ("Rows rejected", "n_rejected"),
        ("Rows with missing values filled from training data", "n_filled_rows"),
        ("Rows with values adjusted to the valid range", "n_adjusted_rows"),
        ("Households outside the training range (kept, flagged)", "n_unusual"),
    ):
        pdf.cell(125, 6, _t(label))
        pdf.cell(0, 6, str(c[key]), **L)
    if meta.get("eval"):
        e = meta["eval"]
        pdf.ln(1)
        rec = "n/a" if e["high_recall"] is None else f"{e['high_recall'] * 100:.0f}%"
        pdf.multi_cell(
            0, 5.5,
            _t(f"Check against the risk_level column you supplied ({e['n']} rows): accuracy {e['accuracy'] * 100:.0f}%, "
               f"High-risk recall {rec}."),
            **L,
        )

    _pdf_heading(pdf, "Main risk drivers among High-risk households")
    hi = res[res["risk_level"] == "High"]
    if len(hi):
        counts = hi["top_risk_driver"].value_counts()
        for f, k in counts.head(6).items():
            _pdf_bar(pdf, SHORT.get(f, f), k / len(hi), (220, 38, 38), f"{k} households", 38)
    else:
        pdf.cell(0, 6, "No households were classified as High risk.", **L)

    _pdf_heading(pdf, "Priority households (highest probability of High risk)")
    widths = [38, 22, 28, 22, 80]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 240, 235)
    for w, h in zip(widths, ["Household", "Risk", "P(High)", "Score", "Main driver"]):
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 9)
    top = res.sort_values(["p_high", "risk_score"], ascending=False).head(15)
    for _, r in top.iterrows():
        vals = [str(r["household_id"])[:20], r["risk_level"], f"{r['p_high'] * 100:.0f}%", str(int(r["risk_score"])),
                SHORT.get(r["top_risk_driver"], r["top_risk_driver"])]
        for w, v in zip(widths, vals):
            pdf.cell(w, 6, _t(v), border=1)
        pdf.ln(6)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, _t(DISCLAIMER + " Results help prioritise households for follow-up assessment; they do not replace a field visit."), **L)
    return bytes(pdf.output())


# =============================================================== small helpers
def gauge(score, height=280):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f172a", "thickness": 0.25},
                "steps": [
                    {"range": [0, 29], "color": "#bbf7d0"},
                    {"range": [29, 57], "color": "#fde68a"},
                    {"range": [57, 100], "color": "#fecaca"},
                ],
            },
        )
    )
    fig.update_layout(height=height, margin=dict(t=30, b=10, l=30, r=30))
    return fig


def require_result(key):
    r = st.session_state["last"]
    if r is None:
        st.info("No assessment yet. Complete the assessment first.", icon=":material/info:")
        st.button("Start assessment", icon=":material/arrow_forward:", on_click=goto, args=("Assessment",), key=f"need_{key}")
    return r


def explain_caption(r):
    if r.get("explain") == "SHAP":
        return "Computed with **SHAP** from the trained model."
    return "Approximate explanation from the indicator points (the `shap` library is not installed)."


# ===================================================================== auth
def page_auth():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="auth-hero">
              <div class="logo">🍚</div>
              <div class="a-title">Food Security<br>Risk Assessment</div>
              <div class="a-sub">Screen households for food-security and malnutrition risk, understand what drives
              the result and plan affordable, nutritious meals.</div>
              <ul>
                <li>🔍&nbsp; Risk screening in under a minute</li>
                <li>📤&nbsp; Upload a CSV and screen many households at once</li>
                <li>🧠&nbsp; See which factors drive every result</li>
                <li>🥗&nbsp; Budget meal plans that meet nutrition targets</li>
              </ul>
              <span class="tag">SDG 3 · Good Health and Well-being</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        with st.container(border=True):
            st.markdown("### Welcome 👋")
            st.caption("Log in or create a free account to continue.")
            flash = st.session_state.pop("flash", None)
            if flash:
                st.success(flash)

            t_login, t_signup, t_reset = st.tabs(
                [":material/login: Log in", ":material/person_add: Sign up", ":material/lock_reset: Forgot password"]
            )
            with t_login:
                with st.form("login_form"):
                    email = st.text_input("Email", placeholder="you@example.com", key="li_email")
                    pw = st.text_input("Password", type="password", key="li_pw")
                    submitted = st.form_submit_button("Log in", icon=":material/login:", type="primary")
                if submitted:
                    row = authenticate(email, pw)
                    if row:
                        login_user(row)
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")

            with t_signup:
                with st.form("signup_form"):
                    name = st.text_input("Full name", placeholder="Your name", key="su_name")
                    email = st.text_input("Email", placeholder="you@example.com", key="su_email")
                    c1, c2 = st.columns(2)
                    pw = c1.text_input("Password", type="password", key="su_pw")
                    pw2 = c2.text_input("Confirm password", type="password", key="su_pw2")
                    ans = st.text_input(
                        f"Security question: {SEC_QUESTION}",
                        help="Used to reset your password if you forget it.",
                        key="su_ans",
                    )
                    submitted = st.form_submit_button("Create account", icon=":material/person_add:", type="primary")
                if submitted:
                    if pw != pw2:
                        st.error("Passwords do not match.")
                    else:
                        ok, result = create_user(name, email, pw, ans)
                        if ok:
                            login_user(q("SELECT * FROM users WHERE id=?", (result,), one=True))
                            st.rerun()
                        else:
                            st.error(result)

            with t_reset:
                with st.form("reset_form"):
                    email = st.text_input("Email", placeholder="you@example.com", key="rs_email")
                    ans = st.text_input(f"{SEC_QUESTION} (answer you gave at sign-up)", key="rs_ans")
                    new_pw = st.text_input("New password", type="password", key="rs_pw")
                    submitted = st.form_submit_button("Reset password", icon=":material/lock_reset:", type="primary")
                if submitted:
                    ok, msg = reset_password(email, ans, new_pw)
                    if ok:
                        st.session_state["flash"] = msg
                        st.rerun()
                    else:
                        st.error(msg)
            st.caption(DISCLAIMER)


# =================================================================== sidebar
NAV_SECTIONS = [
    ("Start", [("Home", ":material/home:"), ("Assessment", ":material/assignment:"), ("Upload Data", ":material/upload_file:")]),
    (
        "Analyse",
        [
            ("Risk Prediction", ":material/insights:"),
            ("Risk Factors", ":material/analytics:"),
            ("Food Recommendations", ":material/eco:"),
            ("Meal Planner", ":material/restaurant:"),
        ],
    ),
    (
        "Track & report",
        [("Trends", ":material/trending_up:"), ("Dashboard", ":material/dashboard:"), ("Report", ":material/picture_as_pdf:")],
    ),
    ("Account", [("Profile", ":material/account_circle:")]),
]


def render_sidebar():
    u = st.session_state["user"]
    initial = html.escape(u["name"].strip()[:1].upper() or "U")
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="b-logo">🍚</div><div><div class="b-name">Food Security</div>'
            '<div class="b-sub">Risk Assessment</div></div></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                f'<div class="user-row"><div class="avatar">{initial}</div><div>'
                f'<div class="u-name">{html.escape(u["name"])}</div>'
                f'<div class="u-mail">{html.escape(u["email"])}</div></div></div>',
                unsafe_allow_html=True,
            )
        for section, items in NAV_SECTIONS:
            st.markdown(f'<div class="sec-label">{section}</div>', unsafe_allow_html=True)
            for label, icon in items:
                wide_button(
                    label,
                    icon=icon,
                    key=f"nav_{label}",
                    on_click=goto,
                    args=(label,),
                    type="primary" if st.session_state["page"] == label else "secondary",
                )
        st.divider()
        wide_button("Log out", icon=":material/logout:", on_click=logout, key="logout_btn")
        st.caption(DISCLAIMER)


# ===================================================================== pages
FEATURE_CARDS = [
    ("Assessment", "📝", "Food Security Assessment", "Enter one household's income, food spending, meals and diet details."),
    ("Upload Data", "📤", "Upload Data (CSV)", "Upload many households. The data is cleaned with the training data and scored by the model."),
    ("Risk Prediction", "🎯", "Risk Prediction & Score", "Get a Low / Moderate / High risk level, probabilities and a 0–100 score."),
    ("Risk Factors", "🔍", "Risk Factor Analysis", "See which inputs pushed the risk up or down (SHAP)."),
    ("Food Recommendations", "🥬", "Food Recommendations", "Affordable, nutritious foods matched to the risk factors."),
    ("Meal Planner", "🍽️", "Budget Meal Planner", "The cheapest meal plan within a daily or weekly budget."),
    ("Trends", "📈", "Risk Trend Tracking", "Save each assessment and compare results over time."),
    ("Dashboard", "🧭", "Dashboard", "Personal and uploaded-data charts and key numbers."),
    ("Report", "📄", "Report", "Download a structured PDF report."),
]


def page_home():
    u = st.session_state["user"]
    first = html.escape(u["name"].strip().split()[0])
    st.markdown(
        f'<div class="hero"><div class="h-title">Hello, {first} 👋</div>'
        '<div class="h-sub">Check a household\'s food security, or upload a CSV to screen many households at once.</div></div>',
        unsafe_allow_html=True,
    )
    assessments = load_assessments(u["id"])
    last = st.session_state["last"]
    batch = st.session_state["batch"]
    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Assessments", len(assessments))
    with c2.container(border=True):
        st.metric("Latest risk level", last["level"] if last else "—")
    with c3.container(border=True):
        st.metric("Latest score", f"{last['score']}/100" if last else "—")
    with c4.container(border=True):
        st.metric("Uploaded households", len(batch["results"]) if batch else "—")

    if last is None and batch is None:
        st.info("Nothing here yet. Start with an assessment or upload a CSV file.", icon=":material/info:")

    st.subheader("All features")
    for start in (0, 3, 6):
        cols = st.columns(3)
        for col, (page, emoji, title, desc) in zip(cols, FEATURE_CARDS[start:start + 3]):
            with col.container(border=True):
                st.markdown(
                    f'<div class="feature-icon">{emoji}</div><div class="f-title">{title}</div>'
                    f'<div class="f-desc">{html.escape(desc)}</div>',
                    unsafe_allow_html=True,
                )
                st.button("Open", icon=":material/arrow_forward:", on_click=goto, args=(page,), key=f"home_{page}")


def page_assessment():
    st.title("📝 Food Security Assessment")
    st.caption("Fill in every field and press **Assess risk**.")
    err = st.session_state.get("form_error")
    if err:
        st.error(err, icon=":material/error:")
    with st.container(border=True):
        with st.form("assess", border=False):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("##### 🏠 Household")
                st.number_input("Household size", 1, 30, value=None, step=1, placeholder="Number of people", key="in_size")
                st.number_input("Monthly household income (BDT)", 0, 10_000_000, value=None, step=500,
                                placeholder="Total monthly income", key="in_income")
                st.number_input("Monthly food expenditure (BDT)", 0, 10_000_000, value=None, step=500,
                                placeholder="Monthly spending on food", key="in_food")
                st.selectbox("Employment", ["Regular", "Daily wage", "Unemployed"], index=None,
                             placeholder="Choose one", key="in_emp")
            with c2:
                st.markdown("##### 🍽️ Food & nutrition")
                st.selectbox("Meals per day", [1, 2, 3, 4], index=None, placeholder="Choose one", key="in_meals")
                st.selectbox("Dietary diversity score (food groups eaten, 0–12)", list(range(13)), index=None,
                             placeholder="Choose one", key="in_div",
                             help="Count how many of the 12 food groups the household ate in the last 24 hours.")
                st.selectbox("Difficulty accessing food", [1, 2, 3, 4, 5], index=None, placeholder="Choose one",
                             format_func=lambda x: ACCESS_LABEL[x], key="in_access")
                st.radio("Child under 5 in the household?", ["No", "Yes"], index=None, horizontal=True, key="in_child")
            st.form_submit_button("Assess risk", on_click=run_assessment, icon=":material/insights:", type="primary")
    st.caption(DISCLAIMER)


def page_prediction():
    st.title("🎯 Risk Prediction & Score")
    r = require_result("pred")
    if r is None:
        return
    icon, grad, msg = LEVEL_STYLE[r["level"]]
    st.markdown(
        f'<div class="result-hero" style="background:{grad}"><div class="rh-icon">{icon}</div><div>'
        f'<div class="rh-title">{r["level"]} risk</div><div class="rh-sub">{msg}</div></div></div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("Risk score (0–100)", r["score"])
    with m2.container(border=True):
        st.metric("Prediction confidence", f"{max(r['probs'].values()) * 100:.0f}%")
    with m3.container(border=True):
        st.metric("Assessment date", r["date"].strftime("%d %b %Y"))

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("##### Food Security Risk Score")
        st.plotly_chart(gauge(r["score"]))
        st.caption("One overall score built from the 7 indicators below.")
    with c2.container(border=True):
        pdf = pd.DataFrame({"Risk level": LEVELS, "Probability (%)": [r["probs"][k] * 100 for k in LEVELS]})
        fig = px.bar(pdf, x="Risk level", y="Probability (%)", color="Risk level", color_discrete_map=COLORS)
        fig.update_layout(showlegend=False, yaxis_range=[0, 100], title="Model probability",
                          height=360, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig)

    with st.container(border=True):
        st.markdown("##### Score indicators")
        ind = pd.DataFrame(
            {
                "Indicator": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS],
                "Risk (0–100%)": [r["indicators"][f] / MAX_RAW[f] * 100 for f in FACTORS],
                "Your value": [describe_input(r["inputs"], f) for f in FACTORS],
            }
        ).sort_values("Risk (0–100%)")
        fig = px.bar(ind, x="Risk (0–100%)", y="Indicator", orientation="h", color="Risk (0–100%)",
                     color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"], range_color=[0, 100],
                     hover_data=["Your value"])
        fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_range=[0, 100])
        st.plotly_chart(fig)
    st.caption("The risk level and probabilities come from the trained model. The 0–100 score is built from the 7 indicators above.")

    b1, b2, b3, _ = st.columns([1, 1, 1, 1])
    b1.button("Why this result?", icon=":material/analytics:", on_click=goto, args=("Risk Factors",), key="p_rf")
    b2.button("Food suggestions", icon=":material/eco:", on_click=goto, args=("Food Recommendations",), key="p_fr")
    b3.button("Get PDF report", icon=":material/picture_as_pdf:", on_click=goto, args=("Report",), key="p_rep")
    st.caption(DISCLAIMER)


def page_risk_factors():
    st.title("🔍 Risk Factor Analysis")
    r = require_result("rf")
    if r is None:
        return
    st.caption("Shows how much each input **raised** (red) or **lowered** (green) the risk compared with an average household. "
               + explain_caption(r))
    contrib = r["contributions"]
    df = pd.DataFrame(
        {"Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS], "key": FACTORS, "Effect": [contrib[f] for f in FACTORS]}
    )
    df["Direction"] = np.where(df["Effect"] > 0, "Raises risk", "Lowers risk")
    df = df.sort_values("Effect")

    with st.container(border=True):
        fig = px.bar(df, x="Effect", y="Factor", orientation="h", color="Direction",
                     color_discrete_map={"Raises risk": "#dc2626", "Lowers risk": "#16a34a"},
                     title="What pushed the risk up or down")
        fig.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
        fig.add_vline(x=0, line_color="#6b7280")
        st.plotly_chart(fig)

    mx = max(abs(v) for v in contrib.values()) or 1
    table = pd.DataFrame(
        {
            "Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS],
            "Value": [describe_input(r["inputs"], f) for f in FACTORS],
            "Effect": ["⬆ raises risk" if contrib[f] > 0 else "⬇ lowers risk" for f in FACTORS],
            "Strength": [abs(contrib[f]) / mx * 100 for f in FACTORS],
        }
    ).sort_values("Strength", ascending=False)
    c1, c2 = st.columns([1.6, 1])
    with c1.container(border=True):
        st.markdown("##### Factor details")
        st.dataframe(
            table, hide_index=True,
            column_config={"Strength": st.column_config.ProgressColumn("Strength", min_value=0, max_value=100, format="%.0f%%")},
        )
    with c2.container(border=True):
        st.markdown("##### Biggest drivers")
        drivers = risk_drivers(contrib, 3)
        if drivers:
            for f in drivers:
                st.markdown(f"**{FACTOR_ICON[f]} {f}** — {describe_input(r['inputs'], f)}")
                st.caption(ADVICE[f])
        else:
            st.success("Nothing is pushing the risk above an average household.")

    imp = global_importance()
    if imp is not None:
        with st.container(border=True):
            imp = imp.sort_values("Mean |effect|")
            fig = px.bar(imp, x="Mean |effect|", y="Factor", orientation="h",
                         title="Which factors matter most overall (SHAP over the training data)")
            fig.update_traces(marker_color="#0f9d58")
            fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig)
    st.button("Get food suggestions", icon=":material/eco:", on_click=goto, args=("Food Recommendations",), key="rf_fr")
    st.caption(DISCLAIMER)


def page_food_recommendations():
    st.title("🥬 Personalised Food Recommendations")
    foods = load_foods()
    if foods is None:
        foods_missing_notice()
        return
    r = st.session_state["last"]
    if r is None:
        st.info("Showing general low-cost, nutritious foods. Complete an assessment to personalise this list.",
                icon=":material/info:")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.4])
        n = c1.slider("How many foods", 3, 9, 6, key="fr_n")
        top_price = int(foods["Price (BDT/kg)"].max())
        max_price = c2.slider("Max price (BDT/kg)", 20, max(top_price, 30), max(top_price, 30), step=10, key="fr_price")
        excl = c3.multiselect("Exclude foods (allergy / preference)", list(foods["Food"]), key="fr_excl")

    picks, drivers, tags = recommend_foods(r, n=n, max_price=max_price, exclude=excl)
    if r:
        chips = "".join(f'<span class="chip red">{FACTOR_ICON[f]} {html.escape(f)}</span>' for f in drivers)
        need = "".join(f'<span class="chip">{html.escape(TAG_LABEL[t])}</span>' for t in tags)
        st.markdown(
            "**Chosen because of the risk factors:** " + (chips or '<span class="chip">no major risk factors</span>')
            + "<br>**Foods that help:** " + (need or '<span class="chip">best value for money</span>'),
            unsafe_allow_html=True,
        )
    if not picks:
        st.warning("No foods match these filters. Try raising the max price or removing exclusions.")
        return

    for start in range(0, len(picks), 3):
        cols = st.columns(3)
        for col, item in zip(cols, picks[start:start + 3]):
            row = item["row"]
            with col.container(border=True):
                emoji = CATEGORY_EMOJI.get(row["Category"], "🍽️")
                st.markdown(
                    f'<div class="food-emoji">{emoji}</div><div class="food-name">{html.escape(row["Food"])}</div>'
                    f'<div class="food-meta">{html.escape(row["Category"])} · {row["Price (BDT/kg)"]:.0f} BDT/kg</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Protein / 10 BDT", f"{row['Protein per 10 BDT (g)']:.1f} g")
                m2.metric("kcal / 10 BDT", f"{row['kcal per 10 BDT']:.0f}")
                chips = "".join(f'<span class="chip">{html.escape(TAG_LABEL[t])}</span>' for t in item["matched"][:3])
                if item["value"] >= 0.67:
                    chips += '<span class="chip warn">best value</span>'
                st.markdown(chips or '<span class="chip">balanced choice</span>', unsafe_allow_html=True)

    with st.expander("See the full food table"):
        st.dataframe(foods.drop(columns=["Tags"]), hide_index=True)
    b1, _ = st.columns([1, 3])
    b1.button("Build a meal plan", icon=":material/restaurant:", on_click=goto, args=("Meal Planner",), key="fr_mp")
    st.caption("Prices and nutrition values come from foods.csv and are approximate. " + DISCLAIMER)


def page_meal_planner():
    st.title("🍽️ Budget Meal Planner")
    foods = load_foods()
    if foods is None:
        foods_missing_notice()
        return
    st.caption("Finds the **cheapest combination of foods** that meets your calorie and protein targets, "
               "then splits it into breakfast, lunch and dinner.")
    r = st.session_state["last"]
    default_people = int(r["inputs"]["size"]) if r else None

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        people = c1.number_input("People in household", 1, 30, value=default_people, placeholder="Number of people", key="mp_people")
        period = c2.radio("Budget period", ["Daily", "Weekly"], horizontal=True, key="mp_period")
        budget = c3.number_input(f"{period} food budget (BDT), optional", 0, 1_000_000, value=None, step=50,
                                 placeholder="Enter your budget to compare", key="mp_budget")
        c4, c5, c6, c7 = st.columns([1, 1, 1, 1.4])
        kcal_pp = c4.slider("Calories per person", 1500, 2800, 2100, step=50, key="mp_kcal")
        prot_pp = c5.slider("Protein per person (g)", 30, 90, 50, step=5, key="mp_prot")
        variety = c6.checkbox("Include vegetables & fruit", value=True, key="mp_var")
        excl = c7.multiselect("Exclude foods", list(foods["Food"]), key="mp_excl")

    if not people:
        st.info("Enter the number of people in the household to build a plan.", icon=":material/info:")
        return
    result = compute_plan(int(people), kcal_pp, prot_pp, variety, tuple(excl))
    if result is None:
        st.error("No plan can meet these targets with the available foods. Try lowering the targets or removing exclusions.")
        return
    total = result["total"]

    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("Minimum cost per day", f"{total:.0f} BDT")
    with m2.container(border=True):
        st.metric("Minimum cost per week", f"{total * 7:.0f} BDT")
    with m3.container(border=True):
        st.metric("Cost per person per day", f"{total / people:.0f} BDT")
    if budget:
        daily_budget = budget if period == "Daily" else budget / 7.0
        if total <= daily_budget:
            st.success(f"Your budget ({daily_budget:.0f} BDT per day) covers this plan with about {daily_budget - total:.0f} BDT per day to spare.")
        else:
            st.warning(f"Your budget ({daily_budget:.0f} BDT per day) is about {total - daily_budget:.0f} BDT per day short of the cheapest plan that meets the targets.")

    t1, t2, t3 = st.tabs([":material/restaurant: Daily meals", ":material/shopping_cart: Shopping list", ":material/bar_chart: Cost & nutrition"])
    with t1:
        st.caption("Amounts are per person, per day.")
        cols = st.columns(3)
        for col, (meal, emoji) in zip(cols, MEALS):
            with col.container(border=True):
                items = result["meals"][meal]
                body = "".join(
                    f'<div class="meal-item"><span>{html.escape(name)}</span><b>{g} g</b></div>' for name, g in items
                ) or "<i>No food assigned</i>"
                st.markdown(
                    f'<div class="meal-title">{emoji} {meal}</div>'
                    f'<div class="meal-kcal">about {result["meal_kcal"][meal]:.0f} kcal per person</div>{body}',
                    unsafe_allow_html=True,
                )
    with t2:
        plan = result["plan"].copy()
        shop = pd.DataFrame(
            {
                "Food": plan["Food"],
                "Category": plan["Category"],
                "Per day (g)": plan["Grams per day"],
                "Per week (kg)": (plan["Grams per day"] * 7 / 1000).round(2),
                "Cost per week (BDT)": (plan["Cost per day (BDT)"] * 7).round(0).astype(int),
            }
        )
        st.dataframe(shop, hide_index=True)
        st.markdown(f"**Estimated weekly total: {shop['Cost per week (BDT)'].sum():,} BDT** for {int(people)} people")
        st.download_button("Download shopping list", shop.to_csv(index=False), file_name="shopping_list.csv",
                           icon=":material/download:")
    with t3:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            fig = px.pie(result["plan"], names="Food", values="Cost per day (BDT)", hole=0.45, title="Where the money goes")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig)
        with c2.container(border=True):
            nut = pd.DataFrame(
                {
                    "Nutrient": ["Calories (kcal)", "Calories (kcal)", "Protein (g)", "Protein (g)"],
                    "Type": ["Target", "Plan", "Target", "Plan"],
                    "Per person": [kcal_pp, result["kcal_pp"], prot_pp, result["protein_pp"]],
                }
            )
            fig = px.bar(nut, x="Nutrient", y="Per person", color="Type", barmode="group",
                         color_discrete_map={"Target": "#9ca3af", "Plan": "#0f9d58"}, title="Nutrition per person per day")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig)
    st.caption("Prices and nutrition values come from foods.csv and are approximate. " + DISCLAIMER)


def page_trends():
    st.title("📈 Risk Trend Tracking")
    u = st.session_state["user"]
    assessments = load_assessments(u["id"])
    if not assessments:
        st.info("No assessments saved yet. Every completed assessment is saved here with its date.", icon=":material/info:")
        st.button("Start assessment", icon=":material/arrow_forward:", on_click=goto, args=("Assessment",), key="tr_start")
        return
    hist = history_df(assessments)

    m1, m2, m3, m4 = st.columns(4)
    with m1.container(border=True):
        st.metric("Assessments", len(hist))
    with m2.container(border=True):
        st.metric("Best score", int(hist["Risk score"].min()))
    with m3.container(border=True):
        if len(hist) >= 2:
            st.metric("Latest vs previous", trend_text(int(hist["Risk score"].iloc[-1] - hist["Risk score"].iloc[-2])))
        else:
            st.metric("Latest score", int(hist["Risk score"].iloc[-1]))
    with m4.container(border=True):
        if len(hist) >= 2:
            st.metric("Since first", trend_text(int(hist["Risk score"].iloc[-1] - hist["Risk score"].iloc[0])))
        else:
            st.metric("Since first", "—")

    with st.container(border=True):
        fig = px.line(hist, x="Date", y="Risk score", markers=True, title="Risk score over time (lower is better)")
        fig.add_hline(y=29, line_dash="dot", annotation_text="Low / Moderate")
        fig.add_hline(y=57, line_dash="dot", annotation_text="Moderate / High")
        fig.update_traces(line=dict(color="#0f9d58", width=3))
        fig.update_layout(yaxis_range=[0, 100], margin=dict(t=50, b=20, l=10, r=10), height=380)
        st.plotly_chart(fig)

    st.subheader("Compare two assessments")
    if len(assessments) < 2:
        st.caption("Complete at least two assessments to compare them.")
    else:
        labels = [f"{a['date']:%d %b %Y} · score {a['score']} ({a['level']}) · #{a['id']}" for a in assessments]
        c1, c2 = st.columns(2)
        ia = labels.index(c1.selectbox("Earlier assessment", labels, index=0, key="cmp_a"))
        ib = labels.index(c2.selectbox("Later assessment", labels, index=len(labels) - 1, key="cmp_b"))
        a, b = assessments[ia], assessments[ib]
        d = b["score"] - a["score"]
        with st.container(border=True):
            k1, k2, k3 = st.columns(3)
            k1.metric("Earlier score", a["score"])
            k2.metric("Later score", b["score"])
            k3.metric("Result", "Improved" if d < 0 else "Worsened" if d > 0 else "No change")
            cmp = pd.DataFrame(
                {
                    "Indicator": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS],
                    "Earlier": [describe_input(a["inputs"], f) for f in FACTORS],
                    "Later": [describe_input(b["inputs"], f) for f in FACTORS],
                    "Risk change (pts)": [round((b["indicators"][f] - a["indicators"][f]) / MAX_RAW[f] * 100) for f in FACTORS],
                }
            )
            st.dataframe(cmp, hide_index=True)
            st.caption("Negative change = lower risk for that indicator.")

    with st.container(border=True):
        st.markdown("##### All assessments")
        show = hist.copy()
        show["Date"] = show["Date"].dt.date
        st.dataframe(show.sort_values("Date", ascending=False), hide_index=True)


# ------------------------------------------------------------ upload page
def read_uploaded_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None
    return None


def render_batch(batch):
    res, meta = batch["results"], batch["meta"]
    c = meta["counts"]
    st.success(f"Showing results for **{meta['filename']}** (processed {meta['created'].replace('T', ' ')}).",
               icon=":material/task_alt:")
    t_sum, t_tab, t_det, t_dq, t_rep = st.tabs(
        [":material/bar_chart: Summary", ":material/table: Results table", ":material/person_search: Household detail",
         ":material/fact_check: Data quality", ":material/picture_as_pdf: Report"]
    )
    ref = training_reference()
    n = len(res)

    with t_sum:
        k1, k2, k3, k4 = st.columns(4)
        with k1.container(border=True):
            st.metric("Households analysed", n)
        with k2.container(border=True):
            st.metric("High risk", f"{(res['risk_level'] == 'High').mean() * 100:.0f}%")
        with k3.container(border=True):
            st.metric("Moderate risk", f"{(res['risk_level'] == 'Moderate').mean() * 100:.0f}%")
        with k4.container(border=True):
            st.metric("Average risk score", f"{res['risk_score'].mean():.0f}")

        c1, c2 = st.columns(2)
        with c1.container(border=True):
            counts = res["risk_level"].value_counts().reindex(LEVELS).fillna(0).reset_index()
            counts.columns = ["Risk level", "Households"]
            fig = px.pie(counts, names="Risk level", values="Households", hole=0.5, color="Risk level",
                         color_discrete_map=COLORS, title="Risk level distribution")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)
        with c2.container(border=True):
            up = (res["risk_level"].value_counts(normalize=True) * 100).reindex(LEVELS).fillna(0)
            cmp = pd.DataFrame({"Risk level": LEVELS, "Uploaded data": up.values})
            title = "Uploaded data vs training data (%)"
            if ref:
                cmp["Training data"] = [ref["risk_pct"][k] for k in LEVELS]
            else:
                title = "Share of households by risk level (%)"
            cmp = cmp.melt(id_vars="Risk level", var_name="Dataset", value_name="% of households")
            fig = px.bar(cmp, x="Risk level", y="% of households", color="Dataset", barmode="group", title=title,
                         color_discrete_map={"Uploaded data": "#0f9d58", "Training data": "#9ca3af"})
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)

        c3, c4 = st.columns(2)
        with c3.container(border=True):
            hi = res[res["risk_level"] == "High"]
            if len(hi):
                drv = hi["top_risk_driver"].value_counts().rename_axis("Main driver").reset_index(name="High-risk households")
                drv = drv.sort_values("High-risk households")
                fig = px.bar(drv, x="High-risk households", y="Main driver", orientation="h",
                             title="Main risk driver of High-risk households")
                fig.update_traces(marker_color="#dc2626")
                fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
                st.plotly_chart(fig)
            else:
                st.markdown("##### Main risk driver")
                st.success("No household was classified as High risk.")
        with c4.container(border=True):
            e = res.groupby("employment")["risk_level"].apply(lambda s: (s == "High").mean() * 100).reset_index()
            e.columns = ["Employment", "High-risk households (%)"]
            fig = px.bar(e, x="Employment", y="High-risk households (%)", title="High risk by employment type")
            fig.update_traces(marker_color="#0f9d58")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)

        with st.container(border=True):
            st.markdown("##### Priority households (highest probability of High risk)")
            top = res.sort_values(["p_high", "risk_score"], ascending=False).head(20)
            show = top[["household_id", "risk_level", "p_high", "risk_score", "top_risk_driver", "income_per_person", "household_size"]].copy()
            show["p_high"] = show["p_high"] * 100
            show.columns = ["Household", "Risk level", "P(High) %", "Risk score", "Main driver", "Income per person", "Household size"]
            st.dataframe(show, hide_index=True,
                         column_config={"P(High) %": st.column_config.ProgressColumn("P(High) %", min_value=0, max_value=100, format="%.0f%%")})

        if batch.get("eval_cm") is not None and meta.get("eval"):
            e = meta["eval"]
            with st.container(border=True):
                st.markdown("##### Model check against your `risk_level` column")
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows with a label", e["n"])
                m2.metric("Accuracy", f"{e['accuracy'] * 100:.0f}%")
                m3.metric("High-risk recall", "n/a" if e["high_recall"] is None else f"{e['high_recall'] * 100:.0f}%")
                fig = px.imshow(batch["eval_cm"], text_auto=True, color_continuous_scale="Greens",
                                labels=dict(x="Predicted", y="Your label", color="Households"))
                fig.update_layout(height=320, margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig)

    with t_tab:
        with st.container(border=True):
            f1, f2, f3 = st.columns([1.2, 1.2, 1])
            lv = f1.multiselect("Risk level", LEVELS, default=LEVELS, key="bt_lv")
            emp = f2.multiselect("Employment", sorted(res["employment"].unique()), default=sorted(res["employment"].unique()), key="bt_emp")
            search = f3.text_input("Search household ID", key="bt_search")
        view = res[res["risk_level"].isin(lv) & res["employment"].isin(emp)]
        if search.strip():
            view = view[view["household_id"].astype(str).str.contains(search.strip(), case=False, regex=False)]
        show = view[["household_id", "risk_level", "p_high", "risk_score", "top_risk_driver", "household_size", "monthly_income",
                     "food_expenditure", "meals_per_day", "dietary_diversity", "employment", "child_under5", "food_access",
                     "data_quality"]].copy()
        show["p_high"] = show["p_high"] * 100
        show = show.rename(columns={
            "household_id": "Household", "risk_level": "Risk level", "p_high": "P(High) %", "risk_score": "Risk score",
            "top_risk_driver": "Main driver", "household_size": "Size", "monthly_income": "Income", "food_expenditure": "Food spend",
            "meals_per_day": "Meals", "dietary_diversity": "Diet diversity", "employment": "Employment",
            "child_under5": "Child <5", "food_access": "Access", "data_quality": "Data check",
        })
        st.caption(f"{len(show)} of {n} households shown")
        st.dataframe(show, hide_index=True,
                     column_config={"P(High) %": st.column_config.ProgressColumn("P(High) %", min_value=0, max_value=100, format="%.0f%%")})
        st.download_button("Download all results (CSV)", res.to_csv(index=False), file_name="household_risk_results.csv",
                           icon=":material/download:", key="dl_results")

    with t_det:
        ids = list(res["household_id"].astype(str))
        pick = st.selectbox("Choose a household", ids, index=None, placeholder="Select a household ID", key="bd_pick")
        if pick is None:
            st.info("Select a household to see its prediction, the factors behind it and food suggestions.", icon=":material/info:")
        else:
            row = res[res["household_id"].astype(str) == pick].iloc[0]
            hr = result_from_batch_row(row, meta.get("explain", "SHAP"))
            icon, grad, msg = LEVEL_STYLE[hr["level"]]
            st.markdown(
                f'<div class="result-hero" style="background:{grad}"><div class="rh-icon">{icon}</div><div>'
                f'<div class="rh-title">{hr["level"]} risk · household {html.escape(pick)}</div>'
                f'<div class="rh-sub">Risk score {hr["score"]}/100 · probability of High risk {hr["probs"]["High"] * 100:.0f}%</div></div></div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1.4, 1])
            with c1.container(border=True):
                df = pd.DataFrame({"Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS],
                                   "Effect": [hr["contributions"][f] for f in FACTORS]})
                df["Direction"] = np.where(df["Effect"] > 0, "Raises risk", "Lowers risk")
                fig = px.bar(df.sort_values("Effect"), x="Effect", y="Factor", orientation="h", color="Direction",
                             color_discrete_map={"Raises risk": "#dc2626", "Lowers risk": "#16a34a"},
                             title="What pushed this household's risk up or down")
                fig.update_layout(height=360, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
                fig.add_vline(x=0, line_color="#6b7280")
                st.plotly_chart(fig)
            with c2.container(border=True):
                st.markdown("##### Recommendations")
                drivers = risk_drivers(hr["contributions"], 3)
                if drivers:
                    for f in drivers:
                        st.markdown(f"**{FACTOR_ICON[f]} {f}** — {describe_input(hr['inputs'], f)}")
                        st.caption(ADVICE[f])
                else:
                    st.success("No major risk factors found.")
                st.caption("Data check: " + str(row["data_quality"]))
            b1, b2, _ = st.columns([1.2, 1.2, 2])

            def _use(page, hr=hr):
                st.session_state["last"] = hr
                st.session_state["page"] = page

            b1.button("Food suggestions", icon=":material/eco:", on_click=_use, args=("Food Recommendations",), key="bd_food")
            b2.button("Meal plan", icon=":material/restaurant:", on_click=_use, args=("Meal Planner",), key="bd_meal")

    with t_dq:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1.container(border=True):
            st.metric("Rows uploaded", c["n_raw"])
        with k2.container(border=True):
            st.metric("Rows analysed", c["n_used"])
        with k3.container(border=True):
            st.metric("Rows rejected", c["n_rejected"])
        with k4.container(border=True):
            st.metric("Values filled", c["n_filled_rows"])
        with k5.container(border=True):
            st.metric("Outside training range", c["n_unusual"])
        if meta.get("used_training_reference"):
            st.caption("Missing optional values were filled with the training-data median or most common value, and "
                       "households far outside the training population were flagged (they are kept in the results).")
        else:
            st.warning("ml_training.db was not found, so missing values could not be filled from the training data. "
                       "Rows with any missing value were rejected. Upload ml_training.db to your repository for full cleaning.",
                       icon=":material/warning:")
        if meta.get("reason_counts"):
            rc = pd.DataFrame(sorted(meta["reason_counts"].items(), key=lambda kv: -kv[1]), columns=["Why rows were rejected", "Rows"])
            st.dataframe(rc, hide_index=True)
        if batch.get("rejected") is not None and len(batch["rejected"]):
            with st.expander(f"Show the {len(batch['rejected'])} rejected rows"):
                st.dataframe(batch["rejected"], hide_index=True)
                st.download_button("Download rejected rows (CSV)", batch["rejected"].to_csv(index=False),
                                   file_name="rejected_rows.csv", icon=":material/download:", key="dl_rej")
        elif c["n_rejected"] and batch.get("rejected") is None:
            st.caption("The rejected rows are only available right after the upload is processed.")
        adj = res[res["data_quality"] != "OK"][["household_id", "data_quality"]]
        if len(adj):
            with st.expander(f"Show the {len(adj)} households with filled, adjusted or flagged values"):
                st.dataframe(adj.rename(columns={"household_id": "Household", "data_quality": "Data check"}), hide_index=True)

    with t_rep:
        st.markdown("The community report contains the risk distribution, data-quality checks, a comparison with the "
                    "training population, the main risk drivers and the 15 priority households.")
        c1, c2, _ = st.columns([1, 1, 2])
        if HAVE_FPDF:
            try:
                pdf_bytes = build_batch_pdf(st.session_state["user"], batch)
                c1.download_button("Download community report (PDF)", pdf_bytes, file_name="community_screening_report.pdf",
                                   mime="application/pdf", icon=":material/picture_as_pdf:", type="primary", key="dl_batch_pdf")
            except Exception as ex:
                st.error(f"Could not build the PDF: {ex}")
        else:
            st.warning("PDF export needs the `fpdf2` library. Add `fpdf2` to requirements.txt.")
        c2.download_button("Download results (CSV)", res.to_csv(index=False), file_name="household_risk_results.csv",
                           icon=":material/table_view:", key="dl_results2")
    st.caption(DISCLAIMER)


def process_clicked():
    s = st.session_state
    up = s.get("batch_file")
    if up is None:
        return
    raw = read_uploaded_csv(up)
    if raw is None or raw.empty:
        s["batch_error"] = "The file is empty or could not be read as a CSV."
        return
    truncated = len(raw) > MAX_UPLOAD_ROWS
    if truncated:
        raw = raw.head(MAX_UPLOAD_ROWS)
    out = process_upload(raw, up.name)
    if "error" in out:
        s["batch_error"] = (
            "Your file is missing these required columns: **" + ", ".join(out["error"]) + "**. "
            "Columns found: " + ", ".join(map(str, out["found"])) + ". Download the template and use its column names."
        )
        s["batch_rejected_only"] = None
        return
    if out.get("empty"):
        s["batch_error"] = "Every row was rejected by the data checks (see the reasons below)."
        s["batch_rejected_only"] = out
        return
    out["meta"]["truncated"] = truncated
    save_batch(s["user"]["id"], out)
    s["batch"] = out
    s["batch_error"] = None
    s["batch_rejected_only"] = None


def page_upload():
    st.title("📤 Upload Household Data")
    st.caption("Upload a CSV with many households. The app cleans it using the training data, scores every household with "
               "the trained model and explains the results.")
    st.markdown(
        '<div class="step-row"><div class="step"><b>1</b> Download the template</div>'
        '<div class="step"><b>2</b> Upload your CSV</div>'
        '<div class="step"><b>3</b> Data is checked and cleaned with the training data</div>'
        '<div class="step"><b>4</b> Model scores every household</div>'
        '<div class="step"><b>5</b> View, download and report</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        c1, c2 = st.columns([1, 2.2], gap="large")
        with c1:
            st.markdown("##### 1. Template")
            template = ",".join(["household_id"] + CANON + ["risk_level"]) + "\n"
            st.download_button("Download CSV template", template, file_name="household_template.csv",
                               icon=":material/download:", key="dl_template")
            st.caption("Use these column names. The `household_id` and `risk_level` columns are optional.")
        with c2:
            st.markdown("##### Column guide")
            st.dataframe(COLUMN_GUIDE, hide_index=True)

    with st.container(border=True):
        st.markdown("##### 2. Upload your file")
        st.file_uploader("Choose a CSV file", type=["csv"], key="batch_file", label_visibility="collapsed")
        st.button("Process file", icon=":material/play_arrow:", type="primary", on_click=process_clicked, key="batch_go",
                  disabled=st.session_state.get("batch_file") is None)
        st.caption(f"Up to {MAX_UPLOAD_ROWS:,} rows are processed. Nothing is shared outside your account.")

    err = st.session_state.get("batch_error")
    if err:
        st.error(err, icon=":material/error:")
        ro = st.session_state.get("batch_rejected_only")
        if ro is not None:
            rc = pd.DataFrame(sorted(ro["reason_counts"].items(), key=lambda kv: -kv[1]), columns=["Why rows were rejected", "Rows"])
            st.dataframe(rc, hide_index=True)
            st.dataframe(ro["rejected"].head(50), hide_index=True)

    batch = st.session_state.get("batch")
    if batch is not None:
        if batch["meta"].get("truncated"):
            st.warning(f"The file had more than {MAX_UPLOAD_ROWS:,} rows, so only the first {MAX_UPLOAD_ROWS:,} were processed.")
        render_batch(batch)
    elif not err:
        st.info("Upload a CSV file and press **Process file** to see the results.", icon=":material/info:")

    runs = model_runs()
    if runs is not None:
        with st.expander("About the model (training results from Colab)"):
            st.dataframe(runs, hide_index=True)
            st.caption("Scores on the held-out test data from the training notebook.")


# ----------------------------------------------------------------- dashboard
def page_dashboard():
    st.title("🧭 Dashboard")
    tab_me, tab_up = st.tabs([":material/person: My dashboard", ":material/upload_file: Uploaded data"])

    with tab_me:
        u = st.session_state["user"]
        r = st.session_state["last"]
        assessments = load_assessments(u["id"])
        if r is None:
            st.info("Complete an assessment to see your personal dashboard.", icon=":material/info:")
            st.button("Start assessment", icon=":material/arrow_forward:", on_click=goto, args=("Assessment",), key="db_start")
        else:
            hist = history_df(assessments)
            k1, k2, k3, k4 = st.columns(4)
            with k1.container(border=True):
                st.metric("Risk level", r["level"])
            with k2.container(border=True):
                st.metric("Risk score", r["score"])
            with k3.container(border=True):
                st.metric("Assessments", len(hist))
            with k4.container(border=True):
                top = risk_drivers(r["contributions"], 1)
                st.metric("Top risk factor", SHORT[top[0]] if top else "None")
            c1, c2 = st.columns(2)
            with c1.container(border=True):
                st.markdown("##### Score gauge")
                st.plotly_chart(gauge(r["score"], 300))
            with c2.container(border=True):
                rad = pd.DataFrame({"Indicator": FACTORS, "Risk %": [r["indicators"][f] / MAX_RAW[f] * 100 for f in FACTORS]})
                fig = px.line_polar(rad, r="Risk %", theta="Indicator", line_close=True, range_r=[0, 100],
                                    title="Risk profile (larger = riskier)")
                fig.update_traces(fill="toself", line_color="#dc2626")
                fig.update_layout(height=330, margin=dict(t=60, b=20, l=60, r=60))
                st.plotly_chart(fig)
            with st.container(border=True):
                fig = px.area(hist, x="Date", y="Risk score", markers=True, title="My score over time")
                fig.update_traces(line_color="#0f9d58", fillcolor="rgba(15,157,88,0.15)")
                fig.update_layout(yaxis_range=[0, 100], height=300, margin=dict(t=50, b=10, l=10, r=10))
                st.plotly_chart(fig)

    with tab_up:
        batch = st.session_state.get("batch")
        if batch is None:
            st.info("Upload a CSV of households to see community charts here.", icon=":material/info:")
            st.button("Upload data", icon=":material/arrow_forward:", on_click=goto, args=("Upload Data",), key="db_upload")
            return
        df = batch["results"]
        st.caption(f"Results from **{batch['meta']['filename']}** ({len(df)} households).")
        with st.container(border=True):
            f1, f2, f3 = st.columns(3)
            emps = sorted(df["employment"].unique())
            emp = f1.multiselect("Employment", emps, default=emps, key="cm_emp")
            hs_min, hs_max = int(df["household_size"].min()), int(df["household_size"].max())
            hs = f2.slider("Household size", hs_min, max(hs_max, hs_min + 1), (hs_min, max(hs_max, hs_min + 1)), key="cm_hs")
            lv = f3.multiselect("Risk level", LEVELS, default=LEVELS, key="cm_lv")
        d = df[df["employment"].isin(emp) & df["household_size"].between(hs[0], hs[1]) & df["risk_level"].isin(lv)]
        if d.empty:
            st.warning("No households match these filters.")
            return
        k1, k2, k3, k4 = st.columns(4)
        with k1.container(border=True):
            st.metric("Households", len(d))
        with k2.container(border=True):
            st.metric("High risk", f"{(d['risk_level'] == 'High').mean() * 100:.0f}%")
        with k3.container(border=True):
            st.metric("Average risk score", f"{d['risk_score'].mean():.0f}")
        with k4.container(border=True):
            st.metric("Avg dietary diversity", f"{d['dietary_diversity'].mean():.1f}")

        c1, c2 = st.columns(2)
        with c1.container(border=True):
            counts = d["risk_level"].value_counts().reindex(LEVELS).fillna(0).reset_index()
            counts.columns = ["Risk level", "Households"]
            fig = px.pie(counts, names="Risk level", values="Households", hole=0.5, color="Risk level",
                         color_discrete_map=COLORS, title="Risk level distribution")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)
        with c2.container(border=True):
            e = d.groupby("employment")["risk_level"].apply(lambda s: (s == "High").mean() * 100).reset_index()
            e.columns = ["Employment", "High-risk households (%)"]
            fig = px.bar(e, x="Employment", y="High-risk households (%)", title="High risk by employment type")
            fig.update_traces(marker_color="#0f9d58")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)
        c3, c4 = st.columns(2)
        with c3.container(border=True):
            h = d.groupby("household_size")["risk_score"].mean().reset_index()
            h.columns = ["Household size", "Average risk score"]
            fig = px.bar(h, x="Household size", y="Average risk score", title="Average risk score by household size")
            fig.update_traces(marker_color="#0f766e")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)
        with c4.container(border=True):
            fig = px.box(d, x="risk_level", y="dietary_diversity", color="risk_level",
                         category_orders={"risk_level": LEVELS}, color_discrete_map=COLORS,
                         labels={"risk_level": "Risk level", "dietary_diversity": "Dietary diversity"},
                         title="Dietary diversity by risk level")
            fig.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10), height=330)
            st.plotly_chart(fig)
        c5, c6 = st.columns(2)
        with c5.container(border=True):
            imp = pd.DataFrame({"Factor": FACTORS, "Mean effect": [d[f"effect_{f}"].mean() for f in FACTORS]}).sort_values("Mean effect")
            imp["Direction"] = np.where(imp["Mean effect"] > 0, "Raises risk", "Lowers risk")
            fig = px.bar(imp, x="Mean effect", y="Factor", orientation="h", color="Direction",
                         color_discrete_map={"Raises risk": "#dc2626", "Lowers risk": "#16a34a"},
                         title="Average effect of each factor on risk")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340, legend_title_text="")
            st.plotly_chart(fig)
        with c6.container(border=True):
            fig = px.scatter(d, x="income_per_person", y="risk_score", color="risk_level",
                             category_orders={"risk_level": LEVELS}, color_discrete_map=COLORS,
                             labels={"income_per_person": "Income per person", "risk_score": "Risk score", "risk_level": "Risk level"},
                             title="Income per person vs risk score")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig)


def page_report():
    st.title("📄 Report")
    r = require_result("rep")
    if r is None:
        return
    u = st.session_state["user"]
    assessments = load_assessments(u["id"])
    hist = history_df(assessments)
    drivers = risk_drivers(r["contributions"], 3)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk level", r["level"])
        c2.metric("Risk score", f"{r['score']}/100")
        c3.metric("Assessments in history", len(hist))
        st.markdown("**Main contributing factors**")
        if drivers:
            for f in drivers:
                st.markdown(f"- {FACTOR_ICON[f]} **{f}** ({describe_input(r['inputs'], f)}): {ADVICE[f]}")
        else:
            st.markdown("- No major risk factors found.")

    st.markdown("##### The PDF report includes")
    st.markdown(
        "✅ Risk level, score and probabilities  \n✅ Household information  \n✅ Risk factor analysis chart  \n"
        "✅ Recommendations  \n✅ Suggested affordable foods  \n✅ Budget meal plan  \n✅ Risk trend history"
    )

    c1, c2, _ = st.columns([1, 1, 2])
    if HAVE_FPDF:
        foods = load_foods()
        recs = recommend_foods(r, n=6)[0] if foods is not None else []
        plan = compute_plan(int(r["inputs"]["size"]), 2100, 50, True, ()) if foods is not None else None
        try:
            pdf_bytes = build_pdf(u, r, assessments, recs, plan)
            c1.download_button("Download PDF report", pdf_bytes, file_name="food_security_report.pdf",
                               mime="application/pdf", icon=":material/picture_as_pdf:", type="primary")
        except Exception as e:  # keep the page alive if PDF building fails
            st.error(f"Could not build the PDF: {e}")
    else:
        st.warning("PDF export needs the `fpdf2` library. Add `fpdf2` to requirements.txt.")
    export = hist.copy()
    export["Date"] = export["Date"].dt.date
    c2.download_button("Download history (CSV)", export.to_csv(index=False), file_name="assessment_history.csv",
                       icon=":material/table_view:")
    st.markdown("Looking for a report on many households? Use **Upload Data → Report**.")
    st.caption(DISCLAIMER)


def page_profile():
    st.title("👤 Profile")
    u = st.session_state["user"]
    with st.container(border=True):
        st.markdown(
            f'<div class="user-row"><div class="avatar">{html.escape(u["name"].strip()[:1].upper())}</div><div>'
            f'<div class="u-name">{html.escape(u["name"])}</div>'
            f'<div class="u-mail">{html.escape(u["email"])} · Member since {html.escape(u["created_at"])}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("##### Update name")
        with st.form("name_form", border=False):
            new_name = st.text_input("Full name", value=u["name"], key="pf_name")
            ok = st.form_submit_button("Save", icon=":material/save:", type="primary")
        if ok:
            if len(new_name.strip()) < 2:
                st.error("Please enter your full name.")
            else:
                q("UPDATE users SET name=? WHERE id=?", (new_name.strip(), u["id"]), write=True)
                st.session_state["user"]["name"] = new_name.strip()
                st.rerun()
    with c2.container(border=True):
        st.markdown("##### Change password")
        with st.form("pw_form", border=False):
            cur = st.text_input("Current password", type="password", key="pf_cur")
            new = st.text_input("New password", type="password", key="pf_new")
            new2 = st.text_input("Confirm new password", type="password", key="pf_new2")
            ok = st.form_submit_button("Update password", icon=":material/lock_reset:", type="primary")
        if ok:
            row = q("SELECT * FROM users WHERE id=?", (u["id"],), one=True)
            if not check_hash(cur, row["pw_hash"]):
                st.error("Current password is incorrect.")
            elif len(new) < 6:
                st.error("New password must be at least 6 characters.")
            elif new != new2:
                st.error("New passwords do not match.")
            else:
                q("UPDATE users SET pw_hash=? WHERE id=?", (make_hash(new), u["id"]), write=True)
                st.success("Password updated.")


def page_setup_error():
    st.title("Setup needed")
    st.error(
        "The trained model file **model.pkl** was not found, or it could not be loaded. "
        "The app needs it to make predictions.",
        icon=":material/error:",
    )
    st.markdown(
        "1. Run the Colab notebook and download `food_security_export.zip`.\n"
        "2. Upload **model.pkl**, **foods.csv** and **ml_training.db** to the root of your GitHub repository, next to `streamlit_app.py`.\n"
        "3. Make sure `requirements.txt` lists `scikit-learn`, `xgboost`, `shap` and `joblib` with the same versions as Colab.\n"
        "4. Reboot the app from **Manage app**."
    )


# ===================================================================== main
if "user" not in st.session_state:
    st.session_state["user"] = None
for _k in ("last", "batch", "form_error", "batch_error", "batch_rejected_only"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

init_db()
st.markdown(CSS, unsafe_allow_html=True)

if st.session_state["user"] is None:
    page_auth()
else:
    render_sidebar()
    ROUTES = {
        "Home": page_home,
        "Assessment": page_assessment,
        "Upload Data": page_upload,
        "Risk Prediction": page_prediction,
        "Risk Factors": page_risk_factors,
        "Food Recommendations": page_food_recommendations,
        "Meal Planner": page_meal_planner,
        "Trends": page_trends,
        "Dashboard": page_dashboard,
        "Report": page_report,
        "Profile": page_profile,
    }
    if load_model_bundle() is None and st.session_state["page"] != "Profile":
        page_setup_error()
    else:
        ROUTES[st.session_state["page"]]()
