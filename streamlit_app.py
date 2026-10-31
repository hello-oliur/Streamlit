"""Food Security & Malnutrition Risk Prediction (SDG 3) - Streamlit app.

How it works
 * After login the user first sees ONLY an upload screen.
 * They upload a raw survey CSV. The app asks them to match the columns it needs,
   ignores every other column, cleans the rows (nulls, wrong values, duplicates) using the
   training data, scores every household with the trained model and explains it with SHAP.
 * Every upload is saved in the database. On the next login the latest upload loads
   automatically, and the user can upload new data at any time.
 * Every page (Home, Households, Risk Factors, Food, Meals, Trends, Dashboard, Report)
   works from the uploaded data.

Files read from the repository root (created by the Colab notebook):
 * model.pkl        trained model bundle: {"model", "columns", "label_map"}   (required)
 * foods.csv        food prices and nutrition                                  (required)
 * ml_training.db   SQLite training data, used to clean uploads                (recommended)
"""
import difflib
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
DB_PATH = os.environ.get("FS_DB_PATH", "foodsecurity.db")  # accounts and saved uploads (created automatically)
MODEL_PATH = "model.pkl"
FOODS_CSV = "foods.csv"
TRAIN_DB = "ml_training.db"
MAX_UPLOAD_ROWS = 20000
NONE_OPT = "— not in my file —"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SEC_QUESTION = "What is your favourite food?"
DISCLAIMER = "Screening guidance only. This is not a medical diagnosis."

LEVELS = ["Low", "Moderate", "High"]
COLORS = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#dc2626"}
LEVEL_STYLE = {
    "Low": ("✅", "linear-gradient(135deg,#15803d,#22c55e)", "Low food-security risk."),
    "Moderate": ("⚠️", "linear-gradient(135deg,#d97706,#fbbf24)", "Some risk factors need attention."),
    "High": ("🚨", "linear-gradient(135deg,#b91c1c,#f87171)", "Several risk factors are present. Act on the recommendations."),
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
MAP_FIELDS = ["household_id"] + CANON + ["risk_level"]
FIELD_LABEL = {
    "household_id": "Household ID (optional)",
    "household_size": "Household size (people)",
    "monthly_income": "Monthly income (BDT)",
    "food_expenditure": "Monthly food spending (BDT)",
    "meals_per_day": "Meals per day",
    "dietary_diversity": "Dietary diversity (0–12)",
    "employment": "Employment type",
    "child_under5": "Child under 5 (yes/no)",
    "food_access": "Difficulty accessing food (1–5)",
    "risk_level": "Known risk level (optional)",
}
COLUMN_GUIDE = pd.DataFrame(
    [
        ("household_id", "Optional", "Any text or number that identifies the household"),
        ("household_size", "Required", "Number of people (1 or more)"),
        ("monthly_income", "Required", "Total monthly household income in BDT"),
        ("food_expenditure", "Required", "Monthly spending on food in BDT"),
        ("meals_per_day", "Recommended", "1, 2, 3 or 4"),
        ("dietary_diversity", "Recommended", "Food groups eaten, 0 to 12"),
        ("employment", "Recommended", "Regular, Daily wage or Unemployed"),
        ("child_under5", "Recommended", "1 / 0 or Yes / No"),
        ("food_access", "Recommended", "1 (easy) to 5 (very hard)"),
        ("risk_level", "Optional", "Low, Moderate or High. If present, the model is checked against it"),
    ],
    columns=["Column", "Needed?", "Allowed values"],
)
ALIASES = {
    "id": "household_id", "hh_id": "household_id", "hhid": "household_id", "hid": "household_id",
    "hh_size": "household_size", "family_size": "household_size", "household_members": "household_size", "members": "household_size",
    "hhsize": "household_size", "hv009": "household_size", "hsize": "household_size",
    "income": "monthly_income", "household_income": "monthly_income", "monthly_household_income": "monthly_income",
    "total_income": "monthly_income", "hh_income": "monthly_income",
    "food_exp": "food_expenditure", "food_spending": "food_expenditure", "monthly_food_expenditure": "food_expenditure",
    "food_expense": "food_expenditure", "monthly_food_spending": "food_expenditure", "food_cost": "food_expenditure",
    "meals": "meals_per_day", "number_of_meals": "meals_per_day", "meals_a_day": "meals_per_day", "meal_frequency": "meals_per_day",
    "diet_diversity": "dietary_diversity", "dietary_diversity_score": "dietary_diversity", "hdds": "dietary_diversity",
    "dds": "dietary_diversity", "food_groups": "dietary_diversity",
    "employment_status": "employment", "job_type": "employment", "occupation_type": "employment", "work_status": "employment",
    "child_under_5": "child_under5", "children_under_5": "child_under5", "has_child_under5": "child_under5",
    "child_under_five": "child_under5", "child": "child_under5", "under5": "child_under5", "under_5": "child_under5",
    "food_access_difficulty": "food_access", "access": "food_access", "food_availability_difficulty": "food_access",
    "access_difficulty": "food_access",
    "risk": "risk_level", "actual_risk": "risk_level", "label": "risk_level",
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
.gate-hero {text-align:center; padding:1.2rem 0 .4rem 0;}
.gate-hero .g-icon {font-size:3rem;}
.gate-hero .g-title {font-size:1.9rem; font-weight:800; margin:.3rem 0 .2rem 0;}
.gate-hero .g-sub {opacity:.75; max-width:640px; margin:0 auto;}

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
    st.session_state["batch"] = load_latest_batch(uid)
    st.session_state["scope"] = "ALL"
    st.session_state["custom"] = None
    st.session_state["page"] = "Home"


def logout():
    for k in ("user", "batch", "custom", "raw_df", "raw_sig", "batch_error", "batch_rejected_only", "flash_app"):
        st.session_state[k] = None
    st.session_state["scope"] = "ALL"
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
        "date": date.today(),
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
    drivers = risk_drivers(r["contributions"], 4)
    tags, value_w = [], 1.0
    for f in drivers:
        tags += FACTOR_NEEDS[f]["tags"]
        if FACTOR_NEEDS[f]["value"]:
            value_w = 2.0
    if r["inputs"]["child"]:
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


# ================================================== CSV upload: map + clean
def norm_name(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def guess_mapping(columns):
    """Matches the user's column names to the app's fields (exact, alias, then close spelling)."""
    normed = {c: norm_name(c) for c in columns}
    guess, used = {}, set()
    for field in MAP_FIELDS:  # pass 1: exact name, then known alias, for every field
        pick = None
        for c, n in normed.items():
            if c not in used and n == field:
                pick = c
                break
        if pick is None:
            for c, n in normed.items():
                if c not in used and ALIASES.get(n) == field:
                    pick = c
                    break
        if pick is not None:
            guess[field] = pick
            used.add(pick)
    for field in MAP_FIELDS:  # pass 2: close spelling, only for fields still unmatched (never the ID / label columns)
        if field in guess or field in ("household_id", "risk_level"):
            continue
        free = {c: n for c, n in normed.items() if c not in used}
        cand = difflib.get_close_matches(field, list(free.values()), n=1, cutoff=0.85)
        if cand:
            pick = next(c for c, n in free.items() if n == cand[0])
            guess[field] = pick
            used.add(pick)
    return guess


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


def _to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.replace(r"[,\s]|BDT|Tk|৳", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return pd.to_numeric(s, errors="coerce")


def clean_upload(raw, mapping, ref):
    """Keeps only the mapped columns, then validates and cleans them using rules and the training data.

    * rows without household size, income or food spending are rejected (also duplicate IDs, impossible values)
    * other missing values are filled with the training-data median / most common value
    * out-of-range values are moved into the valid range
    * households far outside the training population are flagged (kept)
    """
    n = len(raw)
    idx = pd.RangeIndex(n)
    src = pd.DataFrame(index=idx)
    for c in MAP_FIELDS:
        src[c] = raw[mapping[c]].to_numpy() if c in mapping else np.nan

    work = pd.DataFrame(index=idx)
    work["row_number"] = raw.index.to_numpy() + 2  # row number in the spreadsheet (row 1 is the header)

    if "household_id" in mapping:
        ids = src["household_id"].astype(str).str.strip()
        id_given = ~ids.isin(["", "nan", "None"])
        work["household_id"] = np.where(id_given, ids, (work["row_number"] - 1).astype(str))
    else:
        id_given = pd.Series(False, index=idx)
        work["household_id"] = (work["row_number"] - 1).astype(str)

    warnings = []
    for c in FEATURE_NUM:
        work[c] = _to_number(src[c]) if c in mapping else np.nan
        if c in mapping:
            given = int(src[c].notna().sum())
            ok = int(work[c].notna().sum())
            if given and ok / given < 0.9:
                warnings.append(f"Only {ok / given * 100:.0f}% of the values in '{mapping[c]}' could be read as numbers "
                                f"(for {c}). Check that you matched the right column.")
    work["employment"] = src["employment"].map(_norm_emp)
    work["child_under5"] = src["child_under5"].map(_norm_child)
    for c, hint in (("employment", "Regular, Daily wage or Unemployed"), ("child_under5", "Yes / No or 1 / 0")):
        if c in mapping:
            given = int(src[c].notna().sum())
            ok = int(work[c].notna().sum())
            if given and ok / given < 0.9:
                warnings.append(f"Only {ok / given * 100:.0f}% of the values in '{mapping[c]}' were understood as {c} "
                                f"({hint}). Unclear values were filled from the training data.")
    if "risk_level" in mapping:
        lab = src["risk_level"].astype(str).str.strip().str.title().replace({"Medium": "Moderate"})
        work["actual_risk"] = lab.where(lab.isin(LEVELS))

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
            if c in mapping:
                imp[c] = m
            work.loc[m, c] = ref["modes"][c] if c in ref["modes"] else ref["medians"][c]

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

    used_cols = list(dict.fromkeys(mapping.values()))
    rejected = raw[used_cols].reset_index(drop=True)[rej.to_numpy()].copy()
    rejected.insert(0, "Row number", work.loc[rej, "row_number"].to_numpy())
    rejected["Reason"] = ["; ".join(reasons[i]) for i in np.where(rej.to_numpy())[0]]

    unmapped = [c for c in IMPUTABLE if c not in mapping]
    if unmapped:
        warnings.append("These fields are not in your file, so the typical training-data value was used for every "
                        "household: " + ", ".join(unmapped) + ". Results are less reliable for them.")

    reason_counts = Counter(r for rs in reasons for r in rs)
    kept_mask = ~rej
    return {
        "clean": keep,
        "rejected": rejected,
        "warnings": warnings,
        "unmapped": unmapped,
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


def run_batch(clean):
    """Scores every cleaned household with the trained model and explains it with SHAP."""
    bundle = load_model_bundle()
    X = feature_frame(clean, bundle)
    P = predict_proba_frame(bundle, X)
    ind = pd.DataFrame([rule_contributions(row_to_d(r)) for _, r in clean.iterrows()], index=clean.index)
    try:
        C = shap_factor_matrix(bundle, X)
        explain = "SHAP"
    except Exception:
        explain = "rules"
        C = pd.DataFrame({f: ind[f] - BASELINE[f] for f in FACTORS}, index=clean.index)
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


def summarize(res):
    pct = (res["risk_level"].value_counts(normalize=True) * 100).reindex(LEVELS).fillna(0)
    return {
        "n": int(len(res)),
        "pct_low": float(pct["Low"]), "pct_moderate": float(pct["Moderate"]), "pct_high": float(pct["High"]),
        "avg_score": float(res["risk_score"].mean()),
        "effects": {f: float(res[f"effect_{f}"].mean()) for f in FACTORS},
    }


def process_upload(raw, mapping, filename, n_blank=0, truncated=False):
    ref = training_reference()
    cleaned = clean_upload(raw, mapping, ref)
    if cleaned["clean"].empty:
        return {"empty": True, **cleaned}
    results, explain = run_batch(cleaned["clean"])
    ev = evaluate_labels(results)
    rejected = cleaned["rejected"]
    meta = {
        "filename": filename,
        "created": datetime.now().isoformat(timespec="seconds"),
        "explain": explain,
        "mapping": mapping,
        "counts": {**cleaned["counts"], "n_blank": int(n_blank)},
        "reason_counts": cleaned["reason_counts"],
        "warnings": cleaned["warnings"],
        "used_training_reference": cleaned["used_training_reference"],
        "eval": None if ev is None else {k: v for k, v in ev.items() if k != "cm"},
        "truncated": bool(truncated),
        "summary": summarize(results),
        "rejected_csv": rejected.head(300).to_csv(index=False) if len(rejected) else "",
    }
    return {"results": results, "meta": meta, "rejected": rejected, "eval_cm": None if ev is None else ev["cm"]}


def save_batch(user_id, batch):
    return q(
        "INSERT INTO batch_uploads (user_id, created_at, filename, n_rows, meta, results_csv) VALUES (?,?,?,?,?,?)",
        (user_id, batch["meta"]["created"], batch["meta"]["filename"], int(len(batch["results"])),
         json.dumps(batch["meta"], default=float), batch["results"].to_csv(index=False)),
        write=True,
    )


def _batch_from_row(row):
    try:
        res = pd.read_csv(io.StringIO(row["results_csv"]), dtype={"household_id": str})
        meta = json.loads(row["meta"])
    except Exception:
        return None
    cm = None
    if meta.get("eval") and "actual_risk" in res.columns:
        ev = evaluate_labels(res)
        cm = None if ev is None else ev["cm"]
    rejected = None
    if meta.get("rejected_csv"):
        try:
            rejected = pd.read_csv(io.StringIO(meta["rejected_csv"]))
        except Exception:
            rejected = None
    return {"id": row["id"], "results": res, "meta": meta, "rejected": rejected, "eval_cm": cm}


def load_latest_batch(user_id):
    row = q("SELECT * FROM batch_uploads WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,), one=True)
    return _batch_from_row(row) if row else None


def load_batch_by_id(user_id, bid):
    row = q("SELECT * FROM batch_uploads WHERE id=? AND user_id=?", (bid, user_id), one=True)
    return _batch_from_row(row) if row else None


def list_batches(user_id):
    rows = q("SELECT id, created_at, filename, n_rows, meta FROM batch_uploads WHERE user_id=? ORDER BY id ASC", (user_id,))
    out = []
    for r in rows:
        try:
            meta = json.loads(r["meta"])
        except Exception:
            meta = {}
        out.append({"id": r["id"], "created": r["created_at"], "filename": r["filename"], "n": r["n_rows"], "meta": meta})
    return out


def delete_batch(user_id, bid):
    q("DELETE FROM batch_uploads WHERE id=? AND user_id=?", (bid, user_id), write=True)


def result_from_batch_row(r, explain="SHAP"):
    d = row_to_d(r)
    return {
        "level": r["risk_level"],
        "score": int(r["risk_score"]),
        "probs": {"Low": float(r["p_low"]), "Moderate": float(r["p_moderate"]), "High": float(r["p_high"])},
        "contributions": {f: float(r[f"effect_{f}"]) for f in FACTORS},
        "indicators": rule_contributions(d),
        "inputs": d,
        "date": date.today(),
        "explain": explain,
    }


def community_result(batch):
    """A 'typical household' built from the whole upload: median values and average SHAP effects."""
    res = batch["results"]
    mode_emp = res["employment"].mode()
    typical = {
        "size": max(1, int(round(res["household_size"].median()))),
        "income": float(res["monthly_income"].median()),
        "food_exp": float(res["food_expenditure"].median()),
        "meals": int(round(res["meals_per_day"].median())),
        "diversity": float(res["dietary_diversity"].median()),
        "employment": str(mode_emp.iloc[0]) if len(mode_emp) else "Regular",
        "child": bool(res["child_under5"].mean() >= 0.5),
        "access": int(round(res["food_access"].median())),
    }
    probs = {"Low": float(res["p_low"].mean()), "Moderate": float(res["p_moderate"].mean()), "High": float(res["p_high"].mean())}
    return {
        "level": max(probs, key=probs.get),
        "score": int(round(res["risk_score"].mean())),
        "probs": probs,
        "contributions": {f: float(res[f"effect_{f}"].mean()) for f in FACTORS},
        "indicators": rule_contributions(typical),
        "inputs": typical,
        "date": date.today(),
        "explain": batch["meta"].get("explain", "SHAP"),
    }


def scope_result():
    """The household (or whole community) the user chose to analyse. Returns (result, label)."""
    s = st.session_state
    batch = s["batch"]
    scope = s.get("scope", "ALL")
    if scope == "CUSTOM" and s.get("custom"):
        return s["custom"], "your single-household check"
    if scope not in ("ALL", "CUSTOM"):
        res = batch["results"]
        row = res[res["household_id"].astype(str) == str(scope)]
        if len(row):
            return result_from_batch_row(row.iloc[0], batch["meta"].get("explain", "SHAP")), f"household {scope}"
    return community_result(batch), f"the whole community (typical household of {community_result(batch)['inputs']['size']} people)"


def scope_picker(key):
    s = st.session_state
    res = s["batch"]["results"]
    ids = list(res["household_id"].astype(str))
    level = dict(zip(ids, res["risk_level"]))
    options = ["ALL"] + (["CUSTOM"] if s.get("custom") else []) + ids
    cur = s.get("scope", "ALL")
    if cur not in options:
        cur = "ALL"

    def fmt(o):
        if o == "ALL":
            return "Whole community (typical household)"
        if o == "CUSTOM":
            return "My single-household check"
        return f"Household {o} · {level.get(o, '')} risk"

    choice = st.selectbox("Analyse", options, index=options.index(cur), format_func=fmt, key=key,
                          help="Pick the whole community or one household from your uploaded data.")
    s["scope"] = choice


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


def build_pdf(user, r, label, history, recs, plan):
    """Report for one household (or the typical household of the whole community)."""
    L = dict(new_x="LMARGIN", new_y="NEXT")
    pdf = _pdf_start(
        "Food Security Risk Report",
        f"Prepared for {user['name']}   |   {label}   |   {date.today():%d %b %Y}",
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

    if history:
        _pdf_heading(pdf, "Community trend across your uploads")
        for h in history[-8:]:
            s = h["meta"].get("summary")
            if s:
                pdf.cell(0, 5.5, _t(f"{h['created'][:10]}  {h['filename'][:30]}:  average score {s['avg_score']:.0f},  "
                                    f"{s['pct_high']:.0f}% High risk"), **L)
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
    for w in meta.get("warnings", []):
        pdf.multi_cell(0, 5, _t("Note: " + w), **L)
    if meta.get("eval"):
        e = meta["eval"]
        pdf.ln(1)
        rec = "n/a" if e["high_recall"] is None else f"{e['high_recall'] * 100:.0f}%"
        pdf.multi_cell(
            0, 5.5,
            _t(f"Check against the risk level you supplied ({e['n']} rows): accuracy {e['accuracy'] * 100:.0f}%, "
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
def wide_button(*args, **kwargs):
    """Full-width button; falls back for older Streamlit versions."""
    try:
        return st.button(*args, width="stretch", **kwargs)
    except TypeError:
        return st.button(*args, use_container_width=True, **kwargs)


def goto(page):
    st.session_state["page"] = page


def use_household(hid, page):
    st.session_state["scope"] = str(hid)
    st.session_state["page"] = page


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


def explain_caption(r):
    if r.get("explain") == "SHAP":
        return "Computed with **SHAP** from the trained model."
    return "Approximate explanation from the indicator points (the `shap` library is not installed)."


def effect_chart(contrib, title, height=360):
    df = pd.DataFrame({"Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS], "Effect": [contrib[f] for f in FACTORS]})
    df["Direction"] = np.where(df["Effect"] > 0, "Raises risk", "Lowers risk")
    fig = px.bar(df.sort_values("Effect"), x="Effect", y="Factor", orientation="h", color="Direction",
                 color_discrete_map={"Raises risk": "#dc2626", "Lowers risk": "#16a34a"}, title=title)
    fig.update_layout(height=height, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
    fig.add_vline(x=0, line_color="#6b7280")
    return fig


PROB_CFG = {"P(High) %": st.column_config.ProgressColumn("P(High) %", min_value=0, max_value=100, format="%.0f%%")}


# ===================================================================== auth
def page_auth():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="auth-hero">
              <div class="logo">🍚</div>
              <div class="a-title">Food Security<br>Risk Assessment</div>
              <div class="a-sub">Upload your survey data, find which households are at risk of malnutrition,
              understand what drives the risk and plan affordable, nutritious meals.</div>
              <ul>
                <li>📤&nbsp; Upload a survey CSV, only the needed columns are used</li>
                <li>🧹&nbsp; Data is checked and cleaned with the training data</li>
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
    ("Data", [("Home", ":material/home:"), ("Upload Data", ":material/upload_file:")]),
    (
        "Analyse",
        [
            ("Households", ":material/groups:"),
            ("Check One Household", ":material/assignment:"),
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


def render_sidebar(minimal=False):
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
        if not minimal:
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


# ===================================================================== upload
def read_uploaded_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:  # maybe a different separator
                for sep in (";", "\t", "|"):
                    alt = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep)
                    if alt.shape[1] > 1:
                        return alt
            return df
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return None
    return None


def process_clicked():
    s = st.session_state
    raw = s.get("raw_df")
    if raw is None:
        return
    mapping = {}
    for c in MAP_FIELDS:
        v = s.get(f"map_{c}")
        if v and v != NONE_OPT:
            mapping[c] = v
    s["batch_rejected_only"] = None
    missing = [FIELD_LABEL[c] for c in CRITICAL if c not in mapping]
    if missing:
        s["batch_error"] = "Please match these required fields to a column in your file: " + ", ".join(missing) + "."
        return
    used = list(mapping.values())
    if len(set(used)) != len(used):
        s["batch_error"] = "The same column is matched to two different fields. Please check the matching."
        return
    if training_reference() is None and any(c not in mapping for c in IMPUTABLE):
        s["batch_error"] = ("Some recommended fields are not matched, and ml_training.db is missing, so they cannot be filled. "
                            "Match all fields, or add ml_training.db to your repository.")
        return
    n_before = len(raw)
    raw = raw.dropna(how="all")
    n_blank = n_before - len(raw)
    truncated = len(raw) > MAX_UPLOAD_ROWS
    if truncated:
        raw = raw.head(MAX_UPLOAD_ROWS)
    out = process_upload(raw, mapping, s.get("raw_name", "upload.csv"), n_blank, truncated)
    if out.get("empty"):
        s["batch_error"] = "Every row was rejected by the data checks. The reasons are shown below."
        s["batch_rejected_only"] = out
        return
    out["id"] = save_batch(s["user"]["id"], out)
    s["batch"] = out
    s["batch_error"] = None
    s["scope"] = "ALL"
    s["flash_app"] = (f"Done! {len(out['results']):,} households were analysed from **{out['meta']['filename']}**"
                      + (f" ({out['meta']['counts']['n_rejected']:,} rows were rejected)." if out["meta"]["counts"]["n_rejected"] else "."))
    s["raw_df"] = None
    s["raw_sig"] = None
    s["page"] = "Home"


def upload_flow():
    """Template + file picker + column matching + Process button. Used on the first-time screen and on 'Upload Data'."""
    s = st.session_state
    with st.expander("What does my file need? (column guide and template)"):
        c1, c2 = st.columns([1, 2.4], gap="large")
        with c1:
            template = ",".join(["household_id"] + CANON + ["risk_level"]) + "\n"
            st.download_button("Download CSV template", template, file_name="household_template.csv",
                               icon=":material/download:", key="dl_template")
            st.caption("Optional. Your own survey file is fine: you match its columns in the next step, "
                       "and every column the app does not need is ignored.")
        with c2:
            st.dataframe(COLUMN_GUIDE, hide_index=True)

    with st.container(border=True):
        st.markdown("##### Step 1. Choose your survey CSV file")
        f = st.file_uploader("Survey CSV", type=["csv"], key="up_file", label_visibility="collapsed")
        st.caption(f"Up to {MAX_UPLOAD_ROWS:,} rows are processed. Your file can have any number of extra columns.")
        if f is None:
            s["raw_df"], s["raw_sig"] = None, None
            return
        sig = (f.name, f.size)
        if s.get("raw_sig") != sig:
            raw = read_uploaded_csv(f)
            s["raw_sig"], s["raw_name"] = sig, f.name
            s["batch_error"], s["batch_rejected_only"] = None, None
            if raw is None or raw.dropna(how="all").empty:
                s["raw_df"] = None
            else:
                s["raw_df"] = raw
                guess = guess_mapping(raw.columns)
                for c in MAP_FIELDS:
                    s[f"map_{c}"] = guess.get(c, NONE_OPT)
        raw = s.get("raw_df")
        if raw is None:
            st.error("The file is empty or could not be read as a CSV.", icon=":material/error:")
            return
        st.success(f"Found {len(raw):,} rows and {len(raw.columns)} columns.", icon=":material/task_alt:")

    with st.container(border=True):
        st.markdown("##### Step 2. Match your columns")
        st.caption("The app picked the closest column for each field. Check them and change any that are wrong. "
                   "Only these columns are used. All other columns are ignored.")
        options = [NONE_OPT] + [str(c) for c in raw.columns]
        cols = st.columns(3)
        for i, field in enumerate(MAP_FIELDS):
            with cols[i % 3]:
                if s.get(f"map_{field}") not in options:
                    s[f"map_{field}"] = NONE_OPT
                st.selectbox(FIELD_LABEL[field], options, key=f"map_{field}")
        chosen = {c: s[f"map_{c}"] for c in MAP_FIELDS if s.get(f"map_{c}") not in (None, NONE_OPT)}
        if chosen:
            st.markdown("**What the app will use (first 5 rows):**")
            prev = raw[list(dict.fromkeys(chosen.values()))].head(5)
            st.dataframe(prev, hide_index=True)
        unmatched = [c for c in IMPUTABLE if c not in chosen]
        if unmatched:
            st.info("Not matched: " + ", ".join(unmatched) + ". The typical value from the training data will be used for these.",
                    icon=":material/info:")

    with st.container(border=True):
        st.markdown("##### Step 3. Clean and analyse")
        st.caption("The app removes unusable rows, fills small gaps using the training data, and scores every household "
                   "with the trained model.")
        st.button("Clean and analyse my data", icon=":material/play_arrow:", type="primary",
                  on_click=process_clicked, key="batch_go")

    err = s.get("batch_error")
    if err:
        st.error(err, icon=":material/error:")
        ro = s.get("batch_rejected_only")
        if ro is not None:
            rc = pd.DataFrame(sorted(ro["reason_counts"].items(), key=lambda kv: -kv[1]), columns=["Why rows were rejected", "Rows"])
            st.dataframe(rc, hide_index=True)
            st.dataframe(ro["rejected"].head(50), hide_index=True)


def page_gate():
    """Shown after login until the user has uploaded data."""
    st.markdown(
        '<div class="gate-hero"><div class="g-icon">📤</div><div class="g-title">Upload your survey data to begin</div>'
        '<div class="g-sub">Nothing is shown until you upload a CSV. Your file is checked and cleaned with the training '
        'data, then every household is scored by the trained model. Your results are saved, so they are here next time you log in.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-row"><div class="step"><b>1</b> Choose CSV</div><div class="step"><b>2</b> Match columns</div>'
        '<div class="step"><b>3</b> Clean and analyse</div><div class="step"><b>4</b> See results</div></div>',
        unsafe_allow_html=True,
    )
    upload_flow()


def page_upload():
    st.title("📤 Upload Data")
    batch = st.session_state["batch"]
    meta = batch["meta"]
    c = meta["counts"]
    st.success(f"Current data: **{meta['filename']}**, {len(batch['results']):,} households, uploaded {meta['created'].replace('T', ' ')}.",
               icon=":material/task_alt:")
    t_new, t_hist, t_dq = st.tabs([":material/upload_file: Upload new data", ":material/history: Upload history",
                                    ":material/fact_check: Data quality of current data"])
    with t_new:
        st.caption("Uploading a new file analyses it and makes it your current data. Your older uploads are kept in the history.")
        upload_flow()

    with t_hist:
        user = st.session_state["user"]
        hist = list_batches(user["id"])
        rows = []
        for h in hist:
            sm = h["meta"].get("summary", {})
            rows.append({"ID": h["id"], "Date": h["created"].replace("T", " "), "File": h["filename"], "Households": h["n"],
                         "High risk %": round(sm.get("pct_high", 0), 1), "Avg score": round(sm.get("avg_score", 0), 1),
                         "Current": "✔" if h["id"] == batch["id"] else ""})
        st.dataframe(pd.DataFrame(rows).sort_values("ID", ascending=False), hide_index=True)
        labels = {h["id"]: f"#{h['id']} · {h['filename']} · {h['created'][:10]}" for h in hist}
        pick = st.selectbox("Choose an upload", list(labels), format_func=lambda i: labels[i], key="hist_pick")
        b1, b2, _ = st.columns([1.2, 1.2, 2])
        if b1.button("Make this my current data", icon=":material/restore:", key="hist_load", disabled=pick == batch["id"]):
            nb = load_batch_by_id(user["id"], pick)
            if nb:
                st.session_state["batch"] = nb
                st.session_state["scope"] = "ALL"
                st.rerun()
        sure = b2.checkbox("Yes, delete it", key="hist_sure")
        if b2.button("Delete this upload", icon=":material/delete:", key="hist_del", disabled=not sure):
            delete_batch(user["id"], pick)
            if pick == batch["id"]:
                st.session_state["batch"] = load_latest_batch(user["id"])
                st.session_state["scope"] = "ALL"
            st.rerun()

    with t_dq:
        render_data_quality(batch)
    runs = model_runs()
    if runs is not None:
        with st.expander("About the model (training results from Colab)"):
            st.dataframe(runs, hide_index=True)
            st.caption("Scores on the held-out test data from the training notebook.")


def render_data_quality(batch):
    meta, res = batch["meta"], batch["results"]
    c = meta["counts"]
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1.container(border=True):
        st.metric("Rows uploaded", c["n_raw"])
    with k2.container(border=True):
        st.metric("Rows analysed", c["n_used"])
    with k3.container(border=True):
        st.metric("Rows rejected", c["n_rejected"])
    with k4.container(border=True):
        st.metric("Rows with values filled", c["n_filled_rows"])
    with k5.container(border=True):
        st.metric("Outside training range", c["n_unusual"])
    for w in meta.get("warnings", []):
        st.warning(w, icon=":material/warning:")
    if c.get("n_blank"):
        st.caption(f"{c['n_blank']} completely blank rows were skipped.")
    if meta.get("used_training_reference"):
        st.caption("Missing optional values were filled with the training-data median or most common value. Households far "
                   "outside the training population were flagged (they stay in the results).")
    else:
        st.warning("ml_training.db was not found, so missing values could not be filled from the training data. Rows with any "
                   "missing value were rejected.", icon=":material/warning:")
    if meta.get("mapping"):
        with st.expander("Which of your columns were used"):
            st.dataframe(pd.DataFrame({"App field": [FIELD_LABEL[k] for k in meta["mapping"]],
                                       "Your column": list(meta["mapping"].values())}), hide_index=True)
    if meta.get("reason_counts"):
        rc = pd.DataFrame(sorted(meta["reason_counts"].items(), key=lambda kv: -kv[1]), columns=["Why rows were rejected", "Rows"])
        st.dataframe(rc, hide_index=True)
    rej = batch.get("rejected")
    if rej is not None and len(rej):
        with st.expander(f"Show the rejected rows ({len(rej)})"):
            st.dataframe(rej, hide_index=True)
            st.download_button("Download rejected rows (CSV)", rej.to_csv(index=False), file_name="rejected_rows.csv",
                               icon=":material/download:", key="dl_rej")
    adj = res[res["data_quality"] != "OK"][["household_id", "data_quality"]]
    if len(adj):
        with st.expander(f"Show the {len(adj)} households with filled, adjusted or flagged values"):
            st.dataframe(adj.rename(columns={"household_id": "Household", "data_quality": "Data check"}), hide_index=True)
    if batch.get("eval_cm") is not None and meta.get("eval"):
        e = meta["eval"]
        with st.container(border=True):
            st.markdown("##### Model check against the risk level in your file")
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows with a label", e["n"])
            m2.metric("Accuracy", f"{e['accuracy'] * 100:.0f}%")
            m3.metric("High-risk recall", "n/a" if e["high_recall"] is None else f"{e['high_recall'] * 100:.0f}%")
            fig = px.imshow(batch["eval_cm"], text_auto=True, color_continuous_scale="Greens",
                            labels=dict(x="Predicted", y="Your label", color="Households"))
            fig.update_layout(height=320, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig)


# ===================================================================== pages
FEATURE_CARDS = [
    ("Households", "👥", "Households & Risk Prediction", "Every household's Low / Moderate / High risk, probability and 0–100 score."),
    ("Check One Household", "📝", "Check One Household", "Enter one household's details by hand and get a quick prediction."),
    ("Risk Factors", "🔍", "Risk Factor Analysis", "See which inputs pushed the risk up or down (SHAP)."),
    ("Food Recommendations", "🥬", "Food Recommendations", "Affordable, nutritious foods matched to the risk factors."),
    ("Meal Planner", "🍽️", "Budget Meal Planner", "The cheapest meal plan within a daily or weekly budget."),
    ("Trends", "📈", "Risk Trend Tracking", "Compare your uploads over time."),
    ("Dashboard", "🧭", "Dashboard", "Charts and key numbers for your uploaded data."),
    ("Report", "📄", "Report", "Download a structured PDF report."),
    ("Upload Data", "📤", "Upload New Data", "Upload a new file or reopen an older one."),
]


def page_home():
    u = st.session_state["user"]
    batch = st.session_state["batch"]
    res, meta = batch["results"], batch["meta"]
    first = html.escape(u["name"].strip().split()[0])
    flash = st.session_state.pop("flash_app", None)
    if flash:
        st.success(flash, icon=":material/task_alt:")
    st.markdown(
        f'<div class="hero"><div class="h-title">Hello, {first} 👋</div>'
        f'<div class="h-sub">Showing results for <b>{html.escape(meta["filename"])}</b> · '
        f'{len(res):,} households analysed.</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Households analysed", f"{len(res):,}")
    with c2.container(border=True):
        st.metric("High risk", f"{(res['risk_level'] == 'High').mean() * 100:.0f}%")
    with c3.container(border=True):
        st.metric("Moderate risk", f"{(res['risk_level'] == 'Moderate').mean() * 100:.0f}%")
    with c4.container(border=True):
        st.metric("Average risk score", f"{res['risk_score'].mean():.0f}/100")

    a, b = st.columns(2)
    with a.container(border=True):
        counts = res["risk_level"].value_counts().reindex(LEVELS).fillna(0).reset_index()
        counts.columns = ["Risk level", "Households"]
        fig = px.pie(counts, names="Risk level", values="Households", hole=0.5, color="Risk level",
                     color_discrete_map=COLORS, title="Risk level distribution")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig)
    with b.container(border=True):
        eff = {f: float(res[f"effect_{f}"].mean()) for f in FACTORS}
        st.plotly_chart(effect_chart(eff, "What drives risk in your data (average effect)", 320))

    with st.container(border=True):
        st.markdown("##### Households that need attention first")
        top = res.sort_values(["p_high", "risk_score"], ascending=False).head(5)
        show = top[["household_id", "risk_level", "p_high", "risk_score", "top_risk_driver"]].copy()
        show["p_high"] = show["p_high"] * 100
        show.columns = ["Household", "Risk level", "P(High) %", "Risk score", "Main driver"]
        st.dataframe(show, hide_index=True, column_config=PROB_CFG)

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


def page_households():
    st.title("👥 Households & Risk Prediction")
    batch = st.session_state["batch"]
    res, meta = batch["results"], batch["meta"]
    n = len(res)
    st.caption("Risk level and probabilities come from the trained model. The 0–100 score is built from 7 simple indicators.")
    with st.container(border=True):
        f1, f2, f3 = st.columns([1.2, 1.2, 1])
        lv = f1.multiselect("Risk level", LEVELS, default=LEVELS, key="bt_lv")
        emps = sorted(res["employment"].unique())
        emp = f2.multiselect("Employment", emps, default=emps, key="bt_emp")
        search = f3.text_input("Search household ID", key="bt_search")
    view = res[res["risk_level"].isin(lv) & res["employment"].isin(emp)]
    if search.strip():
        view = view[view["household_id"].astype(str).str.contains(search.strip(), case=False, regex=False)]
    view = view.sort_values(["p_high", "risk_score"], ascending=False)
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
    st.caption(f"{len(show):,} of {n:,} households shown (highest risk first)")
    st.dataframe(show, hide_index=True, column_config=PROB_CFG)
    st.download_button("Download all results (CSV)", res.to_csv(index=False), file_name="household_risk_results.csv",
                       icon=":material/download:", key="dl_results")

    st.subheader("Look at one household")
    ids = list(view["household_id"].astype(str))
    pick = st.selectbox("Choose a household", ids, index=None, placeholder="Select a household ID", key="bd_pick")
    if pick is None:
        st.info("Select a household to see its prediction, the factors behind it and what to do.", icon=":material/info:")
        return
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
        st.plotly_chart(effect_chart(hr["contributions"], "What pushed this household's risk up or down"))
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
    b1, b2, b3, _ = st.columns([1.2, 1.2, 1.2, 1])
    b1.button("Why this result?", icon=":material/analytics:", on_click=use_household, args=(pick, "Risk Factors"), key="bd_rf")
    b2.button("Food suggestions", icon=":material/eco:", on_click=use_household, args=(pick, "Food Recommendations"), key="bd_food")
    b3.button("Meal plan", icon=":material/restaurant:", on_click=use_household, args=(pick, "Meal Planner"), key="bd_meal")


def run_check():
    """Form callback for 'Check One Household'."""
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
        "size": int(s["in_size"]), "income": float(s["in_income"]), "food_exp": float(s["in_food"]),
        "meals": int(s["in_meals"]), "diversity": float(s["in_div"]), "employment": s["in_emp"],
        "child": s["in_child"] == "Yes", "access": int(s["in_access"]),
    }
    if data["income"] <= 0:
        s["form_error"] = "Monthly income must be above 0."
        return
    s["form_error"] = None
    s["custom"] = predict_household(data)


def page_check_one():
    st.title("📝 Check One Household")
    st.caption("Enter one household's details by hand. It is scored by the same trained model as your uploaded data.")
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
            st.form_submit_button("Assess risk", on_click=run_check, icon=":material/insights:", type="primary")

    r = st.session_state.get("custom")
    if r is None:
        st.caption(DISCLAIMER)
        return
    icon, grad, msg = LEVEL_STYLE[r["level"]]
    st.markdown(
        f'<div class="result-hero" style="background:{grad}"><div class="rh-icon">{icon}</div><div>'
        f'<div class="rh-title">{r["level"]} risk</div><div class="rh-sub">{msg}</div></div></div>',
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    with m1.container(border=True):
        st.metric("Risk score (0–100)", r["score"])
    with m2.container(border=True):
        st.metric("Prediction confidence", f"{max(r['probs'].values()) * 100:.0f}%")
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.plotly_chart(gauge(r["score"]))
    with c2.container(border=True):
        st.plotly_chart(effect_chart(r["contributions"], "What pushed the risk up or down", 330))
    b1, b2, _ = st.columns([1.2, 1.2, 2])
    b1.button("Food suggestions", icon=":material/eco:", on_click=lambda: st.session_state.update(scope="CUSTOM", page="Food Recommendations"), key="ck_food")
    b2.button("Meal plan", icon=":material/restaurant:", on_click=lambda: st.session_state.update(scope="CUSTOM", page="Meal Planner"), key="ck_meal")
    st.caption(DISCLAIMER)


def page_risk_factors():
    st.title("🔍 Risk Factor Analysis")
    scope_picker("sp_rf")
    r, label = scope_result()
    st.caption(f"Shows how much each input **raised** (red) or **lowered** (green) the risk for {label}, compared with an "
               "average household in the training data. " + explain_caption(r))
    contrib = r["contributions"]
    with st.container(border=True):
        st.plotly_chart(effect_chart(contrib, "What pushed the risk up or down", 380))

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
    scope_picker("sp_fr")
    r, label = scope_result()
    st.caption(f"Foods are chosen for {label}, based on its main risk factors.")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.4])
        n = c1.slider("How many foods", 3, 9, 6, key="fr_n")
        top_price = int(foods["Price (BDT/kg)"].max())
        max_price = c2.slider("Max price (BDT/kg)", 20, max(top_price, 30), max(top_price, 30), step=10, key="fr_price")
        excl = c3.multiselect("Exclude foods (allergy / preference)", list(foods["Food"]), key="fr_excl")

    picks, drivers, tags = recommend_foods(r, n=n, max_price=max_price, exclude=excl)
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
    scope_picker("sp_mp")
    r, label = scope_result()
    st.caption(f"Finds the **cheapest combination of foods** that meets the calorie and protein targets for {label}, "
               "then splits it into breakfast, lunch and dinner.")
    default_people = max(1, int(r["inputs"]["size"]))

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        people = c1.number_input("People in household", 1, 30, value=default_people, key=f"mp_people_{label}")
        period = c2.radio("Budget period", ["Daily", "Weekly"], horizontal=True, key="mp_period")
        budget = c3.number_input(f"{period} food budget (BDT), optional", 0, 1_000_000, value=None, step=50,
                                 placeholder="Enter a budget to compare", key="mp_budget")
        c4, c5, c6, c7 = st.columns([1, 1, 1, 1.4])
        kcal_pp = c4.slider("Calories per person", 1500, 2800, 2100, step=50, key="mp_kcal")
        prot_pp = c5.slider("Protein per person (g)", 30, 90, 50, step=5, key="mp_prot")
        variety = c6.checkbox("Include vegetables & fruit", value=True, key="mp_var")
        excl = c7.multiselect("Exclude foods", list(foods["Food"]), key="mp_excl")

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
    inc = r["inputs"]["food_exp"]
    st.caption(f"For comparison, this household's reported monthly food spending is {inc:,.0f} BDT "
               f"(about {inc / 30:,.0f} BDT per day).")
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
    user = st.session_state["user"]
    hist = [h for h in list_batches(user["id"]) if h["meta"].get("summary")]
    if not hist:
        st.info("No saved uploads found.", icon=":material/info:")
        return
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime([h["created"] for h in hist]),
            "File": [h["filename"] for h in hist],
            "Households": [h["n"] for h in hist],
            "High risk %": [h["meta"]["summary"]["pct_high"] for h in hist],
            "Moderate risk %": [h["meta"]["summary"]["pct_moderate"] for h in hist],
            "Low risk %": [h["meta"]["summary"]["pct_low"] for h in hist],
            "Avg score": [h["meta"]["summary"]["avg_score"] for h in hist],
            "id": [h["id"] for h in hist],
        }
    )
    st.caption("Every upload is saved with its date. Upload new survey data later to see whether the community is improving.")
    m1, m2, m3, m4 = st.columns(4)
    with m1.container(border=True):
        st.metric("Saved uploads", len(df))
    with m2.container(border=True):
        st.metric("Latest average score", f"{df['Avg score'].iloc[-1]:.0f}")
    with m3.container(border=True):
        if len(df) >= 2:
            d = df["Avg score"].iloc[-1] - df["Avg score"].iloc[-2]
            st.metric("Score vs previous upload", "No change" if round(d) == 0 else f"{d:+.0f} pts")
        else:
            st.metric("Score vs previous upload", "—")
    with m4.container(border=True):
        if len(df) >= 2:
            d = df["High risk %"].iloc[-1] - df["High risk %"].iloc[-2]
            st.metric("High-risk share vs previous", "No change" if round(d, 1) == 0 else f"{d:+.1f} pts")
        else:
            st.metric("High-risk share vs previous", "—")

    with st.container(border=True):
        if len(df) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Avg score"], mode="lines+markers", name="Average risk score",
                                     line=dict(color="#0f9d58", width=3)))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["High risk %"], mode="lines+markers", name="High-risk households (%)",
                                     line=dict(color="#dc2626", width=3)))
            fig.update_layout(title="Community risk over time (lower is better)", height=380,
                              margin=dict(t=50, b=20, l=10, r=10), yaxis_range=[0, 100])
            st.plotly_chart(fig)
        else:
            st.info("You have one upload so far. Upload another dataset later (for example next month) to see a trend line.",
                    icon=":material/info:")

    st.subheader("Compare two uploads")
    if len(df) < 2:
        st.caption("You need at least two saved uploads to compare them.")
    else:
        labels = {int(r.id): f"#{int(r.id)} · {r.File} · {r.Date:%d %b %Y}" for r in df.itertuples()}
        ids = list(labels)
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Earlier upload", ids, index=0, format_func=lambda i: labels[i], key="cmp_a")
        ib = c2.selectbox("Later upload", ids, index=len(ids) - 1, format_func=lambda i: labels[i], key="cmp_b")
        ha = next(h for h in hist if h["id"] == ia)["meta"]["summary"]
        hb = next(h for h in hist if h["id"] == ib)["meta"]["summary"]
        with st.container(border=True):
            k1, k2, k3 = st.columns(3)
            k1.metric("Average score", f"{hb['avg_score']:.0f}", f"{hb['avg_score'] - ha['avg_score']:+.0f}", delta_color="inverse")
            k2.metric("High-risk households", f"{hb['pct_high']:.0f}%", f"{hb['pct_high'] - ha['pct_high']:+.1f} pts", delta_color="inverse")
            k3.metric("Households", f"{hb['n']:,}", f"{hb['n'] - ha['n']:+,}", delta_color="off")
            dist = pd.DataFrame({
                "Risk level": LEVELS * 2,
                "Upload": ["Earlier"] * 3 + ["Later"] * 3,
                "% of households": [ha["pct_low"], ha["pct_moderate"], ha["pct_high"], hb["pct_low"], hb["pct_moderate"], hb["pct_high"]],
            })
            fig = px.bar(dist, x="Risk level", y="% of households", color="Upload", barmode="group",
                         color_discrete_map={"Earlier": "#9ca3af", "Later": "#0f9d58"}, title="Risk level distribution")
            fig.update_layout(height=320, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig)
            if ha.get("effects") and hb.get("effects"):
                eff = pd.DataFrame({
                    "Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS] * 2,
                    "Upload": ["Earlier"] * len(FACTORS) + ["Later"] * len(FACTORS),
                    "Average effect on risk": [ha["effects"][f] for f in FACTORS] + [hb["effects"][f] for f in FACTORS],
                })
                fig = px.bar(eff, x="Average effect on risk", y="Factor", color="Upload", barmode="group", orientation="h",
                             color_discrete_map={"Earlier": "#9ca3af", "Later": "#0f9d58"},
                             title="What drives risk, earlier vs later (lower is better)")
                fig.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10))
                st.plotly_chart(fig)

    with st.container(border=True):
        st.markdown("##### All saved uploads")
        show = df.drop(columns=["id"]).copy()
        show["Date"] = show["Date"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(show.sort_values("Date", ascending=False).round(1), hide_index=True)


def page_dashboard():
    st.title("🧭 Dashboard")
    batch = st.session_state["batch"]
    df = batch["results"]
    st.caption(f"Charts for **{batch['meta']['filename']}** ({len(df):,} households).")
    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        emps = sorted(df["employment"].unique())
        emp = f1.multiselect("Employment", emps, default=emps, key="cm_emp")
        hs_min, hs_max = int(df["household_size"].min()), int(df["household_size"].max())
        hs_hi = max(hs_max, hs_min + 1)
        hs = f2.slider("Household size", hs_min, hs_hi, (hs_min, hs_hi), key="cm_hs")
        lv = f3.multiselect("Risk level", LEVELS, default=LEVELS, key="cm_lv")
    d = df[df["employment"].isin(emp) & df["household_size"].between(hs[0], hs[1]) & df["risk_level"].isin(lv)]
    if d.empty:
        st.warning("No households match these filters.")
        return
    k1, k2, k3, k4 = st.columns(4)
    with k1.container(border=True):
        st.metric("Households", f"{len(d):,}")
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
        st.plotly_chart(effect_chart({f: float(d[f"effect_{f}"].mean()) for f in FACTORS},
                                     "Average effect of each factor on risk", 340))
    with c6.container(border=True):
        dd = d.sample(min(len(d), 3000), random_state=1)
        fig = px.scatter(dd, x="income_per_person", y="risk_score", color="risk_level",
                         category_orders={"risk_level": LEVELS}, color_discrete_map=COLORS,
                         labels={"income_per_person": "Income per person", "risk_score": "Risk score", "risk_level": "Risk level"},
                         title="Income per person vs risk score")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig)


def page_report():
    st.title("📄 Report")
    u = st.session_state["user"]
    batch = st.session_state["batch"]
    res = batch["results"]
    if not HAVE_FPDF:
        st.warning("PDF export needs the `fpdf2` library. Add `fpdf2` to requirements.txt.")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("##### 🏘️ Community report")
        st.markdown("Risk distribution, comparison with the training data, data-quality checks, main risk drivers and the "
                    "15 priority households.")
        if HAVE_FPDF:
            try:
                st.download_button("Download community report (PDF)", build_batch_pdf(u, batch),
                                   file_name="community_screening_report.pdf", mime="application/pdf",
                                   icon=":material/picture_as_pdf:", type="primary", key="dl_batch_pdf")
            except Exception as ex:
                st.error(f"Could not build the PDF: {ex}")
        st.download_button("Download all results (CSV)", res.to_csv(index=False), file_name="household_risk_results.csv",
                           icon=":material/table_view:", key="dl_results2")
    with c2.container(border=True):
        st.markdown("##### 👤 Household report")
        scope_picker("sp_rep")
        r, label = scope_result()
        st.markdown("Risk level, score, factor analysis, recommendations, affordable foods, meal plan and community trend.")
        if HAVE_FPDF:
            foods = load_foods()
            recs = recommend_foods(r, n=6)[0] if foods is not None else []
            plan = compute_plan(max(1, int(r["inputs"]["size"])), 2100, 50, True, ()) if foods is not None else None
            try:
                pdf_bytes = build_pdf(u, r, label, list_batches(u["id"]), recs, plan)
                st.download_button("Download household report (PDF)", pdf_bytes, file_name="household_report.pdf",
                                   mime="application/pdf", icon=":material/picture_as_pdf:", type="primary", key="dl_hh_pdf")
            except Exception as e:  # keep the page alive if PDF building fails
                st.error(f"Could not build the PDF: {e}")
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
_DEFAULTS = {"user": None, "batch": None, "custom": None, "scope": "ALL", "page": "Home", "form_error": None,
             "batch_error": None, "batch_rejected_only": None, "raw_df": None, "raw_sig": None, "flash_app": None}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

init_db()
st.markdown(CSS, unsafe_allow_html=True)

if st.session_state["user"] is None:
    page_auth()
elif load_model_bundle() is None:
    render_sidebar(minimal=True)
    page_setup_error()
elif st.session_state["batch"] is None:
    render_sidebar(minimal=True)
    page_gate()
else:
    render_sidebar()
    ROUTES = {
        "Home": page_home,
        "Upload Data": page_upload,
        "Households": page_households,
        "Check One Household": page_check_one,
        "Risk Factors": page_risk_factors,
        "Food Recommendations": page_food_recommendations,
        "Meal Planner": page_meal_planner,
        "Trends": page_trends,
        "Dashboard": page_dashboard,
        "Report": page_report,
        "Profile": page_profile,
    }
    ROUTES.get(st.session_state["page"], page_home)()
