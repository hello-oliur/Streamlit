"""Food Security & Malnutrition Risk Assessment (SDG 3) - full demo app.

All 8 project features are included, running on DEMO data:
 1. Food Security Assessment        -> page "Assessment"
 2. AI Malnutrition Risk Prediction -> page "Risk Prediction"
 3. Food Security Risk Score        -> page "Risk Prediction" (0-100 composite score)
 4. Risk Factor Analysis (SHAP)     -> page "Risk Factors"
 5. Personalised Food Recommendation-> page "Food Recommendations"
 6. Budget-Based Meal Planning      -> page "Meal Planner"
 7. Risk Trend Tracking             -> page "Trends"
 8. Dashboard & Report              -> pages "Dashboard" and "Report" (PDF)

HOW TO REPLACE DEMO DATA LATER (no code changes needed):
 * model.pkl            trained model saved from Google Colab (dict with
                        "model", "columns", "label_map"). If present, it is used
                        for predictions, and SHAP is used if the `shap` library
                        is installed. Otherwise the demo rules are used.
 * community_data.csv   real community data for the Dashboard. Columns:
                        Household size, Income per person, Employment,
                        Dietary diversity, Risk score, Risk level
 * foods.csv            real food table. Columns: Food, Category,
                        Price (BDT/kg), kcal/100g, Protein g/100g,
                        Max g/person/day, Tags (separated by |)
"""
import hashlib
import hmac
import html
import json
import os
import re
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog

try:
    from fpdf import FPDF

    HAVE_FPDF = True
except Exception:  # fpdf2 not installed -> PDF button is disabled
    FPDF = object
    HAVE_FPDF = False

st.set_page_config(page_title="Food Security Risk Assessment", page_icon="🍚", layout="wide")

# ----------------------------------------------------------------- settings
DB_PATH = os.environ.get("FS_DB_PATH", "foodsecurity.db")
MODEL_PATH = "model.pkl"
COMMUNITY_CSV = "community_data.csv"
FOODS_CSV = "foods.csv"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SEC_QUESTION = "What is your favourite food?"
DEMO_EMAIL = "demo@foodsecurity.app"
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
MAX_RAW = {
    "Income per person": 3.0,
    "Meals per day": 3.0,
    "Dietary diversity": 2.8,
    "Difficulty accessing food": 2.4,
    "Employment": 2.0,
    "Child under 5": 0.5,
    "Food spending share": 1.0,
}
BASELINE = {  # a "typical" household, used to turn rule points into +/- effects
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
# What each risk factor means for food choices (used by Food Recommendations)
FACTOR_NEEDS = {
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

# --------------------------------------------------------------------- foods
DEFAULT_FOODS = [
    ("Rice", "Staple", 60, 360, 7.0, 400, "energy|local"),
    ("Whole wheat flour (atta)", "Staple", 55, 340, 12.0, 150, "energy|protein"),
    ("Potato", "Staple", 40, 77, 2.0, 300, "energy|local|vitaminC"),
    ("Lentils (dal)", "Pulse", 140, 350, 25.0, 100, "protein|iron|local"),
    ("Chickpeas", "Pulse", 130, 364, 19.0, 80, "protein|iron"),
    ("Peanuts", "Pulse", 160, 567, 26.0, 40, "energy|protein"),
    ("Eggs", "Protein", 170, 143, 12.6, 100, "protein|vitaminA|iron|local"),
    ("Small fish", "Protein", 300, 120, 18.0, 150, "protein|iron|calcium|vitaminA|local"),
    ("Chicken", "Protein", 230, 215, 20.0, 100, "protein|iron"),
    ("Milk", "Dairy", 80, 61, 3.2, 250, "calcium|protein"),
    ("Spinach (palong shak)", "Vegetable", 30, 23, 2.9, 200, "iron|vitaminA|local"),
    ("Red amaranth (lal shak)", "Vegetable", 25, 23, 2.5, 200, "iron|vitaminA|local"),
    ("Pumpkin", "Vegetable", 35, 26, 1.0, 200, "vitaminA|local"),
    ("Brinjal (eggplant)", "Vegetable", 50, 25, 1.0, 200, "local"),
    ("Tomato", "Vegetable", 60, 18, 0.9, 150, "vitaminC"),
    ("Banana", "Fruit", 60, 89, 1.1, 150, "energy|local"),
    ("Papaya", "Fruit", 40, 43, 0.5, 200, "vitaminA|vitaminC|local"),
    ("Guava", "Fruit", 70, 68, 2.6, 150, "vitaminC|local"),
    ("Soybean oil", "Fat", 160, 884, 0.0, 30, "energy"),
]
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

SEED_INPUTS = [  # sample history for the demo account (risk improves over time)
    (150, dict(size=4, income=9000, food_exp=6000, meals=2, diversity=3, employment="Unemployed", child=True, access=4)),
    (120, dict(size=4, income=11000, food_exp=6800, meals=2, diversity=4, employment="Daily wage", child=True, access=4)),
    (90, dict(size=4, income=14000, food_exp=8000, meals=3, diversity=5, employment="Daily wage", child=True, access=3)),
    (60, dict(size=4, income=18000, food_exp=8500, meals=3, diversity=6, employment="Daily wage", child=True, access=2)),
    (30, dict(size=4, income=24000, food_exp=9500, meals=3, diversity=7, employment="Regular", child=True, access=2)),
]

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
            padding:42px 38px; border-radius:26px; min-height:560px;}
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
    st.session_state["page"] = "Home"


def logout():
    st.session_state["user"] = None
    st.session_state["last"] = None
    st.session_state["page"] = "Home"


def demo_login():
    row = q("SELECT * FROM users WHERE email=?", (DEMO_EMAIL,), one=True)
    if not row:
        uid = q(
            "INSERT INTO users (name, email, pw_hash, sec_hash, created_at) VALUES (?,?,?,?,?)",
            ("Demo User", DEMO_EMAIL, make_hash(os.urandom(8).hex()), make_hash("rice"), date.today().isoformat()),
            write=True,
        )
        seed_history(uid)
        row = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    login_user(row)


# ================================================== prediction (demo + model)
def level_from_score(score):
    return "Low" if score < 29 else "Moderate" if score < 57 else "High"


def rule_contributions(d):
    """Risk points (0 = no risk) for each indicator. Used for the composite
    score, the indicator breakdown and as the demo 'explanation'."""
    people = max(d["size"], 1)
    pc_income = d["income"] / people
    c = {}
    c["Income per person"] = 3 if pc_income < 3000 else 2 if pc_income < 6000 else 1 if pc_income < 10000 else 0
    c["Meals per day"] = {1: 3, 2: 1.5, 3: 0, 4: 0}[d["meals"]]
    c["Dietary diversity"] = round(max(0, 7 - d["diversity"]) * 0.4, 2)
    c["Difficulty accessing food"] = round((d["access"] - 1) * 0.6, 2)
    c["Employment"] = {"Regular": 0, "Daily wage": 1, "Unemployed": 2}[d["employment"]]
    c["Child under 5"] = 0.5 if d["child"] else 0
    share = d["food_exp"] / d["income"] if d["income"] > 0 else 1
    c["Food spending share"] = 1 if share > 0.6 else 0.5 if share > 0.5 else 0
    return {k: float(v) for k, v in c.items()}


def composite_score(raw):
    return int(min(100, round(sum(raw.values()) / 14 * 100)))


def demo_predict(d):
    """DEMO ONLY: rule-based prediction. Replaced automatically by model.pkl."""
    raw = rule_contributions(d)
    total = sum(raw.values())
    centers = np.array([1.0, 6.0, 11.0])
    logits = -(((total - centers) / 3.0) ** 2)
    probs = np.exp(logits) / np.exp(logits).sum()
    score = composite_score(raw)
    return {
        "level": level_from_score(score),
        "score": score,
        "probs": {k: float(v) for k, v in zip(LEVELS, probs)},
        "contributions": {k: float(raw[k] - BASELINE[k]) for k in FACTORS},
        "indicators": raw,
        "inputs": d,
        "source": "demo",
        "explain": "rules",
    }


@st.cache_resource
def load_model_bundle():
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


def model_row(d):
    return {
        "household_size": d["size"],
        "monthly_income": d["income"],
        "food_expenditure": d["food_exp"],
        "meals_per_day": d["meals"],
        "dietary_diversity": d["diversity"],
        "child_under5": int(d["child"]),
        "food_access": d["access"],
        "employment_Daily wage": int(d["employment"] == "Daily wage"),
        "employment_Regular": int(d["employment"] == "Regular"),
        "employment_Unemployed": int(d["employment"] == "Unemployed"),
    }


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


def shap_contributions(bundle, X):
    import shap

    explainer = shap.TreeExplainer(bundle["model"])
    high_idx = list(bundle["model"].classes_).index(bundle["label_map"]["High"])
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        vals = np.array(sv[high_idx])[0]
    else:
        arr = np.array(sv)
        vals = arr[0, :, high_idx] if arr.ndim == 3 else arr[0]
    out = {f: 0.0 for f in FACTORS}
    for name, v in zip(X.columns, vals):
        f = feature_factor(name)
        if f:
            out[f] += float(v)
    return out


def predict_household(d):
    """Single entry point used by the app: uses model.pkl if available."""
    base = demo_predict(d)
    bundle = load_model_bundle()
    if bundle is None:
        return base
    try:
        model = bundle["model"]
        X = pd.DataFrame([model_row(d)]).reindex(columns=bundle["columns"], fill_value=0)
        inv = {v: k for k, v in bundle["label_map"].items()}
        probs = {inv[int(c)]: float(p) for c, p in zip(model.classes_, model.predict_proba(X)[0])}
        if set(probs) != set(LEVELS):
            return base
        base["probs"] = probs
        base["level"] = max(probs, key=probs.get)
        base["source"] = "model"
        try:
            base["contributions"] = shap_contributions(bundle, X)
            base["explain"] = "SHAP"
        except Exception:
            pass  # keep rule-based explanation
        return base
    except Exception:
        return base


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
        "Meals per day": f"{d['meals']} per day",
        "Dietary diversity": f"{d['diversity']} of 12 food groups",
        "Difficulty accessing food": f"{d['access']} out of 5",
        "Employment": d["employment"],
        "Child under 5": "Yes" if d["child"] else "No",
        "Food spending share": f"{share:.0f}% of income",
    }[factor]


# ============================================================== assessments
def save_assessment(user_id, res, when):
    payload = {
        "probs": res["probs"],
        "contributions": res["contributions"],
        "inputs": res["inputs"],
        "source": res.get("source", "demo"),
        "explain": res.get("explain", "rules"),
    }
    q(
        "INSERT INTO assessments (user_id, created_at, score, level, data) VALUES (?,?,?,?,?)",
        (user_id, when.isoformat(), res["score"], res["level"], json.dumps(payload, default=float)),
        write=True,
    )


def result_from_row(row):
    d = json.loads(row["data"])
    inputs = d["inputs"]
    contrib = d["contributions"] if "explain" in d else demo_predict(inputs)["contributions"]
    return {
        "id": row["id"],
        "level": row["level"],
        "score": row["score"],
        "probs": d["probs"],
        "contributions": contrib,
        "indicators": rule_contributions(inputs),
        "inputs": inputs,
        "date": date.fromisoformat(row["created_at"]),
        "source": d.get("source", "demo"),
        "explain": d.get("explain", "rules"),
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


def seed_history(user_id):
    for days_ago, inputs in SEED_INPUTS:
        save_assessment(user_id, predict_household(inputs), date.today() - timedelta(days=days_ago))


def seed_history_cb():
    uid = st.session_state["user"]["id"]
    seed_history(uid)
    st.session_state["last"] = latest_result(uid)


def run_assessment():
    """Form callback: runs before the page re-renders."""
    s = st.session_state
    data = {
        "size": int(s["in_size"]),
        "income": int(s["in_income"]),
        "food_exp": int(s["in_food"]),
        "meals": int(s["in_meals"]),
        "diversity": int(s["in_div"]),
        "employment": s["in_emp"],
        "child": s["in_child"] == "Yes",
        "access": int(s["in_access"]),
    }
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


SHORT = {
    "Income per person": "Income",
    "Meals per day": "Meals",
    "Dietary diversity": "Diet variety",
    "Difficulty accessing food": "Food access",
    "Employment": "Employment",
    "Child under 5": "Child under 5",
    "Food spending share": "Food spending",
}


def goto(page):
    st.session_state["page"] = page


# ================================================================ community
def _demo_households():
    rng = np.random.default_rng(7)
    out = []
    for _ in range(200):
        size = int(rng.integers(1, 9))
        income = int(max(3000, rng.lognormal(9.9, 0.5)))
        out.append(
            {
                "size": size,
                "income": income,
                "food_exp": int(income * rng.uniform(0.3, 0.75)),
                "meals": int(rng.choice([1, 2, 3, 4], p=[0.05, 0.25, 0.6, 0.1])),
                "diversity": int(np.clip(round(rng.normal(6, 2.2)), 0, 12)),
                "employment": str(rng.choice(["Regular", "Daily wage", "Unemployed"], p=[0.45, 0.4, 0.15])),
                "child": bool(rng.integers(0, 2)),
                "access": int(rng.integers(1, 6)),
            }
        )
    return out


@st.cache_data
def community_data():
    """Returns (DataFrame, source). Uses community_data.csv if present."""
    need = {"Household size", "Income per person", "Employment", "Dietary diversity", "Risk score", "Risk level"}
    if os.path.exists(COMMUNITY_CSV):
        try:
            df = pd.read_csv(COMMUNITY_CSV)
            if need.issubset(df.columns):
                return df, "real"
        except Exception:
            pass
    rows = []
    for d in _demo_households():
        r = demo_predict(d)
        rows.append(
            {
                "Household size": d["size"],
                "Income per person": d["income"] / d["size"],
                "Employment": d["employment"],
                "Dietary diversity": d["diversity"],
                "Risk score": r["score"],
                "Risk level": r["level"],
            }
        )
    return pd.DataFrame(rows), "demo"


@st.cache_data
def global_importance():
    """Average absolute effect of each factor over the demo community."""
    acc = {f: [] for f in FACTORS}
    for d in _demo_households():
        for f, v in demo_predict(d)["contributions"].items():
            acc[f].append(abs(v))
    return pd.DataFrame({"Factor": FACTORS, "Mean |effect|": [float(np.mean(acc[f])) for f in FACTORS]})


# ===================================================================== foods
@st.cache_data
def load_foods():
    df = None
    if os.path.exists(FOODS_CSV):
        try:
            tmp = pd.read_csv(FOODS_CSV)
            if set(FOOD_COLS).issubset(tmp.columns):
                df = tmp[FOOD_COLS].copy()
        except Exception:
            df = None
    if df is None:
        df = pd.DataFrame(DEFAULT_FOODS, columns=FOOD_COLS)
    df["Tags"] = df["Tags"].fillna("").astype(str)
    df["Protein per 10 BDT (g)"] = (df["Protein g/100g"] * 10 / df["Price (BDT/kg)"] * 10).round(1)
    df["kcal per 10 BDT"] = (df["kcal/100g"] * 10 / df["Price (BDT/kg)"] * 10).round(0)
    return df


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

    df = foods.copy()
    df = df[~df["Food"].isin(exclude)]
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
    for score, row, matched, v in scored:
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

    kcal_total = float((grams * kcal).sum())
    prot_total = float((grams * prot).sum())
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
        "kcal_pp": kcal_total / people,
        "protein_pp": prot_total / people,
        "meals": meals,
        "meal_kcal": meal_kcal,
        "people": people,
    }


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


def build_pdf(user, r, assessments, recs, plan):
    L = dict(new_x="LMARGIN", new_y="NEXT")
    pdf = _PDF()
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()

    pdf.set_fill_color(15, 118, 110)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_xy(12, 7)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "Food Security Risk Assessment Report", **L)
    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _t(f"Prepared for {user['name']} ({user['email']})   |   {r['date']:%d %b %Y}"), **L)
    pdf.set_y(36)

    def heading(text):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 118, 110)
        pdf.cell(0, 7, _t(text), **L)
        pdf.set_draw_color(200, 220, 210)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(31, 41, 55)
        pdf.set_font("Helvetica", "", 10)

    rgb = {"Low": (22, 163, 74), "Moderate": (245, 158, 11), "High": (220, 38, 38)}[r["level"]]
    pdf.set_fill_color(*rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 14, _t(f"   {r['level']} risk   |   Risk score: {r['score']} / 100"), fill=True, **L)
    pdf.set_text_color(31, 41, 55)

    heading("Prediction probabilities")
    for lvl in LEVELS:
        p = r["probs"][lvl]
        y = pdf.get_y()
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(28, 6, lvl)
        x = pdf.get_x()
        pdf.set_fill_color(230, 232, 235)
        pdf.rect(x, y + 1.6, 100, 3, "F")
        c = {"Low": (22, 163, 74), "Moderate": (245, 158, 11), "High": (220, 38, 38)}[lvl]
        pdf.set_fill_color(*c)
        pdf.rect(x, y + 1.6, max(0.5, 100 * p), 3, "F")
        pdf.set_x(x + 104)
        pdf.cell(0, 6, f"{p * 100:.0f}%", **L)

    heading("Household information")
    i = r["inputs"]
    info = [
        ("Household size", i["size"]),
        ("Monthly income (BDT)", f"{i['income']:,}"),
        ("Monthly food expenditure (BDT)", f"{i['food_exp']:,}"),
        ("Meals per day", i["meals"]),
        ("Dietary diversity score (0-12)", i["diversity"]),
        ("Employment", i["employment"]),
        ("Child under 5", "Yes" if i["child"] else "No"),
        ("Difficulty accessing food (1-5)", i["access"]),
    ]
    for k, v in info:
        pdf.cell(75, 6, _t(k))
        pdf.cell(0, 6, _t(v), **L)

    heading("Risk factor analysis (red raises risk, green lowers it)")
    contrib = r["contributions"]
    mx = max((abs(v) for v in contrib.values()), default=1) or 1
    for f in sorted(FACTORS, key=lambda k: -contrib[k]):
        v = contrib[f]
        y = pdf.get_y()
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(88, 6, _t(f"{f}: {describe_input(i, f)}"))
        cx = 150
        w = 45 * abs(v) / mx
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

    heading("Recommendations")
    drivers = risk_drivers(contrib, 3)
    if drivers:
        for f in drivers:
            pdf.multi_cell(0, 5.5, _t(f"- {f}: {ADVICE[f]}"), **L)
    else:
        pdf.multi_cell(0, 5.5, "- No major risk factors found. Keep up your current food habits.", **L)

    heading("Suggested affordable foods")
    for item in recs:
        row = item["row"]
        why = ", ".join(TAG_LABEL[t] for t in item["matched"]) or "good value for money"
        pdf.multi_cell(
            0, 5.5,
            _t(f"- {row['Food']} ({row['Category']}): {row['Price (BDT/kg)']:.0f} BDT/kg, "
               f"{row['Protein per 10 BDT (g)']:.1f} g protein per 10 BDT - {why}"),
            **L,
        )

    heading("Budget meal plan (demo targets: 2100 kcal and 50 g protein per person)")
    if plan:
        pdf.multi_cell(
            0, 5.5,
            _t(f"Household of {plan['people']}: minimum cost about {plan['total']:.0f} BDT per day "
               f"({plan['total'] * 7:.0f} BDT per week)."),
            **L,
        )
        for _, row in plan["plan"].iterrows():
            pdf.cell(0, 5.5, _t(f"- {row['Food']}: {row['Grams per day']} g/day ({row['Cost per day (BDT)']:.1f} BDT)"), **L)
    else:
        pdf.multi_cell(0, 5.5, "No feasible plan with the current food list.", **L)

    heading("Risk trend")
    if assessments:
        recent = assessments[-8:]
        for a in recent:
            pdf.cell(0, 5.5, _t(f"{a['date']:%d %b %Y}:  score {a['score']}  ({a['level']})"), **L)
        if len(assessments) >= 2:
            diff = assessments[-1]["score"] - assessments[0]["score"]
            word = "improved" if diff < 0 else "worsened" if diff > 0 else "not changed"
            pdf.ln(1)
            pdf.multi_cell(0, 5.5, _t(f"Since the first assessment the score has {word} by {abs(diff)} points."), **L)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, _t(DISCLAIMER + " Results come from " + ("a trained model." if r.get("source") == "model" else "demo data and rule-based placeholders.")), **L)
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


def source_caption(r):
    if r.get("source") == "model":
        st.caption("Prediction made by the trained ML model.")
    else:
        st.caption("Demo mode: prediction comes from rule-based placeholder logic. It switches to your trained model automatically when model.pkl is added.")


# ===================================================================== auth
def page_auth():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="auth-hero">
              <div class="logo">🍚</div>
              <div class="a-title">Food Security<br>Risk Assessment</div>
              <div class="a-sub">Understand your household's food security, get personalised guidance
              and plan affordable, nutritious meals.</div>
              <ul>
                <li>🔍&nbsp; Risk screening in under a minute</li>
                <li>🧠&nbsp; See which factors drive your result</li>
                <li>🥗&nbsp; Budget meal plans that meet nutrition targets</li>
                <li>📈&nbsp; Track your progress over time</li>
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

            st.divider()
            st.button(
                "Continue with demo account",
                icon=":material/rocket_launch:",
                on_click=demo_login,
                key="demo_btn",
                help="Explore the app with sample data, no sign-up needed.",
            )
            st.caption(DISCLAIMER)


# =================================================================== sidebar
NAV_SECTIONS = [
    ("Start", [("Home", ":material/home:"), ("Assessment", ":material/assignment:")]),
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
    ("Assessment", "📝", "Food Security Assessment", "Enter income, food spending, meals and diet details for your household."),
    ("Risk Prediction", "🎯", "Risk Prediction & Score", "Get a Low / Moderate / High risk level, probabilities and a 0–100 score."),
    ("Risk Factors", "🔍", "Risk Factor Analysis", "See which inputs pushed your risk up or down (SHAP-style)."),
    ("Food Recommendations", "🥬", "Food Recommendations", "Affordable, nutritious foods matched to your risk factors."),
    ("Meal Planner", "🍽️", "Budget Meal Planner", "The cheapest meal plan within your daily or weekly budget."),
    ("Trends", "📈", "Risk Trend Tracking", "Save each assessment and compare results over time."),
    ("Dashboard", "🧭", "Dashboard", "Personal and community charts and key numbers."),
    ("Report", "📄", "Report", "Download a structured PDF report of your results."),
]


def page_home():
    u = st.session_state["user"]
    first = html.escape(u["name"].strip().split()[0])
    st.markdown(
        f'<div class="hero"><div class="h-title">Hello, {first} 👋</div>'
        '<div class="h-sub">Check your household\'s food security, understand what drives the result '
        "and plan affordable, nutritious meals.</div></div>",
        unsafe_allow_html=True,
    )
    assessments = load_assessments(u["id"])
    hist = history_df(assessments)
    last = st.session_state["last"]
    c1, c2, c3, c4 = st.columns(4)
    with c1.container(border=True):
        st.metric("Assessments", len(hist))
    with c2.container(border=True):
        st.metric("Latest risk level", last["level"] if last else "—")
    with c3.container(border=True):
        st.metric("Latest score", f"{last['score']}/100" if last else "—")
    with c4.container(border=True):
        if len(hist) >= 2:
            diff = int(hist["Risk score"].iloc[-1] - hist["Risk score"].iloc[-2])
            st.metric("Change vs previous (lower is better)", trend_text(diff))
        else:
            st.metric("Change vs previous", "—")

    if last is None:
        st.info("You have no assessments yet. Start with the first card below.", icon=":material/info:")

    st.subheader("All features")
    for start in (0, 4):
        cols = st.columns(4)
        for col, (page, emoji, title, desc) in zip(cols, FEATURE_CARDS[start:start + 4]):
            with col.container(border=True):
                st.markdown(
                    f'<div class="feature-icon">{emoji}</div><div class="f-title">{title}</div>'
                    f'<div class="f-desc">{html.escape(desc)}</div>',
                    unsafe_allow_html=True,
                )
                st.button("Open", icon=":material/arrow_forward:", on_click=goto, args=(page,), key=f"home_{page}")


def page_assessment():
    st.title("📝 Food Security Assessment")
    st.caption("Fill in the form and press **Assess risk**. Sample values are pre-filled.")
    with st.container(border=True):
        with st.form("assess", border=False):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("##### 🏠 Household")
                st.number_input("Household size", 1, 20, 4, key="in_size")
                st.number_input("Monthly household income (BDT)", 0, 1_000_000, 20000, step=500, key="in_income")
                st.number_input("Monthly food expenditure (BDT)", 0, 1_000_000, 9000, step=500, key="in_food")
                st.selectbox("Employment", ["Regular", "Daily wage", "Unemployed"], key="in_emp")
            with c2:
                st.markdown("##### 🍽️ Food & nutrition")
                st.selectbox("Meals per day", [1, 2, 3, 4], index=2, key="in_meals")
                st.slider("Dietary diversity score (0–12 food groups eaten)", 0, 12, 6, key="in_div")
                st.slider("Difficulty accessing food (1 = easy, 5 = very hard)", 1, 5, 2, key="in_access")
                st.radio("Child under 5 in the household?", ["No", "Yes"], horizontal=True, key="in_child")
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
        st.caption("One overall score built from the 7 indicators on the right.")
    with c2.container(border=True):
        pdf = pd.DataFrame({"Risk level": LEVELS, "Probability (%)": [r["probs"][k] * 100 for k in LEVELS]})
        fig = px.bar(pdf, x="Risk level", y="Probability (%)", color="Risk level", color_discrete_map=COLORS)
        fig.update_layout(showlegend=False, yaxis_range=[0, 100], title="Prediction probability",
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
    source_caption(r)

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
    explain = r.get("explain", "rules")
    st.caption(
        "Shows how much each input **raised** (red) or **lowered** (green) your risk compared with a typical household. "
        + ("Computed with **SHAP** from the trained model." if explain == "SHAP"
           else "Demo version: computed with rule-based points. It becomes real SHAP when model.pkl and the `shap` library are added.")
    )
    contrib = r["contributions"]
    df = pd.DataFrame(
        {
            "Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS],
            "key": FACTORS,
            "Effect": [contrib[f] for f in FACTORS],
        }
    )
    df["Direction"] = np.where(df["Effect"] > 0, "Raises risk", "Lowers risk")
    df = df.sort_values("Effect")

    with st.container(border=True):
        fig = px.bar(df, x="Effect", y="Factor", orientation="h", color="Direction",
                     color_discrete_map={"Raises risk": "#dc2626", "Lowers risk": "#16a34a"},
                     title="What pushed your risk up or down")
        fig.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
        fig.add_vline(x=0, line_color="#6b7280")
        st.plotly_chart(fig)

    mx = max(abs(v) for v in contrib.values()) or 1
    table = pd.DataFrame(
        {
            "Factor": [f"{FACTOR_ICON[f]} {f}" for f in FACTORS],
            "Your value": [describe_input(r["inputs"], f) for f in FACTORS],
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
            st.success("Nothing is pushing your risk above a typical household.")

    with st.container(border=True):
        imp = global_importance().sort_values("Mean |effect|")
        fig = px.bar(imp, x="Mean |effect|", y="Factor", orientation="h",
                     title="Which factors matter most across the whole community (demo data)")
        fig.update_traces(marker_color="#0f9d58")
        fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig)
    st.button("Get food suggestions", icon=":material/eco:", on_click=goto, args=("Food Recommendations",), key="rf_fr")
    st.caption(DISCLAIMER)


def page_food_recommendations():
    st.title("🥬 Personalised Food Recommendations")
    r = st.session_state["last"]
    if r is None:
        st.info("Showing general low-cost, nutritious foods. Complete an assessment to personalise this list.",
                icon=":material/info:")
    foods = load_foods()
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.4])
        n = c1.slider("How many foods", 3, 9, 6, key="fr_n")
        max_price = c2.slider("Max price (BDT/kg)", 20, int(foods["Price (BDT/kg)"].max()), int(foods["Price (BDT/kg)"].max()),
                              step=10, key="fr_price")
        excl = c3.multiselect("Exclude foods (allergy / preference)", list(foods["Food"]), key="fr_excl")

    picks, drivers, tags = recommend_foods(r, n=n, max_price=max_price, exclude=excl)
    if r:
        chips = "".join(f'<span class="chip red">{FACTOR_ICON[f]} {html.escape(f)}</span>' for f in drivers)
        need = "".join(f'<span class="chip">{html.escape(TAG_LABEL[t])}</span>' for t in tags)
        st.markdown(
            "**Chosen because of your risk factors:** " + (chips or '<span class="chip">no major risk factors</span>')
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
    st.caption("Foods, prices and nutrition values are approximate demo data. " + DISCLAIMER)


def page_meal_planner():
    st.title("🍽️ Budget Meal Planner")
    st.caption("Finds the **cheapest combination of foods** that meets your calorie and protein targets, "
               "then splits it into breakfast, lunch and dinner.")
    r = st.session_state["last"]
    default_people = int(r["inputs"]["size"]) if r else 4
    foods = load_foods()

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        people = c1.number_input("People in household", 1, 20, default_people, key="mp_people")
        period = c2.radio("Budget period", ["Daily", "Weekly"], horizontal=True, key="mp_period")
        budget = c3.number_input(f"{period} food budget (BDT)", 50, 100000, 300 if period == "Daily" else 2100,
                                 step=50, key="mp_budget")
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
    daily_budget = budget if period == "Daily" else budget / 7.0

    m1, m2, m3, m4 = st.columns(4)
    with m1.container(border=True):
        st.metric("Minimum cost per day", f"{total:.0f} BDT")
    with m2.container(border=True):
        st.metric("Minimum cost per week", f"{total * 7:.0f} BDT")
    with m3.container(border=True):
        st.metric("Cost per person per day", f"{total / people:.0f} BDT")
    with m4.container(border=True):
        st.metric("Your budget (per day)", f"{daily_budget:.0f} BDT")
    if total <= daily_budget:
        st.success(f"Your budget covers this plan with about {daily_budget - total:.0f} BDT per day to spare.")
    else:
        st.warning(f"Your budget is about {total - daily_budget:.0f} BDT per day short of the cheapest plan that meets the targets.")

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
        st.markdown(f"**Estimated weekly total: {shop['Cost per week (BDT)'].sum():,} BDT** for {people} people")
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
    st.caption("Foods, prices and nutrition values are approximate demo data. " + DISCLAIMER)


def page_trends():
    st.title("📈 Risk Trend Tracking")
    u = st.session_state["user"]
    assessments = load_assessments(u["id"])
    if not assessments:
        st.info("No assessments saved yet. Complete an assessment, or add sample history to preview this page.",
                icon=":material/info:")
        b1, b2 = st.columns([1, 3])
        b1.button("Start assessment", icon=":material/arrow_forward:", on_click=goto, args=("Assessment",), key="tr_start")
        b2.button("Add sample history", icon=":material/auto_awesome:", on_click=seed_history_cb, key="tr_seed")
        return
    hist = history_df(assessments)

    m1, m2, m3, m4 = st.columns(4)
    with m1.container(border=True):
        st.metric("Assessments", len(hist))
    with m2.container(border=True):
        st.metric("Best score", int(hist["Risk score"].min()))
    with m3.container(border=True):
        if len(hist) >= 2:
            diff = int(hist["Risk score"].iloc[-1] - hist["Risk score"].iloc[-2])
            st.metric("Latest vs previous", trend_text(diff))
        else:
            st.metric("Latest score", int(hist["Risk score"].iloc[-1]))
    with m4.container(border=True):
        if len(hist) >= 2:
            total = int(hist["Risk score"].iloc[-1] - hist["Risk score"].iloc[0])
            st.metric("Since first", trend_text(total))
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


def page_dashboard():
    st.title("🧭 Dashboard")
    tab_me, tab_com = st.tabs([":material/person: My dashboard", ":material/groups: Community"])

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

    with tab_com:
        df, src = community_data()
        st.caption("Demo data for 200 sample households." if src == "demo" else "Loaded from community_data.csv.")
        with st.container(border=True):
            f1, f2, f3 = st.columns(3)
            emp = f1.multiselect("Employment", sorted(df["Employment"].unique()), default=sorted(df["Employment"].unique()), key="cm_emp")
            hs_min, hs_max = int(df["Household size"].min()), int(df["Household size"].max())
            hs = f2.slider("Household size", hs_min, hs_max, (hs_min, hs_max), key="cm_hs")
            lv = f3.multiselect("Risk level", LEVELS, default=LEVELS, key="cm_lv")
        d = df[df["Employment"].isin(emp) & df["Household size"].between(hs[0], hs[1]) & df["Risk level"].isin(lv)]
        if d.empty:
            st.warning("No households match these filters.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            with k1.container(border=True):
                st.metric("Households", len(d))
            with k2.container(border=True):
                st.metric("High risk", f"{(d['Risk level'] == 'High').mean() * 100:.0f}%")
            with k3.container(border=True):
                st.metric("Average risk score", f"{d['Risk score'].mean():.0f}")
            with k4.container(border=True):
                st.metric("Avg dietary diversity", f"{d['Dietary diversity'].mean():.1f}")

            c1, c2 = st.columns(2)
            with c1.container(border=True):
                counts = d["Risk level"].value_counts().reindex(LEVELS).fillna(0).reset_index()
                counts.columns = ["Risk level", "Households"]
                fig = px.pie(counts, names="Risk level", values="Households", hole=0.5,
                             color="Risk level", color_discrete_map=COLORS, title="Risk level distribution")
                fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
                st.plotly_chart(fig)
            with c2.container(border=True):
                e = d.groupby("Employment")["Risk level"].apply(lambda s: (s == "High").mean() * 100).reset_index()
                e.columns = ["Employment", "High-risk households (%)"]
                fig = px.bar(e, x="Employment", y="High-risk households (%)", title="High risk by employment type")
                fig.update_traces(marker_color="#0f9d58")
                fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
                st.plotly_chart(fig)
            c3, c4 = st.columns(2)
            with c3.container(border=True):
                h = d.groupby("Household size")["Risk score"].mean().reset_index()
                fig = px.bar(h, x="Household size", y="Risk score", title="Average risk score by household size")
                fig.update_traces(marker_color="#0f766e")
                fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=330)
                st.plotly_chart(fig)
            with c4.container(border=True):
                fig = px.box(d, x="Risk level", y="Dietary diversity", color="Risk level",
                             category_orders={"Risk level": LEVELS}, color_discrete_map=COLORS,
                             title="Dietary diversity by risk level")
                fig.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10), height=330)
                st.plotly_chart(fig)
            with st.container(border=True):
                fig = px.scatter(d, x="Income per person", y="Risk score", color="Risk level",
                                 category_orders={"Risk level": LEVELS}, color_discrete_map=COLORS,
                                 title="Income per person vs risk score")
                fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=360)
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
        recs, _, _ = recommend_foods(r, n=6)
        plan = compute_plan(int(r["inputs"]["size"]), 2100, 50, True, ())
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


# ===================================================================== main
if "user" not in st.session_state:
    st.session_state["user"] = None
if "last" not in st.session_state:
    st.session_state["last"] = None
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
        "Risk Prediction": page_prediction,
        "Risk Factors": page_risk_factors,
        "Food Recommendations": page_food_recommendations,
        "Meal Planner": page_meal_planner,
        "Trends": page_trends,
        "Dashboard": page_dashboard,
        "Report": page_report,
        "Profile": page_profile,
    }
    ROUTES[st.session_state["page"]]()
