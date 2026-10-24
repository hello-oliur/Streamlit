"""Food Security & Malnutrition Risk Assessment - DEMO app with login.

Accounts and assessments are stored in a local SQLite file. The risk
"prediction" is a rule-based placeholder that will later be replaced by the
trained ML model (model.pkl from Google Colab).
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
import streamlit as st
from scipy.optimize import linprog

st.set_page_config(page_title="Food Security Risk Assessment", page_icon="🍚", layout="wide")

DB_PATH = os.environ.get("FS_DB_PATH", "foodsecurity.db")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SEC_QUESTION = "What is your favourite food?"
DEMO_EMAIL = "demo@foodsecurity.app"

LEVELS = ["Low", "Moderate", "High"]
COLORS = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#dc2626"}
LEVEL_STYLE = {
    "Low": ("✅", "linear-gradient(135deg,#15803d,#22c55e)", "Your household shows a low food-security risk. Keep it up!"),
    "Moderate": ("⚠️", "linear-gradient(135deg,#d97706,#fbbf24)", "Some risk factors need attention. See the suggestions below."),
    "High": ("🚨", "linear-gradient(135deg,#b91c1c,#f87171)", "Several risk factors are present. Please act on the recommendations."),
}
DISCLAIMER = "Screening guidance only. This is not a medical diagnosis."

FACTOR_ICON = {
    "Income per person": "💰",
    "Meals per day": "🍽️",
    "Dietary diversity": "🥦",
    "Difficulty accessing food": "🛒",
    "Employment": "💼",
    "Child under 5": "🧒",
    "Food spending share": "🧾",
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

FOODS = pd.DataFrame(
    [
        ("Rice", 60, 360, 7.0, 400),
        ("Lentils (dal)", 140, 350, 25.0, 100),
        ("Eggs", 170, 143, 12.6, 100),
        ("Potato", 40, 77, 2.0, 300),
        ("Spinach (palong shak)", 30, 23, 2.9, 200),
        ("Small fish", 300, 120, 18.0, 150),
        ("Banana", 60, 89, 1.1, 150),
        ("Soybean oil", 160, 884, 0.0, 30),
        ("Milk", 80, 61, 3.2, 250),
    ],
    columns=["Food", "Price (BDT/kg)", "kcal/100g", "Protein g/100g", "Max g/person/day"],
)

# Sample inputs used for demo history (risk improves over time).
SEED_INPUTS = [
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
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}

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
.feature-icon {font-size:1.8rem;}

[data-testid="stSidebar"] [data-testid="stButton"] {width:100%;}
[data-testid="stSidebar"] [data-testid="stButton"] > button {width:100%; justify-content:flex-start;
        border-radius:12px; padding:.55rem .9rem; font-weight:600;}
[data-testid="stFormSubmitButton"] > button {width:100%;}
button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
    background:#0f9d58; border-color:#0f9d58; color:#fff;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {background:#0b7a45; border-color:#0b7a45; color:#fff;}
[data-testid="stMetricValue"] {font-weight:800;}
</style>
"""


# ================================================================ database
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


# ================================================================ security
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


# ================================================================ users
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


# ================================================================ assessments
def level_from_score(score):
    return "Low" if score < 29 else "Moderate" if score < 57 else "High"


def demo_predict(d):
    """DEMO ONLY: rule-based scoring. Replace with model.predict() later."""
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
    total = sum(c.values())

    centers = np.array([1.0, 6.0, 11.0])
    logits = -(((total - centers) / 3.0) ** 2)
    probs = np.exp(logits) / np.exp(logits).sum()
    score = int(min(100, round(total / 14 * 100)))
    return {
        "level": level_from_score(score),
        "score": score,
        "probs": {k: float(v) for k, v in zip(LEVELS, probs)},
        "contributions": {k: float(v) for k, v in c.items()},
        "inputs": d,
    }


def save_assessment(user_id, res, when):
    payload = {"probs": res["probs"], "contributions": res["contributions"], "inputs": res["inputs"]}
    q(
        "INSERT INTO assessments (user_id, created_at, score, level, data) VALUES (?,?,?,?,?)",
        (user_id, when.isoformat(), res["score"], res["level"], json.dumps(payload, default=float)),
        write=True,
    )


def latest_result(user_id):
    row = q(
        "SELECT * FROM assessments WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
        one=True,
    )
    if not row:
        return None
    d = json.loads(row["data"])
    return {
        "level": row["level"],
        "score": row["score"],
        "probs": d["probs"],
        "contributions": d["contributions"],
        "inputs": d["inputs"],
        "date": date.fromisoformat(row["created_at"]),
    }


def load_history(user_id):
    rows = q(
        "SELECT created_at, score, level FROM assessments WHERE user_id=? ORDER BY created_at ASC, id ASC",
        (user_id,),
    )
    return pd.DataFrame(
        {
            "Date": pd.to_datetime([r["created_at"] for r in rows]),
            "Risk score": [r["score"] for r in rows],
            "Risk level": [r["level"] for r in rows],
        }
    )


def seed_history(user_id):
    for days_ago, inputs in SEED_INPUTS:
        save_assessment(user_id, demo_predict(inputs), date.today() - timedelta(days=days_ago))


def seed_history_cb():
    uid = st.session_state["user"]["id"]
    seed_history(uid)
    st.session_state["last"] = latest_result(uid)


def run_assessment():
    """Form callback: runs before the page re-renders."""
    s = st.session_state
    data = {
        "size": s["in_size"],
        "income": s["in_income"],
        "food_exp": s["in_food"],
        "meals": s["in_meals"],
        "diversity": s["in_div"],
        "employment": s["in_emp"],
        "child": s["in_child"] == "Yes",
        "access": s["in_access"],
    }
    res = demo_predict(data)
    res["date"] = date.today()
    save_assessment(s["user"]["id"], res, res["date"])
    s["last"] = res
    s["page"] = "Results"


def go(page):
    st.session_state["page"] = page


@st.cache_data
def community_data():
    """Demo data for 200 households (random, for the dashboard only)."""
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(200):
        size = int(rng.integers(1, 9))
        income = int(max(3000, rng.lognormal(9.9, 0.5)))
        d = {
            "size": size,
            "income": income,
            "food_exp": int(income * rng.uniform(0.3, 0.75)),
            "meals": int(rng.choice([1, 2, 3, 4], p=[0.05, 0.25, 0.6, 0.1])),
            "diversity": int(np.clip(round(rng.normal(6, 2.2)), 0, 12)),
            "employment": str(rng.choice(["Regular", "Daily wage", "Unemployed"], p=[0.45, 0.4, 0.15])),
            "child": bool(rng.integers(0, 2)),
            "access": int(rng.integers(1, 6)),
        }
        r = demo_predict(d)
        rows.append(
            {
                "Household size": size,
                "Income per person": income / size,
                "Employment": d["employment"],
                "Dietary diversity": d["diversity"],
                "Risk score": r["score"],
                "Risk level": r["level"],
            }
        )
    return pd.DataFrame(rows)


# ================================================================ auth page
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


# ================================================================ sidebar
NAV = [
    ("Home", ":material/home:"),
    ("Assessment", ":material/assignment:"),
    ("Results", ":material/insights:"),
    ("Meal Planner", ":material/restaurant:"),
    ("History", ":material/history:"),
    ("Dashboard", ":material/dashboard:"),
    ("Report", ":material/description:"),
    ("Profile", ":material/account_circle:"),
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
        st.write("")
        for label, icon in NAV:
            st.button(
                label,
                icon=icon,
                key=f"nav_{label}",
                on_click=go,
                args=(label,),
                type="primary" if st.session_state["page"] == label else "secondary",
            )
        st.divider()
        st.button("Log out", icon=":material/logout:", on_click=logout, key="logout_btn")
        st.caption(DISCLAIMER)
        st.caption("Demo mode: rule-based placeholder instead of the ML model.")


# ================================================================ pages
def page_home():
    u = st.session_state["user"]
    first = html.escape(u["name"].strip().split()[0])
    st.markdown(
        f'<div class="hero"><div class="h-title">Hello, {first} 👋</div>'
        '<div class="h-sub">Check your household\'s food security, understand what drives the result '
        "and plan affordable, nutritious meals.</div></div>",
        unsafe_allow_html=True,
    )
    hist = load_history(u["id"])
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
            st.metric("Change vs previous", f"{diff:+d}", delta=diff, delta_color="inverse")
        else:
            st.metric("Change vs previous", "—")

    if last is None:
        st.info("You have no assessments yet. Start your first one below.", icon=":material/info:")

    st.subheader("What would you like to do?")
    f1, f2, f3 = st.columns(3)
    with f1.container(border=True):
        st.markdown('<div class="feature-icon">📝</div>', unsafe_allow_html=True)
        st.markdown("**New assessment**")
        st.caption("Answer a few questions and get your risk level, key factors and recommendations.")
        st.button("Start assessment", icon=":material/arrow_forward:", on_click=go, args=("Assessment",), key="h_assess")
    with f2.container(border=True):
        st.markdown('<div class="feature-icon">🥗</div>', unsafe_allow_html=True)
        st.markdown("**Budget meal plan**")
        st.caption("Find the cheapest food mix that meets calorie and protein targets for your household.")
        st.button("Plan meals", icon=":material/arrow_forward:", on_click=go, args=("Meal Planner",), key="h_meal")
    with f3.container(border=True):
        st.markdown('<div class="feature-icon">📈</div>', unsafe_allow_html=True)
        st.markdown("**Track progress**")
        st.caption("See how your risk score changes over time and download your report.")
        st.button("View history", icon=":material/arrow_forward:", on_click=go, args=("History",), key="h_hist")


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


def page_results():
    st.title("📊 Assessment Results")
    r = st.session_state["last"]
    if r is None:
        st.info("No assessment yet. Go to **Assessment** and submit the form.", icon=":material/info:")
        st.button("Start assessment", icon=":material/arrow_forward:", on_click=go, args=("Assessment",), key="r_start")
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
        st.metric("Confidence", f"{max(r['probs'].values()) * 100:.0f}%")
    with m3.container(border=True):
        st.metric("Assessment date", r["date"].strftime("%d %b %Y"))

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        pdf = pd.DataFrame({"Risk level": LEVELS, "Probability (%)": [r["probs"][k] * 100 for k in LEVELS]})
        fig = px.bar(pdf, x="Risk level", y="Probability (%)", color="Risk level", color_discrete_map=COLORS)
        fig.update_layout(showlegend=False, yaxis_range=[0, 100], title="Prediction probability",
                          height=320, margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig)
    with c2.container(border=True):
        cdf = pd.DataFrame({"Factor": list(r["contributions"].keys()), "Contribution": list(r["contributions"].values())})
        cdf = cdf.sort_values("Contribution")
        fig2 = px.bar(cdf, x="Contribution", y="Factor", orientation="h", color="Contribution",
                      color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"])
        fig2.update_layout(coloraxis_showscale=False, title="Why this result?", height=320,
                           margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig2)
        st.caption("Demo explainability chart. The final app will use SHAP.")

    st.subheader("💡 Recommendations")
    top = [k for k, v in sorted(r["contributions"].items(), key=lambda kv: -kv[1]) if v > 0][:3]
    if top:
        cols = st.columns(len(top))
        for col, k in zip(cols, top):
            with col.container(border=True):
                st.markdown(f"##### {FACTOR_ICON[k]} {k}")
                st.write(ADVICE[k])
    else:
        st.success("No major risk factors found. Keep up your current food habits.")
    st.caption(DISCLAIMER)


def page_meal_planner():
    st.title("🥗 Budget Meal Planner")
    st.caption("Finds the **cheapest combination of foods** that meets your calorie and protein targets.")
    default_people = st.session_state["last"]["inputs"]["size"] if st.session_state["last"] else 4

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        people = c1.number_input("People", 1, 20, int(default_people))
        budget = c2.number_input("Daily food budget (BDT)", 50, 5000, 300, step=10)
        kcal_pp = c3.slider("Calories per person", 1500, 2800, 2100, step=50)
        prot_pp = c4.slider("Protein per person (g)", 30, 90, 50, step=5)

    cost = FOODS["Price (BDT/kg)"].to_numpy() / 1000.0
    kcal = FOODS["kcal/100g"].to_numpy() / 100.0
    prot = FOODS["Protein g/100g"].to_numpy() / 100.0
    bounds = [(0, m * people) for m in FOODS["Max g/person/day"]]
    res = linprog(
        cost,
        A_ub=-np.vstack([kcal, prot]),
        b_ub=-np.array([kcal_pp * people, prot_pp * people], dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        st.error("No plan can meet these targets with the demo foods. Try lowering the targets.")
        return

    grams = res.x
    plan = FOODS[["Food"]].copy()
    plan["Grams per day"] = grams.round(0).astype(int)
    plan["Cost per day (BDT)"] = (grams * cost).round(1)
    plan = plan[plan["Grams per day"] > 0].reset_index(drop=True)
    total = float(res.fun)

    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("Minimum daily cost (BDT)", f"{total:.0f}")
    with m2.container(border=True):
        st.metric("Weekly cost (BDT)", f"{total * 7:.0f}")
    with m3.container(border=True):
        st.metric("Your budget (BDT/day)", f"{budget}")
    if total <= budget:
        st.success(f"Your budget covers this plan with about {budget - total:.0f} BDT per day to spare.")
    else:
        st.warning(f"Your budget is about {total - budget:.0f} BDT per day short of the cheapest plan that meets the targets.")

    t1, t2 = st.columns([1.1, 1])
    with t1.container(border=True):
        st.markdown("##### Suggested daily quantities (whole household)")
        st.dataframe(plan, hide_index=True)
    with t2.container(border=True):
        fig = px.pie(plan, names="Food", values="Cost per day (BDT)", hole=0.45, title="Where the money goes")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig)
    st.caption("Foods, prices and nutrition values are approximate demo data. " + DISCLAIMER)


def page_history():
    st.title("📈 Risk Trend Tracking")
    u = st.session_state["user"]
    hist = load_history(u["id"])
    if hist.empty:
        st.info("No assessments saved yet. Complete an assessment, or add sample history to preview this page.",
                icon=":material/info:")
        b1, b2 = st.columns([1, 3])
        b1.button("Start assessment", icon=":material/arrow_forward:", on_click=go, args=("Assessment",), key="hs_start")
        b2.button("Add sample history", icon=":material/auto_awesome:", on_click=seed_history_cb, key="hs_seed")
        return

    with st.container(border=True):
        fig = px.line(hist, x="Date", y="Risk score", markers=True, title="Risk score over time (lower is better)")
        fig.add_hline(y=29, line_dash="dot", annotation_text="Low / Moderate")
        fig.add_hline(y=57, line_dash="dot", annotation_text="Moderate / High")
        fig.update_traces(line=dict(color="#0f9d58", width=3))
        fig.update_layout(yaxis_range=[0, 100], margin=dict(t=50, b=20, l=10, r=10), height=380)
        st.plotly_chart(fig)

    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("Assessments", len(hist))
    with m2.container(border=True):
        st.metric("Best score", int(hist["Risk score"].min()))
    with m3.container(border=True):
        if len(hist) >= 2:
            diff = int(hist["Risk score"].iloc[-1] - hist["Risk score"].iloc[-2])
            st.metric("Latest vs previous", int(hist["Risk score"].iloc[-1]), delta=diff, delta_color="inverse")
        else:
            st.metric("Latest score", int(hist["Risk score"].iloc[-1]))

    with st.container(border=True):
        st.markdown("##### All assessments")
        show = hist.copy()
        show["Date"] = show["Date"].dt.date
        st.dataframe(show.sort_values("Date", ascending=False), hide_index=True)


def page_dashboard():
    st.title("🧭 Community Dashboard")
    st.caption("Demo data for 200 sample households.")
    df = community_data()
    k1, k2, k3, k4 = st.columns(4)
    with k1.container(border=True):
        st.metric("Households", len(df))
    with k2.container(border=True):
        st.metric("High risk", f"{(df['Risk level'] == 'High').mean() * 100:.0f}%")
    with k3.container(border=True):
        st.metric("Average risk score", f"{df['Risk score'].mean():.0f}")
    with k4.container(border=True):
        st.metric("Avg dietary diversity", f"{df['Dietary diversity'].mean():.1f}")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        counts = df["Risk level"].value_counts().reindex(LEVELS).reset_index()
        counts.columns = ["Risk level", "Households"]
        fig = px.pie(counts, names="Risk level", values="Households", hole=0.5,
                     color="Risk level", color_discrete_map=COLORS, title="Risk level distribution")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig)
    with c2.container(border=True):
        emp = df.groupby("Employment")["Risk level"].apply(lambda s: (s == "High").mean() * 100).reset_index()
        emp.columns = ["Employment", "High-risk households (%)"]
        fig = px.bar(emp, x="Employment", y="High-risk households (%)", title="High risk by employment type")
        fig.update_traces(marker_color="#0f9d58")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig)

    with st.container(border=True):
        fig = px.scatter(df, x="Income per person", y="Risk score", color="Risk level",
                         color_discrete_map=COLORS, title="Income per person vs risk score")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=380)
        st.plotly_chart(fig)


def page_report():
    st.title("📄 Report")
    r = st.session_state["last"]
    if r is None:
        st.info("Complete an assessment first, then come back to download your report.", icon=":material/info:")
        return
    u = st.session_state["user"]
    i = r["inputs"]
    top = [k for k, v in sorted(r["contributions"].items(), key=lambda kv: -kv[1]) if v > 0][:3]
    lines = [
        "FOOD SECURITY RISK ASSESSMENT REPORT (DEMO)",
        f"Prepared for: {u['name']} ({u['email']})",
        f"Date: {r['date']}",
        "",
        f"Predicted risk level: {r['level']}",
        f"Risk score: {r['score']} / 100",
        "Probabilities: " + ", ".join(f"{k} {r['probs'][k] * 100:.0f}%" for k in LEVELS),
        "",
        "Household information",
        f"- Household size: {i['size']}",
        f"- Monthly income (BDT): {i['income']}",
        f"- Monthly food expenditure (BDT): {i['food_exp']}",
        f"- Meals per day: {i['meals']}",
        f"- Dietary diversity score: {i['diversity']}",
        f"- Employment: {i['employment']}",
        f"- Child under 5: {'Yes' if i['child'] else 'No'}",
        f"- Difficulty accessing food (1-5): {i['access']}",
        "",
        "Main contributing factors and recommendations",
    ]
    lines += [f"- {k}: {ADVICE[k]}" for k in top] or ["- No major risk factors found."]
    lines += ["", DISCLAIMER]
    text = "\n".join(lines)

    with st.container(border=True):
        st.text(text)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download report", text, file_name="food_security_report.txt", icon=":material/download:")
    hist = load_history(u["id"]).copy()
    hist["Date"] = hist["Date"].dt.date
    c2.download_button("Download history", hist.to_csv(index=False), file_name="assessment_history.csv",
                       icon=":material/table_view:")


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


# ================================================================ main
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
        "Results": page_results,
        "Meal Planner": page_meal_planner,
        "History": page_history,
        "Dashboard": page_dashboard,
        "Report": page_report,
        "Profile": page_profile,
    }
    ROUTES[st.session_state["page"]]()
