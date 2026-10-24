"""Food Security & Malnutrition Risk Assessment - DEMO frontend.

Everything here uses demo / sample data. The risk "prediction" is a simple
rule-based placeholder that will later be replaced by the trained ML model.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import linprog

st.set_page_config(page_title="Food Security Risk Assessment", page_icon="🍚", layout="wide")

LEVELS = ["Low", "Moderate", "High"]
COLORS = {"Low": "#2e9e5b", "Moderate": "#f0a202", "High": "#d64545"}
PAGES = ["🏠 Home", "📝 Assessment", "📊 Results", "🥗 Meal Planner", "📈 History", "🧭 Dashboard", "📄 Report"]
DISCLAIMER = "Screening guidance only. This is not a medical diagnosis."

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


# ---------------------------------------------------------------- demo logic
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
        "probs": dict(zip(LEVELS, probs)),
        "contributions": c,
        "inputs": d,
    }


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
    s["last"] = res
    new_row = pd.DataFrame([{"Date": res["date"], "Risk score": res["score"], "Risk level": res["level"]}])
    s["history"] = pd.concat([s["history"], new_row], ignore_index=True)
    s["page"] = "📊 Results"


# ------------------------------------------------------------ session state
if "history" not in st.session_state:
    today = date.today()
    scores = [78, 70, 62, 51, 44]
    st.session_state["history"] = pd.DataFrame(
        {
            "Date": [today - timedelta(days=d) for d in (150, 120, 90, 60, 30)],
            "Risk score": scores,
            "Risk level": [level_from_score(x) for x in scores],
        }
    )
if "last" not in st.session_state:
    st.session_state["last"] = None
if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]


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


# --------------------------------------------------------------------- pages
def page_home():
    st.title("🍚 Food Security & Malnutrition Risk Assessment")
    st.caption("SDG 3 – Good Health and Well-being  |  DEMO version with sample data")
    st.write(
        "Answer a few questions about your household and get a **risk level** (Low, Moderate or High), "
        "the **factors behind it**, practical **recommendations**, a **budget meal plan** and a downloadable report."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Assessments saved", len(st.session_state["history"]))
    c2.metric("Latest risk level", st.session_state["last"]["level"] if st.session_state["last"] else "—")
    c3.metric("Demo households in dashboard", 200)

    st.subheader("How it works")
    st.markdown(
        "1. **Assessment** – fill in the household form.\n"
        "2. **Results** – see your risk level, probabilities, key factors and recommendations.\n"
        "3. **Meal Planner** – get the lowest-cost meal plan that meets nutrition targets.\n"
        "4. **History** – track your risk score over time.\n"
        "5. **Dashboard** – see community-level charts.\n"
        "6. **Report** – download a summary."
    )
    st.info("Use the menu on the left to start. Click **📝 Assessment** first.")
    st.caption(DISCLAIMER)


def page_assessment():
    st.title("📝 Food Security Assessment")
    st.write("Fill in the form and press **Assess risk**. (Sample values are pre-filled.)")
    with st.form("assess"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Household size", 1, 20, 4, key="in_size")
            st.number_input("Monthly household income (BDT)", 0, 1_000_000, 20000, step=500, key="in_income")
            st.number_input("Monthly food expenditure (BDT)", 0, 1_000_000, 9000, step=500, key="in_food")
            st.selectbox("Meals per day", [1, 2, 3, 4], index=2, key="in_meals")
        with c2:
            st.slider("Dietary diversity score (0–12 food groups eaten)", 0, 12, 6, key="in_div")
            st.selectbox("Employment", ["Regular", "Daily wage", "Unemployed"], key="in_emp")
            st.radio("Child under 5 in the household?", ["No", "Yes"], horizontal=True, key="in_child")
            st.slider("Difficulty accessing food (1 = easy, 5 = very hard)", 1, 5, 2, key="in_access")
        st.form_submit_button("Assess risk", on_click=run_assessment, type="primary")
    st.caption(DISCLAIMER)


def page_results():
    st.title("📊 Assessment Results")
    r = st.session_state["last"]
    if r is None:
        st.info("No assessment yet. Go to **📝 Assessment** and submit the form.")
        return

    left, right = st.columns([1, 1.4])
    with left:
        color = COLORS[r["level"]]
        st.markdown(
            f"<div style='padding:22px;border-radius:14px;background:{color};color:white;"
            f"font-size:30px;font-weight:700;text-align:center'>{r['level']} risk</div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Risk score (0–100)", r["score"])
        m2.metric("Confidence", f"{max(r['probs'].values()) * 100:.0f}%")
    with right:
        pdf = pd.DataFrame({"Risk level": LEVELS, "Probability (%)": [r["probs"][k] * 100 for k in LEVELS]})
        fig = px.bar(pdf, x="Risk level", y="Probability (%)", color="Risk level", color_discrete_map=COLORS)
        fig.update_layout(showlegend=False, yaxis_range=[0, 100], title="Prediction probability", height=300)
        st.plotly_chart(fig)

    st.subheader("Why this result?")
    st.caption("Demo version of the explainability chart. In the final app this comes from SHAP.")
    cdf = pd.DataFrame({"Factor": list(r["contributions"].keys()), "Contribution": list(r["contributions"].values())})
    cdf = cdf.sort_values("Contribution")
    fig2 = px.bar(cdf, x="Contribution", y="Factor", orientation="h", color="Contribution",
                  color_continuous_scale=["#2e9e5b", "#f0a202", "#d64545"])
    fig2.update_layout(coloraxis_showscale=False, height=350)
    st.plotly_chart(fig2)

    st.subheader("Recommendations")
    top = [k for k, v in sorted(r["contributions"].items(), key=lambda kv: -kv[1]) if v > 0][:3]
    if top:
        for k in top:
            st.markdown(f"- **{k}:** {ADVICE[k]}")
    else:
        st.success("No major risk factors found. Keep up your current food habits.")
    st.caption(DISCLAIMER)


def page_meal_planner():
    st.title("🥗 Budget Meal Planner")
    st.write("Finds the **cheapest combination of foods** that meets your calorie and protein targets.")
    default_people = st.session_state["last"]["inputs"]["size"] if st.session_state["last"] else 4

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
    m1.metric("Minimum daily cost (BDT)", f"{total:.0f}")
    m2.metric("Weekly cost (BDT)", f"{total * 7:.0f}")
    m3.metric("Your budget (BDT/day)", f"{budget}")
    if total <= budget:
        st.success(f"Your budget covers this plan with about {budget - total:.0f} BDT per day to spare.")
    else:
        st.warning(f"Your budget is about {total - budget:.0f} BDT per day short of the cheapest plan that meets the targets.")

    st.subheader("Suggested daily food quantities (whole household)")
    st.dataframe(plan, hide_index=True)
    fig = px.pie(plan, names="Food", values="Cost per day (BDT)", title="Where the money goes")
    st.plotly_chart(fig)
    st.caption("Foods, prices and nutrition values are approximate demo data. " + DISCLAIMER)


def page_history():
    st.title("📈 Risk Trend Tracking")
    hist = st.session_state["history"].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    fig = px.line(hist, x="Date", y="Risk score", markers=True, title="Risk score over time (lower is better)")
    fig.add_hline(y=29, line_dash="dot", annotation_text="Low / Moderate")
    fig.add_hline(y=57, line_dash="dot", annotation_text="Moderate / High")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig)

    if len(hist) >= 2:
        latest, prev = hist["Risk score"].iloc[-1], hist["Risk score"].iloc[-2]
        st.metric("Latest score vs previous", int(latest), int(latest - prev), delta_color="inverse")
    st.subheader("All assessments")
    show = hist.copy()
    show["Date"] = show["Date"].dt.date
    st.dataframe(show.sort_values("Date", ascending=False), hide_index=True)
    st.caption("Demo history is pre-loaded. New assessments are added while the app session is open.")


def page_dashboard():
    st.title("🧭 Community Dashboard")
    st.caption("Demo data for 200 sample households.")
    df = community_data()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Households", len(df))
    k2.metric("High risk", f"{(df['Risk level'] == 'High').mean() * 100:.0f}%")
    k3.metric("Average risk score", f"{df['Risk score'].mean():.0f}")
    k4.metric("Avg dietary diversity", f"{df['Dietary diversity'].mean():.1f}")

    c1, c2 = st.columns(2)
    with c1:
        counts = df["Risk level"].value_counts().reindex(LEVELS).reset_index()
        counts.columns = ["Risk level", "Households"]
        fig = px.pie(counts, names="Risk level", values="Households", hole=0.45,
                     color="Risk level", color_discrete_map=COLORS, title="Risk level distribution")
        st.plotly_chart(fig)
    with c2:
        emp = df.groupby("Employment")["Risk level"].apply(lambda s: (s == "High").mean() * 100).reset_index()
        emp.columns = ["Employment", "High-risk households (%)"]
        fig = px.bar(emp, x="Employment", y="High-risk households (%)", title="High risk by employment type")
        st.plotly_chart(fig)

    fig = px.scatter(df, x="Income per person", y="Risk score", color="Risk level",
                     color_discrete_map=COLORS, title="Income per person vs risk score")
    st.plotly_chart(fig)


def page_report():
    st.title("📄 Report")
    r = st.session_state["last"]
    if r is None:
        st.info("Complete an assessment first, then come back to download your report.")
        return
    i = r["inputs"]
    top = [k for k, v in sorted(r["contributions"].items(), key=lambda kv: -kv[1]) if v > 0][:3]
    lines = [
        "FOOD SECURITY RISK ASSESSMENT REPORT (DEMO)",
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

    st.text(text)
    st.download_button("⬇️ Download report (.txt)", text, file_name="food_security_report.txt")
    st.download_button("⬇️ Download history (.csv)", st.session_state["history"].to_csv(index=False),
                       file_name="assessment_history.csv")


# ------------------------------------------------------------------- layout
st.sidebar.title("Menu")
st.sidebar.radio("Go to", PAGES, key="page", label_visibility="collapsed")
st.sidebar.warning("Demo mode: sample data and a rule-based placeholder instead of the ML model.")
st.sidebar.caption(DISCLAIMER)

ROUTES = {
    "🏠 Home": page_home,
    "📝 Assessment": page_assessment,
    "📊 Results": page_results,
    "🥗 Meal Planner": page_meal_planner,
    "📈 History": page_history,
    "🧭 Dashboard": page_dashboard,
    "📄 Report": page_report,
}
ROUTES[st.session_state["page"]]()
