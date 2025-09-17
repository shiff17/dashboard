import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Vulnerability Hunters", layout="wide")
sns.set_palette("husl")

# ---------------------- THEME TOGGLE ----------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

theme = st.session_state["theme"]

if theme == "light":
    primary_color = "#3498db"  # Light blue
    bg_color = "#f0f8ff"
    text_color = "#000000"
else:
    primary_color = "#1e3a8a"  # Dark blue
    bg_color = "#0d1117"
    text_color = "#ffffff"

# ---------------------- CSS STYLING ----------------------
st.markdown(
    f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    /* Centered Title */
    .centered-title {{
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
        color: {primary_color};
    }}
    /* Top Navigation Bar */
    .topnav {{
        overflow: hidden;
        background-color: transparent;
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .topnav button {{
        background-color: transparent;
        border: none;
        color: {text_color};
        padding: 14px 20px;
        font-size: 18px;
        cursor: pointer;
        transition: 0.3s;
    }}
    .topnav button:hover {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
    }}
    /* Toggle Button */
    .toggle-btn {{
        position: absolute;
        top: 20px;
        right: 20px;
        background-color: {primary_color};
        color: white;
        border: none;
        padding: 10px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
    }}
    .toggle-btn:hover {{
        background-color: #2563eb;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- TITLE ----------------------
st.markdown('<div class="centered-title">🛡 Vulnerability Hunters</div>', unsafe_allow_html=True)

# ---------------------- TOGGLE THEME ----------------------
if st.button("🌗 Toggle Theme", key="theme_toggle", help="Switch between Light and Dark Mode"):
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"
    st.rerun()

# ---------------------- HELPERS ----------------------
def safe_numeric_df(df):
    """Return numeric columns; if none, try coercing."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        num = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    return num

def make_severity_from_cvss(df):
    """Create 'severity' column if 'cvss' exists."""
    if "cvss" in df.columns:
        bins = [0, 3.9, 6.9, 8.9, 10]
        labels = ["Low", "Medium", "High", "Critical"]
        df["severity"] = pd.cut(df["cvss"], bins=bins, labels=labels, include_lowest=True)

def baseline_regression(df, target_col):
    """Simple baseline linear regression using numeric features."""
    if target_col is None or target_col not in df.columns:
        return {"r2": None, "rmse": None, "note": "No valid target selected."}

    numeric = safe_numeric_df(df).copy()
    if target_col not in numeric.columns:
        numeric[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    y = numeric[target_col].dropna()
    X = numeric.drop(columns=[target_col], errors="ignore").loc[y.index]

    if X.shape[1] == 0 or X.shape[0] < 10:
        return {"r2": None, "rmse": None, "note": "Not enough data/features for baseline."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(Xtr, ytr)
    preds = model.predict(Xte)

    mse = mean_squared_error(yte, preds)
    rmse = round(float(np.sqrt(mse)), 4)

    return dict(r2=round(float(r2_score(yte, preds)), 4), rmse=rmse)

def rl_cleaning_search(df, target_col, iterations=10):
    rng = np.random.default_rng(42)
    best = {"r2": -1e9, "rmse": 1e9, "df": None}
    logs = []

    numeric = safe_numeric_df(df)
    if target_col not in df.columns:
        return {"best": best, "logs": logs}

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = numeric.drop(columns=[c for c in numeric.columns if c == target_col], errors="ignore")
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    if X.shape[1] == 0 or X.shape[0] < 40:
        return {"best": best, "logs": logs}

    def evaluate(strategy):
        Xt = X.copy()
        if strategy["impute"] == "mean":
            Xt = Xt.fillna(Xt.mean())
        elif strategy["impute"] == "median":
            Xt = Xt.fillna(Xt.median())

        if strategy["outliers"] == "z3":
            z = (Xt - Xt.mean()) / Xt.std(ddof=0)
            mask_in = (np.abs(z) <= 3).all(axis=1)
            Xt = Xt.loc[mask_in]
        yt = y.loc[Xt.index]

        if strategy["scale"] == "standard":
            Xt = pd.DataFrame(StandardScaler().fit_transform(Xt), index=Xt.index, columns=Xt.columns)
        elif strategy["scale"] == "minmax":
            Xt = pd.DataFrame(MinMaxScaler().fit_transform(Xt), index=Xt.index, columns=Xt.columns)

        if strategy["kbest"] > 0 and strategy["kbest"] < Xt.shape[1]:
            selector = SelectKBest(score_func=f_regression, k=strategy["kbest"]).fit(Xt, yt)
            cols = Xt.columns[selector.get_support(indices=True)]
            Xt = Xt.loc[:, cols]

        if Xt.shape[0] < 10 or Xt.shape[1] == 0:
            return -1e9, 1e9, Xt, yt

        Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte, pred)
        rmse = mean_squared_error(yte, pred, squared=False)
        return r2, rmse, Xt, yt

    for i in range(iterations):
        strategy = {
            "impute": rng.choice(["mean", "median"]),
            "scale": rng.choice(["none", "standard", "minmax"]),
            "outliers": rng.choice(["none", "z3"]),
            "kbest": int(rng.choice([0, 5, 10]))
        }
        r2, rmse, Xt, yt = evaluate(strategy)
        logs.append({"iter": i+1, "r2": round(float(r2), 4) if np.isfinite(r2) else None,
                     "rmse": round(float(rmse), 4) if np.isfinite(rmse) else None,
                     "strategy": strategy})
        if r2 > best["r2"]:
            df_combined = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True).rename(target_col)], axis=1)
            best = {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4), "df": df_combined}

    return {"best": best, "logs": logs}

# ---------------------- NAVIGATION ----------------------
pages = ["Home", "Analysis & Insights", "Custom Visualizations", "Timeline", "Recommendations"]
cols = st.columns(len(pages))

selected_page = None
for i, page in enumerate(pages):
    if cols[i].button(page, key=f"nav_{i}"):
        selected_page = page

if selected_page is None:
    selected_page = "Home"

# ---------------------- PAGE CONTENT ----------------------
if selected_page == "Home":
    st.subheader("📂 Upload your dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        make_severity_from_cvss(df)
        st.session_state["df"] = df
        st.write("### Data Preview", df.head())

elif selected_page == "Analysis & Insights":
    if "df" not in st.session_state:
        st.warning("Please upload data first.")
    else:
        df = st.session_state["df"]
        st.subheader("📊 Analysis & Insights")

        st.write("*Summary Stats*")
        st.write(df.describe(include="all"))

        st.write("*Correlation Heatmap*")
        num = safe_numeric_df(df)
        if num.shape[1] > 1:
            plt.figure(figsize=(6, 4))
            sns.heatmap(num.corr(), annot=True, cmap="Blues")
            st.pyplot(plt.gcf())

        # ---------------------- Before & After Cleaning ----------------------
        if "clean" in st.session_state and st.session_state["clean"] is not None:
            st.subheader("🧹 Before vs After Cleaning")
            cols = safe_numeric_df(st.session_state["clean"]).columns.tolist()
            if cols:
                choice = st.selectbox("Choose a numeric column to compare", cols)
                plot_type = st.radio("Select plot type", ["Histogram", "Boxplot", "KDE"], horizontal=True)

                if choice:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    before = st.session_state["df"][choice].dropna()
                    after = st.session_state["clean"][choice].dropna()

                    if plot_type == "Histogram":
                        sns.histplot(before, kde=True, color="red", ax=axes[0])
                        sns.histplot(after, kde=True, color="green", ax=axes[1])
                    elif plot_type == "Boxplot":
                        sns.boxplot(y=before, color="red", ax=axes[0])
                        sns.boxplot(y=after, color="green", ax=axes[1])
                    elif plot_type == "KDE":
                        sns.kdeplot(before, fill=True, color="red", ax=axes[0])
                        sns.kdeplot(after, fill=True, color="green", ax=axes[1])

                    axes[0].set_title("Before Cleaning")
                    axes[1].set_title("After Cleaning")
                    st.pyplot(fig)

                    # ---------------- EXPORT OPTIONS ----------------
                    st.write("### 📤 Export Options")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    st.download_button(
                        label="⬇ Download Comparison Plot (PNG)",
                        data=buf.getvalue(),
                        file_name=f"{choice}_before_after.png",
                        mime="image/png"
                    )
                    csv = st.session_state["clean"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇ Download Cleaned Data (CSV)",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )

elif selected_page == "Custom Visualizations":
    st.subheader("🎨 Custom Visualizations")
    st.info("Interactive charts and visual exploration.")

elif selected_page == "Timeline":
    st.subheader("⏳ Vulnerability Timeline")
    st.info("Trends and temporal progression.")

elif selected_page == "Recommendations":
    st.subheader("💡 Recommendations")
    st.info("Self-healing patch orchestration and mitigation strategies.")
