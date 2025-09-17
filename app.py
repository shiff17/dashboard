import io, zipfile, time
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Vulnerability Dashboard", layout="wide")
sns.set_palette("husl")

st.title("🛡 Vulnerability Data Dashboard")

# ---------------------- Helpers ----------------------
def safe_numeric_df(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        num = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    return num

def make_severity_from_cvss(df):
    if "cvss" in df.columns:
        bins = [0, 3.9, 6.9, 8.9, 10]
        labels = ["Low", "Medium", "High", "Critical"]
        df["severity"] = pd.cut(df["cvss"], bins=bins, labels=labels, include_lowest=True)

def baseline_regression(df, target_col):
    if target_col not in df.columns:
        return None
    numeric = safe_numeric_df(df)
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = numeric.drop(columns=[c for c in numeric.columns if c == target_col], errors="ignore")
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    if X.shape[1] == 0 or X.shape[0] < 12:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)
    preds = model.predict(Xte)
    return dict(r2=round(float(r2_score(yte, preds)), 4),
                mse = mean_squared_error(yte, preds)
                rmse = round(float(np.sqrt(mse)), 4)

def rl_cleaning_search(df, target_col, iterations=10):
    rng = np.random.default_rng(42)
    best = {"r2": -1e9, "rmse": 1e9, "df": None}
    logs = []

    numeric = safe_numeric_df(df)
    if target_col not in df.columns: return {"best": best, "logs": logs}
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = numeric.drop(columns=[c for c in numeric.columns if c == target_col], errors="ignore")
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    if X.shape[1] == 0 or X.shape[0] < 40: return {"best": best, "logs": logs}

    def evaluate(strategy):
        Xt = X.copy()
        if strategy["impute"] == "mean": Xt = Xt.fillna(Xt.mean())
        if strategy["impute"] == "median": Xt = Xt.fillna(Xt.median())
        if strategy["outliers"] == "z3":
            z = (Xt - Xt.mean())/Xt.std(ddof=0)
            Xt = Xt[(np.abs(z) <= 3).all(axis=1)]
        yt = y.loc[Xt.index]
        if strategy["scale"] == "standard":
            Xt = pd.DataFrame(StandardScaler().fit_transform(Xt), index=Xt.index)
        elif strategy["scale"] == "minmax":
            Xt = pd.DataFrame(MinMaxScaler().fit_transform(Xt), index=Xt.index)
        if strategy["kbest"] > 0 and strategy["kbest"] < Xt.shape[1]:
            Xt = pd.DataFrame(SelectKBest(score_func=f_regression, k=strategy["kbest"]).fit_transform(Xt, yt), index=Xt.index)
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
            "scale": rng.choice(["none","standard","minmax"]),
            "outliers": rng.choice(["none","z3"]),
            "kbest": int(rng.choice([0,5,10]))
        }
        r2, rmse, Xt, yt = evaluate(strategy)
        logs.append({"iter": i+1, "r2": round(r2,4), "rmse": round(rmse,4), "strategy": strategy})
        if r2 > best["r2"]:
            best = {"r2": round(r2,4), "rmse": round(rmse,4), "df": pd.concat([Xt, yt.rename(target_col)], axis=1)}
    return {"best": best, "logs": logs}

# ---------------------- Upload & Process ----------------------
uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    make_severity_from_cvss(df)
    st.write("### Raw Data Preview", df.head())

    # Pick target column
    num_cols = safe_numeric_df(df).columns.tolist()
    target_col = st.selectbox("🎯 Choose a numeric target column for regression", ["<none>"]+num_cols)
    target_col = None if target_col=="<none>" else target_col

    if st.button("🚀 Process Data"):
        with st.spinner("Processing..."):
            prep = df.dropna().copy()
            num = safe_numeric_df(prep)
            if num.shape[1]>=2 and num.shape[0]>=10:
                km = KMeans(n_clusters=3, random_state=42).fit(StandardScaler().fit_transform(num))
                prep["cluster_before"] = km.labels_
            baseline = baseline_regression(prep, target_col)
            st.session_state["prep"] = prep
            st.session_state["baseline"] = baseline
        st.success("Processing complete ✅")
        st.write("Baseline Regression:", baseline)

    if "prep" in st.session_state:
        if st.button("🧹 Clean Data (RL-style)"):
            with st.spinner("RL Cleaning..."):
                rl = rl_cleaning_search(st.session_state["prep"], target_col)
                st.session_state["rl"] = rl
                st.session_state["clean"] = rl["best"]["df"]
            st.success(f"RL cleaning done. Best R²: {rl['best']['r2']}")

# ---------------------- Results ----------------------
if "clean" in st.session_state:
    before = st.session_state["prep"]
    after = st.session_state["clean"]

    st.header("📈 Results & Visualizations")
    c1,c2 = st.columns(2)
    with c1:
        st.write("### Clusters Before")
        if "cluster_before" in before.columns:
            num = safe_numeric_df(before)
            plt.figure()
            sns.scatterplot(x=num.iloc[:,0], y=num.iloc[:,1], hue=before["cluster_before"], palette="Set2")
            st.pyplot(plt.gcf())
    with c2:
        st.write("### Clusters After")
        if "cluster_after" in after.columns:
            num = safe_numeric_df(after)
            plt.figure()
            sns.scatterplot(x=num.iloc[:,0], y=num.iloc[:,1], hue=after["cluster_after"], palette="Set1")
            st.pyplot(plt.gcf())

    st.write("### RL Logs")
    st.dataframe(pd.DataFrame(st.session_state["rl"]["logs"]))
