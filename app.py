# vulnerability_dashboard_fixed.py
import io
import zipfile
import time
import numpy as np
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

def baseline regression(df,target_col):
    """Simple baseline linear regression using numeric features."""
    if target_col is None or target_col not in df.columns:
        return {"r2":None,"rmse":None,"note":"No valid targeted selected."}
        
#work only with numeric features + ensure target numeric
numeric=safe_numeric_df(df).copy()
if target_col not in numeric.columns:
     numeric[target_col]=pd.to_numeric(df[target_col],errors="coerce")

     y=numeric[target_col].dropna()
     x=numeric.drop(columns=[target_col],errors="ignore").loc[y.index]

     if X.shape[1]==0 or X.shape[0]<10:
         return{"r2":None,"rmse":None,"note":"Not enough data/features for baseline."}

         Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
         model=LinearRegression().fit(Xtr,ytr)
         preds=model.predict(Xte)

         mse=mean_squared_error(yte,preds)
         rmse=round(float(np.sqrt(mse)),4)
         return{
             "r2":round(float(r2_score(yte,preds)),4),
             "rmse":rmse
         }

def rl_cleaning_search(df, target_col, iterations=10):
    """
    A randomised search over a few simple preprocessing strategies
    and returns the best transformed dataframe (with target) and logs.
    """
    rng = np.random.default_rng(42)
    best = {"r2": -1e9, "rmse": 1e9, "df": None}
    logs = []

    # use numeric columns
    numeric = safe_numeric_df(df)
    if target_col not in df.columns:
        return {"best": best, "logs": logs}

    # create numeric target and aligned X
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = numeric.drop(columns=[c for c in numeric.columns if c == target_col], errors="ignore")
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    if X.shape[1] == 0 or X.shape[0] < 40:
        return {"best": best, "logs": logs}

    def evaluate(strategy):
        Xt = X.copy()
        # impute
        if strategy["impute"] == "mean":
            Xt = Xt.fillna(Xt.mean())
        elif strategy["impute"] == "median":
            Xt = Xt.fillna(Xt.median())

        # outlier removal by z-score
        if strategy["outliers"] == "z3":
            z = (Xt - Xt.mean()) / Xt.std(ddof=0)
            mask_in = (np.abs(z) <= 3).all(axis=1)
            Xt = Xt.loc[mask_in]
        yt = y.loc[Xt.index]

        # scaling
        if strategy["scale"] == "standard":
            Xt = pd.DataFrame(StandardScaler().fit_transform(Xt), index=Xt.index, columns=Xt.columns)
        elif strategy["scale"] == "minmax":
            Xt = pd.DataFrame(MinMaxScaler().fit_transform(Xt), index=Xt.index, columns=Xt.columns)

        # feature selection (SelectKBest keeps column names via get_support)
        if strategy["kbest"] > 0 and strategy["kbest"] < Xt.shape[1]:
            selector = SelectKBest(score_func=f_regression, k=strategy["kbest"]).fit(Xt, yt)
            cols = Xt.columns[selector.get_support(indices=True)]
            Xt = Xt.loc[:, cols]

        if Xt.shape[0] < 10 or Xt.shape[1] == 0:
            return -1e9, 1e9, Xt, yt  # poor score

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
        logs.append({"iter": i+1, "r2": round(float(r2), 4) if np.isfinite(r2) else None, "rmse": round(float(rmse), 4) if np.isfinite(rmse) else None, "strategy": strategy})
        if r2 > best["r2"]:
            # keep a concatenated df (features + target), and reset target name
            df_combined = pd.concat([Xt.reset_index(drop=True), yt.reset_index(drop=True).rename(target_col)], axis=1)
            best = {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4), "df": df_combined}

    return {"best": best, "logs": logs}

# ---------------------- Upload & Process ----------------------
uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    # Strip whitespace from column names (fix for "white spaces" in headers)
    df.columns = df.columns.str.strip()
    make_severity_from_cvss(df)
    st.write("### Raw Data Preview", df.head())

    # Pick target column
    num_cols = safe_numeric_df(df).columns.tolist()
    target_col = st.selectbox("🎯 Choose a numeric target column for regression", ["<none>"] + num_cols)
    target_col = None if target_col == "<none>" else target_col

    if st.button("🚀 Process Data"):
        with st.spinner("Processing..."):
            # prefer not to drop all rows: create a cleaned copy for processing
            prep = df.copy()
            # keep only rows that have at least one non-null numeric value
            prep = prep.dropna(how="all", subset=safe_numeric_df(prep).columns.tolist())
            num = safe_numeric_df(prep)
            if num.shape[1] >= 2 and num.shape[0] >= 10:
                km = KMeans(n_clusters=3, random_state=42).fit(StandardScaler().fit_transform(num))
                prep["cluster_before"] = km.labels_
            else:
                prep["cluster_before"] = -1
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
                # if we have a cleaned DF, try clustering it and add a cluster_after column
                if st.session_state["clean"] is not None:
                    clean_num = safe_numeric_df(st.session_state["clean"])
                    if clean_num.shape[0] >= 3 and clean_num.shape[1] >= 2:
                        km2 = KMeans(n_clusters=3, random_state=42).fit(StandardScaler().fit_transform(clean_num))
                        st.session_state["clean"]["cluster_after"] = km2.labels_
            st.success(f"RL cleaning done. Best R²: {rl['best']['r2']}")

# ---------------------- Results ----------------------
if "clean" in st.session_state and st.session_state["clean"] is not None:
    before = st.session_state["prep"]
    after = st.session_state["clean"]

    st.header("📈 Results & Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Clusters Before")
        if "cluster_before" in before.columns:
            num = safe_numeric_df(before)
            if num.shape[1] >= 2:
                plt.figure()
                sns.scatterplot(x=num.iloc[:, 0], y=num.iloc[:, 1], hue=before["cluster_before"], palette="Set2")
                st.pyplot(plt.gcf())
            else:
                st.write("Not enough numeric columns to plot clusters.")
    with c2:
        st.write("### Clusters After")
        if "cluster_after" in after.columns:
            num = safe_numeric_df(after)
            if num.shape[1] >= 2:
                plt.figure()
                sns.scatterplot(x=num.iloc[:, 0], y=num.iloc[:, 1], hue=after["cluster_after"], palette="Set1")
                st.pyplot(plt.gcf())
            else:
                st.write("Not enough numeric columns to plot clusters.")

    st.write("### RL Logs")
    st.dataframe(pd.DataFrame(st.session_state["rl"]["logs"]))
    
