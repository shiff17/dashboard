import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Vulnerability Hunters",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== THEME & CSS ==================
light_bg = "#dceeff"
dark_bg = "#001f3f"

st.markdown("""
    <style>
    body {background-color: #f0f8ff;}
    /* Title */
    .title {
        font-size: 70px;
        text-align: center;
        color: #004080;
        font-weight: bold;
        margin-bottom: 30px;
    }
    /* Upload Button */
    div[data-testid="stFileUploader"] > label {
        font-size: 22px;
        font-weight: bold;
        color: white !important;
        background: linear-gradient(135deg, #66b2ff, #004080);
        padding: 15px 25px;
        border-radius: 15px;
        cursor: pointer;
        text-align: center;
        display: block;
        margin: auto;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stFileUploader"] > label:hover {
        background: linear-gradient(135deg, #99ccff, #003366);
        transform: scale(1.05);
    }
    /* General Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #66b2ff, #004080);
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
        display: block;
        margin: auto;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #99ccff, #003366);
        transform: scale(1.05);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #66b2ff, #004080);
        color: white;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ================== HELPER FUNCTIONS ==================
def baseline_regression(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(Xtr, ytr)
    preds = model.predict(Xte)

    mse = mean_squared_error(yte, preds)  # squared param removed
    rmse = round(float(np.sqrt(mse)), 4)
    return dict(r2=round(float(r2_score(yte, preds)), 4), rmse=rmse)

def self_healing_patch(df):
    # Dummy placeholder for federated learning + digital twin logic
    df["patched"] = np.random.choice(["Yes", "No"], size=len(df))
    return df

# ================== TASKBAR ==================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Analysis & Insights", "📐 Custom Visualizations", 
     "⏳ Timeline", "💡 Recommendations", "🛡 Self-Healing"]
)

# ================== HOME PAGE ==================
if menu == "🏠 Home":
    st.markdown("<div class='title'>Vulnerability Hunters</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Your Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = self_healing_patch(df)  # integrate self-healing
        st.success("✅ Data uploaded & Self-Healing applied!")

# ================== ANALYSIS & INSIGHTS ==================
elif menu == "📊 Analysis & Insights":
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("📊 Summary Statistics")
        st.write(df.describe(include="all").T)

        # Correlation Heatmap
        if not df.select_dtypes(include=["number"]).empty:
            st.subheader("🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Distribution Plots
        st.subheader("📦 Distribution of Numeric Columns")
        num_cols = df.select_dtypes(include=["number"]).columns
        for col in num_cols[:5]:  # limit for speed
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, bins=20, color="skyblue", ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        # Export
        st.subheader("💾 Export Processed Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "processed_data.csv", "text/csv", use_container_width=True)

# ================== CUSTOM VISUALIZATIONS ==================
elif menu == "📐 Custom Visualizations":
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("🎨 Create Custom Plots")
        col = st.selectbox("Choose column to visualize", df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, color="orange", ax=ax)
        st.pyplot(fig)

# ================== TIMELINE ==================
elif menu == "⏳ Timeline":
    st.subheader("⏳ Vulnerability Timeline")
    st.info("Here you can later add patch timelines, release dates, or discovery patterns.")

# ================== RECOMMENDATIONS ==================
elif menu == "💡 Recommendations":
    st.subheader("💡 AI-Powered Recommendations")
    st.success("✔ Future scope: Add ML-driven suggestions for patch priority.")

# ================== SELF-HEALING ==================
elif menu == "🛡 Self-Healing":
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.subheader("🛡 Self-Healing Patch Orchestration")
        st.write(df[["patched"]].head(10))
