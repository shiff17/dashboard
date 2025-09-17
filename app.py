import streamlit as st
import pandas as pd
import numpy as np

from visualizations import run_models, feature_selection, clustering

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Vulnerability Hunters", layout="wide")

# Load CSS
with open("ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🛡 Vulnerability Hunters Dashboard")

# ---------------------- File Upload ----------------------
uploaded = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    # ---------------------- Data Cleaning ----------------------
    st.subheader("🧹 Data Cleaning")
    df.columns = df.columns.str.strip()  # strip spaces
    df_clean = pd.get_dummies(df, drop_first=True)  # encode categories
    num_df = df_clean.select_dtypes(include=[np.number]).dropna()

    st.write("✅ Cleaned numeric data ready for ML")
    st.write(num_df.head())

    # ---------------------- Run ML + Visualizations ----------------------
    if num_df.shape[1] >= 2:
        run_models(num_df)
        feature_selection(num_df)
        clustering(num_df)
    else:
        st.warning("⚠ Not enough numeric columns for ML. Please upload a dataset with more features.")

else:
    st.info("📂 Please upload a CSV file to start.")
