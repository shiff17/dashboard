import streamlit as st
import pandas as pd

from ui import load_theme, show_title, navigation_bar
from visualizations import (
    plot_histogram, plot_boxplot, plot_kde, plot_correlation_heatmap,
    plot_pairplot, plot_scatter, plot_violin, plot_count, plot_line, plot_heatmap_nulls
)

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Vulnerability Hunters", layout="wide")

# ---------------------- Theme + UI ----------------------
theme = load_theme()
show_title()
selected_page = navigation_bar()

# ---------------------- Pages ----------------------
if selected_page == "Home":
    st.subheader("📂 Upload your dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()

        # Convert all to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")

        # Store in session
        st.session_state["df"] = df
        st.write("### Data Preview", df.head())
        st.write("### Column Data Types", df.dtypes)
        st.write("### Missing Values per Column", df.isna().sum())

elif selected_page == "Analysis & Insights":
    if "df" not in st.session_state:
        st.warning("Please upload data first.")
    else:
        df = st.session_state["df"]
        st.subheader("📊 Analysis & Insights")

        st.write("*Summary Stats*")
        st.write(df.describe(include="all"))

        st.write("*Correlation Heatmap*")
        plot_correlation_heatmap(df)

elif selected_page == "Custom Visualizations":
    st.subheader("🎨 Custom Visualizations")
    if "df" not in st.session_state:
        st.warning("Please upload data first.")
    else:
        df = st.session_state["df"]
        num_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        viz = st.selectbox("Select Graph Type", [
            "Histogram", "Boxplot", "KDE", "Correlation Heatmap",
            "Pairplot", "Scatterplot", "Violin Plot",
            "Count Plot", "Line Plot", "Missing Values Heatmap"
        ])

        if viz in ["Histogram", "Boxplot", "KDE", "Violin Plot", "Line Plot"] and num_cols:
            col = st.selectbox("Select numeric column", num_cols)
            if viz == "Histogram": plot_histogram(df, col)
            elif viz == "Boxplot": plot_boxplot(df, col)
            elif viz == "KDE": plot_kde(df, col)
            elif viz == "Violin Plot": plot_violin(df, col)
            elif viz == "Line Plot": plot_line(df, col)

        elif viz == "Scatterplot" and len(num_cols) >= 2:
            x = st.selectbox("X-axis", num_cols, index=0)
            y = st.selectbox("Y-axis", num_cols, index=1)
            plot_scatter(df, x, y)

        elif viz == "Count Plot":
            col = st.selectbox("Select categorical column", all_cols)
            plot_count(df, col)

        elif viz == "Correlation Heatmap":
            plot_correlation_heatmap(df)

        elif viz == "Pairplot":
            plot_pairplot(df)

        elif viz == "Missing Values Heatmap":
            plot_heatmap_nulls(df)

elif selected_page == "Timeline":
    st.subheader("⏳ Vulnerability Timeline")
    st.info("Trends and temporal progression.")

elif selected_page == "Recommendations":
    st.subheader("💡 Recommendations")
    st.info("Self-healing patch orchestration and mitigation strategies.")
