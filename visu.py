import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

def export_plot(fig, filename, df=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.download_button("⬇ Download Plot (PNG)", buf.getvalue(), file_name=f"{filename}.png", mime="image/png")
    if df is not None:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Data (CSV)", csv, file_name=f"{filename}.csv", mime="text/csv")

def plot_histogram(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    st.pyplot(fig)
    export_plot(fig, f"{column}_histogram", df[[column]].dropna())

def plot_boxplot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[column].dropna(), ax=ax)
    ax.set_title(f"Boxplot of {column}")
    st.pyplot(fig)
    export_plot(fig, f"{column}_boxplot", df[[column]].dropna())

def plot_kde(df, column):
    fig, ax = plt.subplots()
    sns.kdeplot(df[column].dropna(), fill=True, ax=ax)
    ax.set_title(f"KDE of {column}")
    st.pyplot(fig)
    export_plot(fig, f"{column}_kde", df[[column]].dropna())

def plot_correlation_heatmap(df):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        st.warning("Not enough numeric columns for heatmap.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    export_plot(fig, "correlation_heatmap", num.corr().reset_index())

def plot_pairplot(df):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        st.warning("Not enough numeric columns for pairplot.")
        return
    fig = sns.pairplot(num)
    st.pyplot(fig)
    csv = num.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Data (CSV)", csv, file_name="pairplot_data.csv", mime="text/csv")

def plot_scatter(df, x_col, y_col):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    ax.set_title(f"Scatterplot of {x_col} vs {y_col}")
    st.pyplot(fig)
    export_plot(fig, f"{x_col}vs{y_col}_scatter", df[[x_col, y_col]].dropna())

def plot_violin(df, column):
    fig, ax = plt.subplots()
    sns.violinplot(y=df[column].dropna(), ax=ax)
    ax.set_title(f"Violin Plot of {column}")
    st.pyplot(fig)
    export_plot(fig, f"{column}_violin", df[[column]].dropna())

def plot_count(df, column):
    fig, ax = plt.subplots()
    sns.countplot(x=df[column], ax=ax)
    ax.set_title(f"Count Plot of {column}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    export_plot(fig, f"{column}_countplot", df[[column]].dropna())

def plot_line(df, column):
    fig, ax = plt.subplots()
    df[column].dropna().reset_index(drop=True).plot(ax=ax)
    ax.set_title(f"Line Plot of {column}")
    st.pyplot(fig)
    export_plot(fig, f"{column}_lineplot", df[[column]].dropna())

def plot_heatmap_nulls(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Heatmap of Missing Values")
    st.pyplot(fig)
    export_plot(fig, "missing_values_heatmap", df.isnull().astype(int).reset_index())
