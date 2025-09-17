import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------- Models ----------------------
def run_models(num_df):
    st.subheader("🤖 Machine Learning Models")
    try:
        X = num_df.iloc[:, :-1]
        y = num_df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        st.write(f"*Linear Regression R²:* {r2_score(y_test, y_pred_lr):.3f}")
        st.write(f"*Linear Regression RMSE:* {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.3f}")

        # Random Forest
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        st.write(f"*Random Forest R²:* {r2_score(y_test, y_pred_rf):.3f}")
        st.write(f"*Random Forest RMSE:* {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")

    except Exception as e:
        st.error(f"⚠ Model training failed: {e}")


# ---------------------- Feature Selection ----------------------
def feature_selection(num_df):
    st.subheader("📌 Feature Importance / Selection")
    try:
        X = num_df.iloc[:, :-1]
        y = num_df.iloc[:, -1]

        selector = SelectKBest(score_func=f_regression, k="all")
        selector.fit(X, y)
        scores = selector.scores_

        feat_scores = pd.DataFrame({
            "Feature": X.columns,
            "Score": scores
        }).sort_values(by="Score", ascending=False)

        st.write(feat_scores)

        fig, ax = plt.subplots()
        sns.barplot(data=feat_scores, x="Score", y="Feature", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠ Feature selection failed: {e}")


# ---------------------- Clustering ----------------------
def clustering(num_df):
    st.subheader("🔗 KMeans Clustering")
    try:
        X = num_df.iloc[:, :-1]
        k = st.slider("Select number of clusters (k)", 2, 10, 3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(X)

        num_df["Cluster"] = clusters
        st.write("✅ Cluster labels added")
        st.write(num_df.head())

        fig, ax = plt.subplots()
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=clusters, palette="tab10", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠ Clustering failed: {e}")
