import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def run_ai_insights_tab(df: pd.DataFrame):
    st.header("AI-Powered Insights")

    data = st.session_state.get('preprocessed_data', df)

    st.write("Generating automatic descriptive and analytical insights based on your data...")

    # -----------------------------
    # 1. Descriptive Stats
    # -----------------------------
    st.subheader("📊 Basic Statistics")
    st.dataframe(data.describe(include='all').T)

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            mean = data[col].mean()
            median = data[col].median()
            std = data[col].std()
            st.markdown(f"**{col}** → mean = {mean:.2f}, median = {median:.2f}, std = {std:.2f}")
        else:
            counts = data[col].value_counts().head(5)
            st.markdown(f"**{col}** → top categories:")
            st.write(counts)

    st.markdown("---")

    # -----------------------------
    # 2. Correlation Heatmap
    # -----------------------------
    numeric_df = data.select_dtypes(include='number')

    if numeric_df.shape[1] > 1:
        st.subheader("📈 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    st.markdown("---")

    # -----------------------------
    # 3. Trend Detection (Time-based)
    # -----------------------------
    date_cols = [col for col in data.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(data[col])]
    
    if date_cols:
        st.subheader("📅 Trend Analysis Over Time")
        time_col = st.selectbox("Select a datetime column for trend analysis:", date_cols)
        numeric_cols = numeric_df.columns.tolist()

        if numeric_cols:
            trend_metric = st.selectbox("Select a numeric column to analyze trend:", numeric_cols)
            df_trend = data.copy()
            df_trend[time_col] = pd.to_datetime(df_trend[time_col], errors='coerce')
            df_trend = df_trend.dropna(subset=[time_col])
            df_trend.set_index(time_col, inplace=True)
            df_trend = df_trend.sort_index()
            df_trend = df_trend[[trend_metric]].resample('D').mean().dropna()

            st.line_chart(df_trend)
        else:
            st.info("No numeric columns available for trend analysis.")
    else:
        st.info("No datetime column found for trend analysis.")

    st.markdown("---")

    # -----------------------------
    # 4. Hidden Pattern Detection (Clustering)
    # -----------------------------
    if numeric_df.shape[0] >= 5:
        st.subheader("🧩 Hidden Pattern Discovery (Clustering)")

        num_clusters = st.slider("Select number of clusters", 2, 6, 3)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_data)

        st.write(f"KMeans clustering applied with {num_clusters} clusters.")
        cluster_df = numeric_df.copy()
        cluster_df['Cluster'] = labels
        st.dataframe(cluster_df.head())

        fig, ax = plt.subplots()
        sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=labels, palette='viridis', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric rows to perform clustering.")

    st.markdown("---")

    # -----------------------------
    # 5. Outlier Detection
    # -----------------------------
    if numeric_df.shape[0] >= 10:
        st.subheader("🚨 Outlier Detection")

        iso = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso.fit_predict(numeric_df)

        outliers = data[outlier_labels == -1]
        st.write(f"Detected {len(outliers)} potential outliers (5% contamination):")
        st.dataframe(outliers.head())
    else:
        st.info("Not enough data for outlier detection.")

    st.markdown("---")
    st.success("✅ Advanced AI insights generation complete.")



# import streamlit as st
# import pandas as pd

# def run_ai_insights_tab(df: pd.DataFrame):
#     st.header("AI-Powered Insights")

#     data = st.session_state.get('preprocessed_data', df)

#     st.write("Generating automatic descriptive and analytical insights based on your data...")

#     # Basic stats
#     st.subheader("Basic Statistics")
#     st.dataframe(data.describe(include='all').T)

#     # Example insights:
#     for col in data.columns:
#         if pd.api.types.is_numeric_dtype(data[col]):
#             mean = data[col].mean()
#             median = data[col].median()
#             std = data[col].std()
#             st.markdown(f"**{col}**: mean = {mean:.2f}, median = {median:.2f}, std = {std:.2f}")
#         else:
#             counts = data[col].value_counts().head(5)
#             st.markdown(f"**{col}**: top categories:")
#             st.write(counts)

#     st.markdown("---")

#     # Add simple correlation heatmap for numeric data
#     numeric_df = data.select_dtypes(include='number')
#     if numeric_df.shape[1] > 1:
#         import matplotlib.pyplot as plt
#         import seaborn as sns

#         st.subheader("Correlation Heatmap")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         corr = numeric_df.corr()
#         sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)
#     else:
#         st.info("Not enough numeric columns for correlation heatmap.")

#     # You can extend with additional AI/NLP or ML-based insights here
