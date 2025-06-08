import streamlit as st
import pandas as pd

def run_ai_insights_tab(df: pd.DataFrame):
    st.header("AI-Powered Insights")

    data = st.session_state.get('preprocessed_data', df)

    st.write("Generating automatic descriptive and analytical insights based on your data...")

    # Basic stats
    st.subheader("Basic Statistics")
    st.dataframe(data.describe(include='all').T)

    # Example insights:
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            mean = data[col].mean()
            median = data[col].median()
            std = data[col].std()
            st.markdown(f"**{col}**: mean = {mean:.2f}, median = {median:.2f}, std = {std:.2f}")
        else:
            counts = data[col].value_counts().head(5)
            st.markdown(f"**{col}**: top categories:")
            st.write(counts)

    st.markdown("---")

    # Add simple correlation heatmap for numeric data
    numeric_df = data.select_dtypes(include='number')
    if numeric_df.shape[1] > 1:
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # You can extend with additional AI/NLP or ML-based insights here
