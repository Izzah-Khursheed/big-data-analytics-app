import streamlit as st
import pandas as pd
import numpy as np

def run_preprocessing_tab(df: pd.DataFrame):
    st.header("Data Preprocessing")

    # Step 1: Handling Missing Values
    st.markdown("### 1. Handling Missing Values")
    missing_info = df.isnull().sum()
    st.write("Missing values per column:")
    st.dataframe(missing_info[missing_info > 0])

    strategy = st.radio("Select missing value handling strategy:", 
                        options=["Drop Rows with Missing Values", "Fill Missing Values (Mean/Mode)"])
    df_cleaned = df.copy()
    if strategy == "Drop Rows with Missing Values":
        before_shape = df_cleaned.shape
        df_cleaned = df_cleaned.dropna()
        after_shape = df_cleaned.shape
        st.write(f"Dropped rows with missing values. Shape before: {before_shape}, after: {after_shape}")
    else:
        # Fill numeric columns with mean, categorical with mode
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype in [np.float64, np.int64]:
                mean_val = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)
            else:
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
        st.write("Filled missing values: numeric with mean, categorical with mode.")

    st.markdown("---")

    # Step 2: Memory Optimization
    st.markdown("### 2. Memory Optimization")
    mem_before = df_cleaned.memory_usage(deep=True).sum() / 1024**2
    st.write(f"Memory usage before optimization: {mem_before:.2f} MB")

    df_optimized = optimize_memory(df_cleaned)
    mem_after = df_optimized.memory_usage(deep=True).sum() / 1024**2
    st.write(f"Memory usage after optimization: {mem_after:.2f} MB")

    st.markdown("---")

    # Step 3: Filtering Data (Optional)
    st.markdown("### 3. Data Filtering")
    filter_column = st.selectbox("Select column to filter (optional):", options=[None] + list(df_optimized.columns))
    filtered_df = df_optimized
    if filter_column:
        unique_vals = df_optimized[filter_column].unique()
        selected_vals = st.multiselect(f"Select values to keep in '{filter_column}':", unique_vals)
        if selected_vals:
            filtered_df = df_optimized[df_optimized[filter_column].isin(selected_vals)]
            st.write(f"Filtered data shape: {filtered_df.shape}")

    st.markdown("---")

    # Step 4: Feature Engineering (Simple Example)
    st.markdown("### 4. Feature Engineering")
    st.write("Add simple new features based on existing columns (if applicable).")
    new_feature_added = False

    numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        feat1 = st.selectbox("Select first numeric column:", numeric_cols)
        feat2 = st.selectbox("Select second numeric column:", numeric_cols)
        if st.button("Create New Feature: Sum of two columns"):
            filtered_df['feature_sum'] = filtered_df[feat1] + filtered_df[feat2]
            st.success(f"Created feature 'feature_sum' = {feat1} + {feat2}")
            new_feature_added = True
    else:
        st.info("No numeric columns available for feature engineering.")

    st.markdown("---")

    # Final preprocessed data preview
    st.subheader("Final Preprocessed Dataset")
    st.dataframe(filtered_df.head(100))

    # Store preprocessed data in session state for further tabs
    st.session_state['preprocessed_data'] = filtered_df if (new_feature_added or filter_column or True) else df_optimized


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to reduce memory usage."""
    df_optimized = df.copy()
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        if col_type == 'float64':
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        elif col_type == 'int64':
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    return df_optimized
