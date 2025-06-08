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
    
    # Step 4: Feature Engineering
    st.markdown("### 4. Feature Engineering")
    st.write("Automatically generate new features from existing columns.")

    fe_df = filtered_df.copy()
    new_features = []

    # 1. Datetime features
    datetime_cols = fe_df.select_dtypes(include=['datetime', 'object']).columns
    for col in datetime_cols:
        try:
            fe_df[col] = pd.to_datetime(fe_df[col])
            fe_df[f"{col}_year"] = fe_df[col].dt.year
            fe_df[f"{col}_month"] = fe_df[col].dt.month
            fe_df[f"{col}_day"] = fe_df[col].dt.day
            fe_df[f"{col}_weekday"] = fe_df[col].dt.weekday
            new_features.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"])
        except Exception:
            continue

    # 2. Ratio and difference features from numeric columns
    numeric_cols = fe_df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                fe_df[f"{col1}_plus_{col2}"] = fe_df[col1] + fe_df[col2]
                fe_df[f"{col1}_minus_{col2}"] = fe_df[col1] - fe_df[col2]
                fe_df[f"{col1}_div_{col2}"] = fe_df[col1] / (fe_df[col2] + 1e-6)
                new_features.extend([f"{col1}_plus_{col2}", f"{col1}_minus_{col2}", f"{col1}_div_{col2}"])

    # 3. Boolean from age if exists
    if 'age' in fe_df.columns or 'Age' in fe_df.columns:
        age_col = 'age' if 'age' in fe_df.columns else 'Age'
        fe_df['is_adult'] = fe_df[age_col] >= 18
        new_features.append('is_adult')

    # 4. Text length features
    text_cols = fe_df.select_dtypes(include=['object']).columns
    for col in text_cols:
        fe_df[f"{col}_len"] = fe_df[col].astype(str).apply(len)
        new_features.append(f"{col}_len")

    if new_features:
        st.success(f"Created {len(new_features)} new features.")
        st.write("Preview of engineered features:")
        st.dataframe(fe_df[new_features].head())
    else:
        st.info("No features were created.")

    st.markdown("---")

    # Final preprocessed data preview
    st.subheader("Final Preprocessed Dataset")
    st.dataframe(fe_df.head(100))

    # Store preprocessed data in session state for further tabs
    st.session_state['preprocessed_data'] = fe_df

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
