import streamlit as st
import pandas as pd

def load_dataset(uploaded_file):
    """Load dataset from uploaded file."""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None
            st.success(f"Loaded dataset: {uploaded_file.name} (shape: {df.shape})")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        st.info("Please upload a dataset to start.")
        return None
