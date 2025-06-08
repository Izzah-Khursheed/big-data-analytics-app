import streamlit as st
import pandas as pd

from utils import preprocessing, visualization, model_training, ai_insights, groq_chatbot

# Set page config
st.set_page_config(
    page_title="Big Data Analytics Web App",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="logo.png",
)

# Load custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar - App title and logo
st.sidebar.image("assets/logo.png", width=120)
st.sidebar.title("Big Data Analytics")

# --- Step 1: Data Upload ---
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (CSV, Excel, etc.)", type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    # Load data
    try:
        if uploaded_file.name.endswith(("xlsx", "xls")):
            raw_df = pd.read_excel(uploaded_file)
        else:
            raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.success(f"Dataset loaded successfully! Shape: {raw_df.shape}")

    # Store raw data in session state on first load
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = raw_df
    if "preprocessed_df" not in st.session_state:
        st.session_state.preprocessed_df = raw_df.copy()

    # --- Show Tabs ---
    tabs = st.tabs(
        [
            "1️⃣ Data Upload (Preview)",
            "2️⃣ Preprocessing",
            "3️⃣ Data Visualization",
            "4️⃣ Model Training & Evaluation",
            "5️⃣ AI-Powered Insights",
            "6️⃣ Algorithm Reference",
            "7️⃣ Chatbot",
        ]
    )

    # Tab 1: Data Upload (Preview)
    with tabs[0]:
        st.header("Raw Dataset Preview")
        st.dataframe(st.session_state.raw_df.head(100))  # show top 100 rows max for performance

    # Tab 2: Preprocessing
    with tabs[1]:
        # Run preprocessing tab and update preprocessed_df in session state
        updated_df = preprocessing.run_preprocessing_tab(st.session_state.preprocessed_df)
        if updated_df is not None:
            st.session_state.preprocessed_df = updated_df

    # Tab 3: Data Visualization
    with tabs[2]:
        visualization.run_visualization_tab(st.session_state.preprocessed_df)

    # Tab 4: Model Training & Evaluation
    with tabs[3]:
        model_training.run_model_training_tab(st.session_state.preprocessed_df)

    # Tab 5: AI-Powered Insights
    with tabs[4]:
        ai_insights.run_ai_insights_tab(st.session_state.preprocessed_df)

    # Tab 6: Algorithm Reference
    with tabs[5]:
        st.header("Algorithm Reference")
        st.markdown(
            """
            ### Classification Algorithms
            - **SVM:** Support Vector Machines, good for small to medium datasets.
            - **KNN:** K-Nearest Neighbors, simple and effective for classification.
            - **Decision Tree:** Tree-based model, easy to interpret.
            - **Logistic Regression:** Linear model for binary classification.
            - **Naive Bayes:** Probabilistic classifier based on Bayes theorem.
            - **Random Forest:** Ensemble of decision trees, robust and accurate.

            ### Regression Algorithms
            - **Linear Regression:** Predicts continuous values using linear model.
            - **KNN Regressor:** KNN adapted for regression tasks.
            - **Decision Tree Regressor:** Non-linear regression via tree splitting.

            *(More detailed explanations will be added here...)*
            """
        )

    # Tab 7: Chatbot (Groq API)
    with tabs[6]:
        groq_chatbot.run_groq_chatbot_tab()

else:
    st.sidebar.info("Please upload a dataset to start using the app.")
    st.title("Welcome to Big Data Analytics Web App")
    st.markdown(
        """
        Upload your dataset in the sidebar (CSV or Excel) to get started.
        """
    )
