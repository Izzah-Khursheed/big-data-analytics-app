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
st.sidebar.image("assets/logo.png", width=100)
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
            "🗂️ Data Upload (Preview)",
            "🧹 Preprocessing",
            "📊 Data Visualization",
            "🤖 Model Training & Evaluation",
            "💡 AI-Powered Insights",
            "📘 Algorithm Reference",
            "💬 Chatbot",
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
            - **SVM (Support Vector Machines):**  
              A powerful algorithm that finds the best boundary (hyperplane) to separate classes by maximizing the margin between data points of different classes. Works well for small to medium-sized datasets and supports linear and non-linear classification using kernels.
            - **KNN (K-Nearest Neighbors):**  
              A simple, instance-based learning method that classifies a data point based on the majority class of its K closest neighbors in the feature space. No training phase; useful for small datasets.
            - **Decision Tree:**  
              A tree-structured model where each node splits the data based on feature thresholds, leading to easy-to-interpret rules. Prone to overfitting but fast and intuitive.
            - **Logistic Regression:**  
              A linear model used for binary classification that estimates probabilities using the logistic (sigmoid) function. Effective for linearly separable classes.
            - **Naive Bayes:**  
              A probabilistic classifier based on Bayes' theorem assuming feature independence. Very fast and performs well on high-dimensional data, especially text classification.
            - **Random Forest:**  
              An ensemble method that builds multiple decision trees on random subsets of data and features, then aggregates their results for better accuracy and robustness against overfitting.

            ### Regression Algorithms
            - **Linear Regression:**  
              A fundamental algorithm to predict continuous target variables by fitting a linear relationship between input features and the output. Minimizes the sum of squared errors.
            - **KNN Regressor:**  
              Extension of KNN for regression tasks; predicts the target by averaging the values of the K nearest neighbors. Non-parametric and simple but sensitive to noisy data.
            - **Decision Tree Regressor:**  
              Uses tree-based splits to model non-linear relationships by dividing the feature space into regions with similar output values. Easy to visualize and interpret.
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
