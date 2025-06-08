import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

CLASSIFIERS = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Logistic Regression": LogisticRegression,
    "Naive Bayes": GaussianNB,
    "Random Forest": RandomForestClassifier,
}

REGRESSORS = {
    "Linear Regression": LinearRegression,
    "KNN Regressor": KNeighborsRegressor,
    "Decision Tree Regressor": DecisionTreeRegressor,
    "SVM Regressor": SVR,
}

def run_model_training_tab(df: pd.DataFrame):
    st.header("Model Training & Evaluation")

    data = st.session_state.get('preprocessed_data', df)

    # Select target variable
    target = st.selectbox("Select target variable (dependent variable)", options=data.columns)
    feature_cols = [col for col in data.columns if col != target]

    # Detect if classification or regression based on target dtype (simple heuristic)
    if pd.api.types.is_numeric_dtype(data[target]):
        task = st.selectbox("Select task type", options=["Regression", "Classification"], index=0)
    else:
        task = "Classification"

    if task == "Classification":
        model_name = st.selectbox("Select classification algorithm", options=list(CLASSIFIERS.keys()))
    else:
        model_name = st.selectbox("Select regression algorithm", options=list(REGRESSORS.keys()))

    test_size = st.slider("Test data size (percentage)", min_value=10, max_value=50, value=20)

    if st.button("Train Model"):
        X = data[feature_cols]
        y = data[target]

        # Handle categorical data (simple one-hot for features)
        X = pd.get_dummies(X)
        if task == "Classification":
            if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
                y = pd.factorize(y)[0]  # encode labels

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        if task == "Classification":
            model = CLASSIFIERS[model_name]()
        else:
            model = REGRESSORS[model_name]()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluation & display
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.json(report)

            # Feature importance if applicable
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Importance")
                feat_imp = model.feature_importances_
                feat_names = X.columns
                fig2, ax2 = plt.subplots()
                sns.barplot(x=feat_imp, y=feat_names, ax=ax2)
                ax2.set_title("Feature Importance")
                st.pyplot(fig2)
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success(f"Mean Squared Error: {mse:.4f}")
            st.success(f"R-squared: {r2:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
