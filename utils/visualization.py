import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run_visualization_tab(df: pd.DataFrame):
    st.header("Data Visualization")

    data = st.session_state.get('preprocessed_data', df)

    all_columns = data.columns.tolist()
    selected_cols = st.multiselect("Select one or more columns to visualize", all_columns)

    chart_type = st.selectbox(
        "Select chart type",
        options=["Bar Chart", "Pie Chart", "Histogram", "Line Chart", "Scatter Plot"]
    )

    if st.button("Show Visualization") and selected_cols:
        st.markdown(f"### {chart_type}")

        if chart_type == "Bar Chart":
            plot_bar_chart(data, selected_cols)
        elif chart_type == "Pie Chart":
            plot_pie_chart(data, selected_cols)
        elif chart_type == "Histogram":
            plot_histogram(data, selected_cols)
        elif chart_type == "Line Chart":
            plot_line_chart(data, selected_cols)
        elif chart_type == "Scatter Plot":
            if len(selected_cols) >= 2:
                plot_scatter_chart(data, selected_cols[0], selected_cols[1])
            else:
                st.warning("Select at least two columns for Scatter Plot.")

def plot_bar_chart(df, cols):
    if len(cols) == 1:
        col = cols[0]
        counts = df[col].value_counts()
        fig = px.bar(x=counts.index, y=counts.values, labels={'x': col, 'y': 'Count'}, title=f"Bar Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bar chart supports only single-column categorical data.")

def plot_pie_chart(df, cols):
    if len(cols) == 1:
        col = cols[0]
        counts = df[col].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title=f"Pie Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pie chart supports only single-column categorical data.")

def plot_histogram(df, cols):
    fig, ax = plt.subplots()
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=False, ax=ax, label=col, bins=30)
        else:
            st.warning(f"Skipping non-numeric column {col} for histogram.")
    ax.legend()
    ax.set_title("Histogram")
    st.pyplot(fig)

def plot_line_chart(df, cols):
    fig = px.line(df, y=cols, title="Line Chart")
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_chart(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
    st.plotly_chart(fig, use_container_width=True)
