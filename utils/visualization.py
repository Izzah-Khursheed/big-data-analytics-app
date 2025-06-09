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
        elif chart_type == "Scatter Plot":
            if len(selected_cols) >= 2:
                plot_scatter_chart(data, selected_cols[0], selected_cols[1])
            else:
                st.warning("Select at least two columns for Scatter Plot.")

# ------------------------ Plotting Functions ------------------------

def plot_bar_chart(df, cols):
    if len(cols) == 1:
        col = cols[0]
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
            st.warning(f"'{col}' seems like a continuous variable. Bar chart may not be suitable.")
        counts = df[col].value_counts()
        fig = px.bar(x=counts.index.astype(str), y=counts.values,
                     labels={'x': col, 'y': 'Count'}, title=f"Bar Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Bar chart supports only single-column categorical or discrete data.")

def plot_pie_chart(df, cols):
    if len(cols) == 1:
        col = cols[0]
        counts = df[col].value_counts()
        if counts.shape[0] > 20:
            st.warning("Too many unique values for a readable pie chart. Please choose a column with fewer categories.")
            return
        fig = px.pie(values=counts.values, names=counts.index.astype(str),
                     title=f"Pie Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pie chart supports only single-column categorical data.")

def plot_histogram(df, cols):
    if not cols:
        st.warning("Please select at least one column.")
        return

    if len(cols) == 1:
        col = cols[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[col], kde=True, bins=30, ax=ax, color='skyblue')
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning(f"'{col}' is not a numeric column.")
    
    elif len(cols) == 2:
        x_col, group_col = cols[0], cols[1]
        if not pd.api.types.is_numeric_dtype(df[x_col]):
            st.warning(f"'{x_col}' must be numeric.")
            return
        if not pd.api.types.is_categorical_dtype(df[group_col]) and not pd.api.types.is_object_dtype(df[group_col]):
            st.warning(f"'{group_col}' should be categorical for grouping.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x=x_col, hue=group_col, kde=True, bins=30, multiple="stack", ax=ax)
        ax.set_title(f"Grouped Histogram of {x_col} by {group_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    else:
        st.warning("Histogram supports only one numeric or one numeric + one categorical column.")

def plot_scatter_chart(df, x_col, y_col):
    if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
        st.warning("Scatter plot requires numeric columns for both X and Y.")
        return
    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
    st.plotly_chart(fig, use_container_width=True)





# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# def run_visualization_tab(df: pd.DataFrame):
#     st.header("Data Visualization")

#     data = st.session_state.get('preprocessed_data', df)

#     all_columns = data.columns.tolist()
#     selected_cols = st.multiselect("Select one or more columns to visualize", all_columns)

#     chart_type = st.selectbox(
#         "Select chart type",
#         options=["Bar Chart", "Pie Chart", "Histogram", "Line Chart", "Scatter Plot"]
#     )

#     if st.button("Show Visualization") and selected_cols:
#         st.markdown(f"### {chart_type}")

#         if chart_type == "Bar Chart":
#             plot_bar_chart(data, selected_cols)
#         elif chart_type == "Pie Chart":
#             plot_pie_chart(data, selected_cols)
#         elif chart_type == "Histogram":
#             plot_histogram(data, selected_cols)
#         elif chart_type == "Line Chart":
#             plot_line_chart(data, selected_cols)
#         elif chart_type == "Scatter Plot":
#             if len(selected_cols) >= 2:
#                 plot_scatter_chart(data, selected_cols[0], selected_cols[1])
#             else:
#                 st.warning("Select at least two columns for Scatter Plot.")

# def plot_bar_chart(df, cols):
#     if len(cols) == 1:
#         col = cols[0]
#         counts = df[col].value_counts()
#         fig = px.bar(x=counts.index, y=counts.values, labels={'x': col, 'y': 'Count'}, title=f"Bar Chart of {col}")
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.warning("Bar chart supports only single-column categorical data.")

# def plot_pie_chart(df, cols):
#     if len(cols) == 1:
#         col = cols[0]
#         counts = df[col].value_counts()
#         fig = px.pie(values=counts.values, names=counts.index, title=f"Pie Chart of {col}")
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.warning("Pie chart supports only single-column categorical data.")

# def plot_histogram(df, cols):
#     fig, ax = plt.subplots()

#     for col in cols:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             sns.histplot(df[col], kde=False, ax=ax, label=col, bins=30)
#         else:
#             # Plot histogram for categorical data as a bar chart of counts
#             counts = df[col].value_counts().sort_index()
#             ax.bar(counts.index.astype(str), counts.values, label=col)

#     ax.legend()
#     ax.set_title("Histogram / Frequency Chart")
#     st.pyplot(fig)

# def plot_line_chart(df, cols):
#     fig = px.line(df, y=cols, title="Line Chart")
#     st.plotly_chart(fig, use_container_width=True)

# def plot_scatter_chart(df, x_col, y_col):
#     fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
#     st.plotly_chart(fig, use_container_width=True)
