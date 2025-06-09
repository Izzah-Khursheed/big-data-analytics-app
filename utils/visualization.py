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
    if len(cols) == 1:
        col = cols[0]
        fig, ax = plt.subplots()
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), bins=30, ax=ax)
        else:
            counts = df[col].value_counts()
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        ax.set_title("Histogram / Frequency Plot")
        st.pyplot(fig)

    elif len(cols) == 2:
        x_col, y_col = cols
        if pd.api.types.is_numeric_dtype(df[y_col]):
            grouped = df.groupby(x_col)[y_col].count().reset_index(name='Count')
            fig = px.bar(grouped, x=x_col, y='Count',
                         labels={x_col: x_col, 'Count': f'Count of {y_col}'},
                         title=f"Histogram: Count of {y_col} grouped by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Second column should be numeric for meaningful histogram-style aggregation.")
    else:
        st.warning("Select only one or two columns for histogram.")

def plot_line_chart(df, cols):
    # Use datetime or index as x-axis
    time_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_col = col
            break

    if time_col:
        fig = px.line(df, x=time_col, y=cols, title="Line Chart Over Time")
    else:
        fig = px.line(df.reset_index(), y=cols, title="Line Chart (Index-based)")

    st.plotly_chart(fig, use_container_width=True)

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
