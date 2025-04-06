import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'files/Clustering/wholesale_customers_clustering_dataset.csv'
df = pd.read_csv(file_path)

# Create the Streamlit app
st.title("Wholesale Customers Dataset Analysis")

# Show dataset info
st.markdown("### Dataset Overview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

# Show basic statistics
st.markdown("### Basic Statistics")
st.dataframe(df.describe())

# Distribution plots
st.markdown("### Feature Distributions")
for column in df.columns:
    if column not in ['Channel', 'Region']:  # Skip categorical columns
        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        st.plotly_chart(fig)

# Correlation heatmap
st.markdown("### Correlation Analysis")
corr = df.corr()
fig = px.imshow(corr, 
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu')
st.plotly_chart(fig)

# Scatter plots
st.markdown("### Feature Relationships")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox("Select X Feature", numeric_cols)
with col2:
    y_feature = st.selectbox("Select Y Feature", numeric_cols)

fig = px.scatter(df, x=x_feature, y=y_feature,
                 title=f"{x_feature} vs {y_feature}")
st.plotly_chart(fig)

# Box plots by Channel
st.markdown("### Distribution by Channel")
for column in numeric_cols:
    fig = px.box(df, x='Channel', y=column,
                 title=f"{column} Distribution by Channel")
    st.plotly_chart(fig)