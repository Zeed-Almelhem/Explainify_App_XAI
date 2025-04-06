#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Initialize session state for model selection and data
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'custom_model' not in st.session_state:
    st.session_state.custom_model = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'is_custom' not in st.session_state:
    st.session_state.is_custom = False
if 'start_exploration' not in st.session_state:
    st.session_state.start_exploration = False
if 'prev_model_type' not in st.session_state:
    st.session_state.prev_model_type = None
if 'prev_selected_model' not in st.session_state:
    st.session_state.prev_selected_model = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'last_uploaded_model' not in st.session_state:
    st.session_state.last_uploaded_model = None

def set_sidebar_style():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            padding-top: 2rem;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: white;
        }
        
        .sidebar-title {
            color: white;
            font-size: 24px;
            margin-left: 10px;
        }
        
        .sidebar-section {
            color: white;
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            background-color: #1E1E1E;
            padding: 1rem 0.5rem;
        }
        
        .sidebar-logo {
            width: 50px;
            height: 50px;
            margin-right: 1rem;
            filter: invert(1);
        }
        
        /* Custom button styling */
        .stButton > button {
            background-color: #2E4F4F;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #0E8388;
            color: white;
            border: none;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(14, 131, 136, 0.3);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #2E4F4F;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #0E8388;
            color: white;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0E8388 !important;
            color: white !important;
        }
        
        /* Selectbox styling */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #2E4F4F;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stSelectbox [data-baseweb="select"] > div:hover {
            background-color: #0E8388;
            border: none;
        }
        
        /* Remove default padding from sidebar content */
        [data-testid="stSidebarContent"] {
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def handle_file_upload(require_target=True):
    # Only reset exploration if a new file is uploaded
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'], key="dataset_uploader")
    
    # Reset only if file changed
    if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.start_exploration = False
        st.session_state.last_uploaded_file = uploaded_file
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            
            # Allow user to select target column if required
            if require_target:
                columns = data.columns.tolist()
                target_col = st.selectbox(
                    "Select target column (y)",
                    columns,
                    key="target_column_selector"
                )
                st.session_state.target_column = target_col
            
            # Show data preview
            st.markdown("### Data Preview")
            st.dataframe(data.head())
            
            # Show basic statistics
            st.markdown("### Dataset Information")
            st.write(f"Number of rows: {len(data)}")
            st.write(f"Number of columns: {len(data.columns)}")
            
            return True
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return False
    return False

def handle_text_upload():
    uploaded_file = st.file_uploader("Upload your text dataset (CSV)", type=['csv'], key="text_uploader")
    text_column = None
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            
            # Select text column
            columns = data.columns.tolist()
            text_column = st.selectbox(
                "Select text column",
                columns,
                key="text_column_selector"
            )
            
            # Select label column if available
            label_column = st.selectbox(
                "Select label column (optional)",
                ["None"] + columns,
                key="label_column_selector"
            )
            
            if label_column != "None":
                st.session_state.target_column = label_column
            
            # Show data preview
            st.markdown("### Data Preview")
            st.dataframe(data.head())
            
            return True
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return False
    return False

def handle_model_upload():
    # Only reset exploration if a new model is uploaded
    if "last_uploaded_model" not in st.session_state:
        st.session_state.last_uploaded_model = None
    
    model_library = st.selectbox(
        "Select the library used for training",
        ["scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "LightGBM", "CatBoost"],
        key="model_library"
    )
    
    model_file = st.file_uploader("Upload your trained model file", type=['pkl', 'joblib', 'h5', 'pt'], key="model_uploader")
    
    # Reset only if model changed
    if model_file is not None and model_file != st.session_state.last_uploaded_model:
        st.session_state.start_exploration = False
        st.session_state.last_uploaded_model = model_file
    
    if model_file is not None:
        try:
            # For scikit-learn models (pkl/joblib files)
            if model_file.name.endswith(('.pkl', '.joblib')):
                model = pickle.load(model_file)
                st.session_state.custom_model = model
                st.success(f"Model loaded successfully!")
                return True
            # For other frameworks, we'll need to implement specific loading logic
            else:
                st.warning("Support for this model format coming soon!")
                return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    return False

class Model:
    def __init__(self, name):
        self.name = name
        self.is_coming_soon = False

    def __str__(self):
        return self.name

def models(model_name):
    model = Model(model_name)
    if model_name in ["Brazilian E-Commerce Dataset", "No-show Flight Prediction Dataset", "International Football Results", "Air Quality Dataset", "20 Newsgroups Classification"]:
        model.is_coming_soon = True
    return model

def main():
    st.set_page_config(layout='wide', page_icon='🤖', page_title='ML Model Explorer')
    set_sidebar_style()

    # Sidebar
    with st.sidebar:
        # Logo and Title in a row
        st.markdown("""
            <div class="header-container">
                <img src="https://raw.githubusercontent.com/Zeed-Almelhem/Explainify/6f7fab71d30a185b9f88acfe59b4b8ff44137f3d/analyze-svgrepo-com.svg" class="sidebar-logo">
                <div class="sidebar-title">Explainify</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Selection Header
        st.markdown('<div class="sidebar-section">Model Selection</div>', unsafe_allow_html=True)
        
        # Model type selection
        if 'prev_model_type' not in st.session_state:
            st.session_state.prev_model_type = None
            
        model_type = st.radio(
            "Select Model Type",
            ["Classification", "Regression", "Clustering", "Natural Language Processing"],
            key="model_type_radio"
        )
        
        # Only reset exploration if model type actually changed
        if st.session_state.prev_model_type != model_type:
            st.session_state.start_exploration = False
            st.session_state.prev_model_type = model_type

        st.markdown("---")

        # Model options based on type
        if model_type == "Classification":
            model_options = [
                "Binary Customer Churn Classifier",
                "Brazilian E-Commerce Dataset",
                "No-show Flight Prediction Dataset",
                "Custom Dataset and Model"
            ]
        elif model_type == "Regression":
            model_options = [
                "House Price Predictor",
                "International Football Results",
                "Air Quality Dataset",
                "Custom Dataset and Model"
            ]
        elif model_type == "Clustering":
            model_options = [
                "Customer Segmentation (Wholesale)",
                "Custom Dataset and Model"
            ]
        else:  # NLP
            model_options = [
                "20 Newsgroups Classification",
                "Custom Dataset and Model"
            ]
            
        if 'prev_selected_model' not in st.session_state:
            st.session_state.prev_selected_model = None

        selected_model = st.selectbox(
            "Select Model",
            model_options,
            key="model_selector"
        )
        
        # Only reset exploration if model actually changed
        if st.session_state.prev_selected_model != selected_model:
            st.session_state.start_exploration = False
            st.session_state.prev_selected_model = selected_model

        # Save selections to session state
        st.session_state.model_type = model_type
        st.session_state.selected_model = selected_model
        st.session_state.is_custom = selected_model == "Custom Dataset and Model"

    # Main content area
    st.title("ML Model Explorer")

    if not st.session_state.is_custom:
        # Display model description
        st.markdown("### Dataset Description")
        
        # Create model instance to check if it's coming soon
        model = models(st.session_state.selected_model)
        
        if model.is_coming_soon:
            st.warning("🚧 This model is currently in development and will be available after the hackathon! Stay tuned! 🚧")
            if st.session_state.model_type == "Classification":
                if st.session_state.selected_model == "Brazilian E-Commerce Dataset":
                    st.write("Coming soon: Predict order status in Brazilian e-commerce transactions.")
                elif st.session_state.selected_model == "No-show Flight Prediction Dataset":
                    st.write("Coming soon: Predict whether a passenger will show up for their flight.")
            elif st.session_state.model_type == "Regression":
                if st.session_state.selected_model == "International Football Results":
                    st.write("Coming soon: Predict football match scores based on historical data.")
                elif st.session_state.selected_model == "Air Quality Dataset":
                    st.write("Coming soon: Predict air quality index based on various environmental factors.")
            else:  # NLP
                if st.session_state.selected_model == "20 Newsgroups Classification":
                    st.write("Coming soon: Classify news articles into 20 different categories using natural language processing.")
        else:
            if st.session_state.model_type == "Classification":
                if st.session_state.selected_model == "Binary Customer Churn Classifier":
                    st.write("Predict customer churn based on various customer attributes and behaviors.")
                    
                    # Load and display dataset info
                    df = pd.read_csv('files/Classification/customer_churn_classification_dataset.csv')
                    target_col = 'Churn'
                    
                    # Show target column
                    st.markdown("#### Target Column")
                    st.info(f"Target Column: {target_col}")
                    
                    # Show data preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head())
                    
                    # Show dataset info
                    st.markdown("#### Dataset Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Rows", df.shape[0])
                    with col2:
                        st.metric("Number of Features", df.shape[1] - 1)  # Subtract target column
                    
                    # Add Start Exploration button
                    if st.button("🔍 Start Model Exploration", key="explore_churn"):
                        st.session_state.start_exploration = True
                        
            elif st.session_state.model_type == "Regression":
                if st.session_state.selected_model == "House Price Predictor":
                    st.write("Predict house prices based on various features like location, size, and amenities.")
                    
                    # Load and display dataset info
                    df = pd.read_csv('files/Regression/house_price/house_price_regression_dataset.csv')
                    target_col = 'Price'
                    
                    # Show target column
                    st.markdown("#### Target Column")
                    st.info(f"Target Column: {target_col}")
                    
                    # Show data preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head())
                    
                    # Show dataset info
                    st.markdown("#### Dataset Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Rows", df.shape[0])
                    with col2:
                        st.metric("Number of Features", df.shape[1] - 1)  # Subtract target column
                    
                    # Add Start Exploration button
                    if st.button("🔍 Start Model Exploration", key="explore_house"):
                        st.session_state.start_exploration = True
                    
            elif st.session_state.model_type == "Clustering":
                if st.session_state.selected_model == "Customer Segmentation (Wholesale)":
                    st.write("Segment wholesale customers based on their annual spending across different product categories.")
                    
                    # Load and display dataset info
                    df = pd.read_csv('files/Clustering/wholesale_customers_clustering_dataset.csv')
                    
                    # Show data preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head())
                    
                    # Show dataset info
                    st.markdown("#### Dataset Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Rows", df.shape[0])
                    with col2:
                        st.metric("Number of Features", df.shape[1])  # All columns are features in clustering
                    
                    # Add Start Exploration button
                    if st.button("🔍 Start Model Exploration", key="explore_cluster"):
                        st.session_state.start_exploration = True
    else:
        # Custom dataset and model upload interface
        st.markdown("### Upload Your Data")
        if st.session_state.model_type in ["Classification", "Regression"]:
            if handle_file_upload(require_target=True):
                st.success(f"Dataset loaded successfully! Target column: {st.session_state.target_column}")
                
                # Show model upload section only after data is loaded
                st.markdown("### Upload Your Model")
                if handle_model_upload():
                    # Add Start Exploration button for custom models
                    if st.button("🔍 Start Model Exploration", key="explore_custom"):
                        st.session_state.start_exploration = True
                        
        elif st.session_state.model_type == "Clustering":
            if handle_file_upload(require_target=False):
                st.success("Dataset loaded successfully!")
                
                # Show model upload section only after data is loaded
                st.markdown("### Upload Your Model")
                if handle_model_upload():
                    # Add Start Exploration button for custom models
                    if st.button("🔍 Start Model Exploration", key="explore_custom_cluster"):
                        st.session_state.start_exploration = True
        else:  # NLP
            if handle_text_upload():
                st.success("Text data loaded successfully!")
                
                # Show model upload section only after data is loaded
                st.markdown("### Upload Your Model")
                if handle_model_upload():
                    # Add Start Exploration button for custom models
                    if st.button("🔍 Start Model Exploration", key="explore_custom_nlp"):
                        st.session_state.start_exploration = True

    # Add visualization section if exploration has started
    if 'start_exploration' in st.session_state and st.session_state.start_exploration:
        st.markdown("---")
        st.markdown("## Model Exploration")
        
        # Create tabs for different types of visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs([
            "📊 Data Analysis",
            "🎯 Model Performance",
            "🔍 Feature Importance"
        ])
        
        with viz_tab1:
            st.markdown("### Data Analysis")
            
            # Get the appropriate data based on model type
            if not st.session_state.is_custom:
                if st.session_state.model_type == "Classification":
                    if st.session_state.selected_model == "Binary Customer Churn Classifier":
                        df = pd.read_csv('files/Classification/customer_churn_classification_dataset.csv')
                        target_col = 'Churn'
                elif st.session_state.model_type == "Regression":
                    if st.session_state.selected_model == "House Price Predictor":
                        df = pd.read_csv('files/Regression/house_price/house_price_regression_dataset.csv')
                        target_col = 'Price'
                elif st.session_state.model_type == "Clustering":
                    if st.session_state.selected_model == "Customer Segmentation (Wholesale)":
                        df = pd.read_csv('files/Clustering/wholesale_customers_clustering_dataset.csv')
                        target_col = None
            else:
                df = st.session_state.uploaded_data
                target_col = st.session_state.target_column if st.session_state.model_type != "Clustering" else None
            
            # Data Analysis Visualizations
            if df is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot Type Selection
                    if 'plot_type' not in st.session_state:
                        st.session_state.plot_type = "Distribution"
                    
                    plot_type = st.selectbox(
                        "Select Plot Type",
                        ["Distribution", "Box Plot", "Correlation", "Scatter Plot"],
                        key="viz_plot_type",  
                        on_change=None  
                    )
                    st.session_state.plot_type = plot_type
                
                with col2:
                    # Feature Selection for Analysis
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    if target_col in numeric_cols:
                        numeric_cols.remove(target_col)
                    
                    if 'selected_feature' not in st.session_state:
                        st.session_state.selected_feature = numeric_cols[0] if numeric_cols else None
                    
                    selected_feature = st.selectbox(
                        "Select Feature to Analyze",
                        numeric_cols,
                        key="viz_feature_selector",  
                        on_change=None  
                    )
                    st.session_state.selected_feature = selected_feature
                
                # Create visualization based on selection
                if plot_type == "Distribution":
                    fig = px.histogram(df, x=selected_feature, title=f"Distribution of {selected_feature}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Box Plot":
                    fig = px.box(df, y=selected_feature, title=f"Box Plot of {selected_feature}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Correlation":
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Feature Correlation Matrix",
                                  color_continuous_scale='RdBu')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Scatter Plot" and target_col and target_col in df.columns:
                    fig = px.scatter(df, x=selected_feature, y=target_col,
                                   title=f"{selected_feature} vs {target_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.markdown("### Model Performance")
            
            if st.session_state.model_type != "Clustering":
                # Classification Metrics
                if st.session_state.model_type == "Classification":
                    st.markdown("#### Classification Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", "0.85")
                    with col2:
                        st.metric("Precision", "0.83")
                    with col3:
                        st.metric("Recall", "0.87")
                    
                    # ROC Curve using plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
                    fig.add_trace(go.Scatter(x=[0, 0.2, 0.5, 0.8, 1], 
                                          y=[0, 0.4, 0.7, 0.9, 1], 
                                          mode='lines', 
                                          name='Model'))
                    fig.update_layout(title='ROC Curve',
                                    xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Regression Metrics
                elif st.session_state.model_type == "Regression":
                    st.markdown("#### Regression Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", "0.82")
                    with col2:
                        st.metric("MAE", "0.15")
                    with col3:
                        st.metric("RMSE", "0.23")
                    
                    # Residual Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[0, 1, 2, 3], 
                                          y=[0.1, -0.1, 0.05, -0.05], 
                                          mode='markers',
                                          name='Residuals'))
                    fig.update_layout(title='Residual Plot',
                                    xaxis_title='Predicted Values',
                                    yaxis_title='Residuals')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Clustering Performance
            else:
                st.markdown("#### Clustering Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", "0.68")
                with col2:
                    st.metric("Calinski-Harabasz Score", "156.32")
                
                # Cluster Visualization
                fig = go.Figure()
                # Add sample cluster visualization
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.markdown("### Feature Importance")
            
            if st.session_state.model_type != "Clustering":
                # Feature importance plot
                importances = {
                    'Feature 1': 0.3,
                    'Feature 2': 0.25,
                    'Feature 3': 0.2,
                    'Feature 4': 0.15,
                    'Feature 5': 0.1
                }
                
                fig = px.bar(
                    x=list(importances.keys()),
                    y=list(importances.values()),
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP Values
                st.markdown("#### SHAP Values")
                st.info("SHAP values show how each feature contributes to the model's predictions")
            else:
                st.markdown("#### Cluster Feature Analysis")
                # Add cluster-specific feature analysis

if __name__ == '__main__':
    main()
