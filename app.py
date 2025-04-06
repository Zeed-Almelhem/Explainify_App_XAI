#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle

# Initialize session state for model selection and data
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'custom_model' not in st.session_state:
    st.session_state.custom_model = None
if 'is_custom' not in st.session_state:
    st.session_state.is_custom = False

def set_sidebar_style():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            padding: 0;
        }
        .header-container {
            background-color: #1E1E1E;
            padding: 1rem 0.5rem;
            margin-bottom: 2rem;
            color: white;
            display: flex;
            align-items: center;
        }
        .sidebar-logo {
            width: 50px;
            height: 50px;
            margin-right: 1rem;
        }
        .sidebar-title {
            font-size: 2.5rem !important;
            font-weight: bold;
            margin: 0;
        }
        .sidebar-section {
            background-color: #1E1E1E;
            padding: 0 0.5rem;
            margin-bottom: 1rem;
            color: white;
            font-size: 1.5rem !important;
            font-weight: bold;
            text-align: left;
        }
        .stRadio > label {
            color: white !important;
            padding-left: 0.5rem;
        }
        .stSelectbox > label {
            color: white !important;
            padding-left: 0.5rem;
        }
        /* Adjust radio buttons alignment */
        .stRadio > div {
            padding-left: 0.5rem;
        }
        /* Adjust selectbox alignment */
        .stSelectbox > div {
            padding-left: 0.5rem;
        }
        /* Remove default padding from sidebar content */
        [data-testid="stSidebarContent"] {
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def handle_file_upload(require_target=True):
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'], key="dataset_uploader")
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
    model_library = st.selectbox(
        "Select the library used for training",
        ["scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "LightGBM", "CatBoost"],
        key="model_library"
    )
    
    model_file = st.file_uploader("Upload your trained model file", type=['pkl', 'joblib', 'h5', 'pt'], key="model_uploader")
    
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
        model_type = st.radio(
            "Select Model Type",
            ["Classification", "Regression", "Clustering", "Natural Language Processing"],
            key="model_type_radio"
        )

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

        selected_model = st.selectbox(
            "Select Model",
            model_options,
            key="model_selector"
        )

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
    else:
        # Custom dataset and model upload interface
        st.markdown("### Upload Your Data")
        if st.session_state.model_type in ["Classification", "Regression"]:
            if handle_file_upload(require_target=True):
                st.success(f"Dataset loaded successfully! Target column: {st.session_state.target_column}")
        elif st.session_state.model_type == "Clustering":
            if handle_file_upload(require_target=False):
                st.success("Dataset loaded successfully!")
        else:  # NLP
            if handle_text_upload():
                if st.session_state.target_column:
                    st.success(f"Text dataset loaded successfully! Label column: {st.session_state.target_column}")
                else:
                    st.success("Text dataset loaded successfully!")
        
        # Model upload section
        if st.session_state.uploaded_data is not None:
            st.markdown("### Upload Your Model")
            handle_model_upload()

if __name__ == '__main__':
    main()
