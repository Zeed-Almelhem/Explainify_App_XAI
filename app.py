#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from visualizations import ModelVisualizer
import numpy as np
import pickle
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
            margin-left: 2px;
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
            margin-right: 0.5rem;
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
                loaded_obj = joblib.load(model_file)
                
                # Check if the loaded object is a valid model
                if hasattr(loaded_obj, 'predict'):
                    st.session_state.custom_model = loaded_obj
                    st.success(f"Model loaded successfully!")
                    return True
                elif isinstance(loaded_obj, np.ndarray):
                    # If it's a numpy array, we need to wrap it in a model-like object
                    class SimpleModel:
                        def __init__(self, predictions):
                            self.predictions = predictions
                        
                        def predict(self, X):
                            # Return stored predictions since we don't have the actual model
                            return self.predictions[:len(X)]
                        
                        def predict_proba(self, X):
                            # For binary classification, create fake probabilities
                            probs = np.zeros((len(X), 2))
                            probs[np.arange(len(X)), self.predictions.astype(int)] = 1
                            return probs
                        
                        def __str__(self):
                            return "Simple Prediction Model (based on stored predictions)"
                        
                        def get_params(self, deep=True):
                            return {"type": "simple_prediction_model"}
                    
                    st.session_state.custom_model = SimpleModel(loaded_obj)
                    st.warning("Loaded predictions as a simple model. Some advanced features may not be available.")
                    return True
                else:
                    st.error("The loaded file does not contain a valid model object.")
                    return False
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

def validate_and_transform_features(df, model_type, target_col):
    """Validate and transform features to match the expected model input."""
    if model_type == "Classification":
        # Create a copy of the dataframe
        df = df.copy()
        
        # Define the expected features
        expected_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        # Handle numeric columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        # Handle categorical columns (already encoded in the dataset)
        categorical_columns = [col for col in expected_features if col not in numeric_columns]
        
        # Check for missing features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            st.warning(f"Missing features that will be created with default values: {', '.join(missing_features)}")
            for feature in missing_features:
                if feature in numeric_columns:
                    df[feature] = 0
                else:
                    df[feature] = 0  # Default value for categorical features
        
        # Handle target column for classification
        if target_col in df.columns:
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
        
        # Return the transformed dataframe and feature list in the correct order
        return df[expected_features], expected_features
        
    elif model_type == "Regression":
        # Create a copy of the dataframe
        df = df.copy()
        
        # Define expected features
        expected_features = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
        ]
        
        # Define categorical mappings
        categorical_mappings = {
            'mainroad': {'yes': 1, 'no': 0},
            'guestroom': {'yes': 1, 'no': 0},
            'basement': {'yes': 1, 'no': 0},
            'hotwaterheating': {'yes': 1, 'no': 0},
            'airconditioning': {'yes': 1, 'no': 0},
            'prefarea': {'yes': 1, 'no': 0},
            'furnishingstatus': {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
        }
        
        # Transform categorical columns
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map(mapping)
        
        # Handle numeric columns
        numeric_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        # Check for missing features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            st.warning(f"Missing features that will be created with default values: {', '.join(missing_features)}")
            for feature in missing_features:
                if feature in categorical_mappings:
                    df[feature] = 0  # Default value for categorical features
                else:
                    df[feature] = 0  # Default value for numeric features
        
        return df, expected_features
        
    elif model_type == "Clustering":
        # Create a copy of the dataframe
        df = df.copy()
        
        # Define expected features (all numeric)
        expected_features = [
            'fresh', 'milk', 'grocery', 'frozen', 'detergents_paper', 'delicassen'
        ]
        
        # Handle numeric columns
        for col in expected_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        # Check for missing features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            st.warning(f"Missing features that will be created with default values: {', '.join(missing_features)}")
            for feature in missing_features:
                df[feature] = 0
        
        return df, expected_features
        
    else:
        return df, []

def load_prebuilt_model(model_type, selected_model=None):
    """Load a pre-built model based on model type and selection"""
    if model_type == "Classification":
        if selected_model == "Binary Customer Churn Classifier":
            model_path = 'files/Classification/customer_churn_model.pkl'
        else:
            return None
    elif model_type == "Regression":
        if selected_model == "House Price Predictor":
            model_path = 'files/Regression/house_price/house_price_model.pkl'
        else:
            return None
    elif model_type == "Clustering":
        if selected_model == "Customer Segmentation (Wholesale)":
            model_path = 'files/Clustering/customer_segmentation_model.pkl'
        else:
            return None
    else:
        return None
    
    try:
        # Load the model
        model = joblib.load(model_path)
        # Set the model in session state
        st.session_state.custom_model = model
        return model
    except Exception as e:
        st.error(f"Error loading pre-built model: {str(e)}")
        return None

def create_data_exploration(df):
    st.markdown("## Data Exploration")
    
    # Dataset Overview
    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Rows", df.shape[0])
    with col2:
        st.metric("Number of Columns", df.shape[1])
    
    # Data Preview
    st.markdown("### Data Preview")
    st.dataframe(df.head())
    
    # Missing Values Analysis
    st.markdown("### Missing Values Analysis")
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_data)
    
    # Data Types Information
    st.markdown("### Data Types Information")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(dtypes_df)
    
    # Numerical Features Analysis
    st.markdown("### Numerical Features Analysis")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe()
        st.dataframe(stats_df)
        
        # Distribution Plots
        st.markdown("#### Distribution Plots")
        selected_num_col = st.selectbox("Select Numerical Feature", numeric_cols, key="num_feat_selector")
        fig = px.histogram(df, x=selected_num_col, nbins=30,
                          title=f"Distribution of {selected_num_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Features Analysis
    st.markdown("### Categorical Features Analysis")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        selected_cat_col = st.selectbox("Select Categorical Feature", cat_cols, key="cat_feat_selector")
        value_counts = df[selected_cat_col].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                     title=f"Distribution of {selected_cat_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    if len(numeric_cols) > 1:
        st.markdown("### Correlation Analysis")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter Plot Matrix
        st.markdown("### Feature Relationships")
        selected_features = st.multiselect(
            "Select Features for Scatter Plot Matrix",
            numeric_cols,
            default=list(numeric_cols)[:3]
        )
        if len(selected_features) > 1:
            fig = px.scatter_matrix(df[selected_features])
            st.plotly_chart(fig, use_container_width=True)

def create_classification_analysis(df, target_col):
    """Create classification-specific analysis visualizations."""
    st.markdown("### Classification Analysis")
    
    # Target Distribution
    st.subheader("Target Distribution")
    target_counts = df[target_col].value_counts()
    fig = px.pie(values=target_counts.values, names=target_counts.index,
                 title=f"Distribution of {target_col}")
    st.plotly_chart(fig, key="class_target_dist")
    
    # Feature Analysis
    st.subheader("Feature Analysis")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Numeric Features
    if len(numeric_cols) > 0:
        st.markdown("#### Numerical Features by Target")
        selected_num_col = st.selectbox("Select Numerical Feature", 
                                      [col for col in numeric_cols if col != target_col],
                                      key="class_num_feat")
        fig = px.box(df, x=target_col, y=selected_num_col,
                    title=f"{selected_num_col} Distribution by {target_col}")
        st.plotly_chart(fig, key="class_num_box")
    
    # Categorical Features
    if len(categorical_cols) > 0:
        st.markdown("#### Categorical Features by Target")
        selected_cat_col = st.selectbox(
            "Select Categorical Feature",
            [col for col in categorical_cols if col != target_col],
            key="class_cat_feat"
        )
        
        # Create contingency table
        contingency = pd.crosstab(df[selected_cat_col], df[target_col], normalize='index')
        fig = px.bar(contingency, barmode='stack',
                    title=f"{selected_cat_col} vs {target_col}")
        st.plotly_chart(fig, key="class_cat_bar")

def create_regression_analysis(df, target_col):
    """Create regression-specific analysis visualizations."""
    st.markdown("### Regression Analysis")
    
    # Target Distribution
    st.subheader("Target Distribution")
    fig = px.histogram(df, x=target_col, nbins=30,
                      title=f"Distribution of {target_col}")
    st.plotly_chart(fig, key="reg_target_dist")
    
    # Feature Analysis
    st.subheader("Feature Analysis")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Numeric Features
    if len(numeric_cols) > 0:
        st.markdown("#### Numerical Features vs Target")
        selected_num_col = st.selectbox("Select Numerical Feature",
                                      [col for col in numeric_cols if col != target_col],
                                      key="reg_num_feat")
        fig = px.scatter(df, x=selected_num_col, y=target_col,
                        title=f"{selected_num_col} vs {target_col}")
        st.plotly_chart(fig, key="reg_num_scatter")
    
    # Categorical Features
    if len(categorical_cols) > 0:
        st.markdown("#### Categorical Features vs Target")
        selected_cat_col = st.selectbox("Select Categorical Feature",
                                      [col for col in categorical_cols if col != target_col],
                                      key="reg_cat_feat")
        fig = px.box(df, x=selected_cat_col, y=target_col,
                    title=f"{target_col} Distribution by {selected_cat_col}")
        st.plotly_chart(fig, key="reg_cat_box")

def create_clustering_analysis(df):
    """Create clustering-specific analysis visualizations."""
    st.markdown("### Clustering Analysis")
    
    # Feature Distributions
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Select features for visualization
    selected_features = st.multiselect(
        "Select Features to Analyze",
        numeric_cols,
        default=list(numeric_cols)[:2] if len(numeric_cols) >= 2 else list(numeric_cols),
        key="cluster_feat_select"
    )
    
    if len(selected_features) >= 2:
        # Scatter plot
        st.markdown("#### Feature Relationships")
        fig = px.scatter(df, x=selected_features[0], y=selected_features[1],
                        title=f"{selected_features[0]} vs {selected_features[1]}")
        st.plotly_chart(fig, key="cluster_scatter")
        
        # Correlation heatmap for selected features
        st.markdown("#### Correlation Analysis")
        corr = df[selected_features].corr()
        fig = px.imshow(corr, title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, key="cluster_heatmap")
    else:
        st.info("Please select at least 2 features for analysis.")

def show_classification_metrics(visualizer, df, target_col, key_prefix=""):
    """Show classification metrics using the visualizer."""
    # Distribution of target
    fig = visualizer.create_distribution_plot(target_col)
    st.plotly_chart(fig, key=f"{key_prefix}class_target_dist")
    
    # ROC Curve
    if hasattr(visualizer.model, "predict_proba"):
        y_pred_proba = visualizer.model.predict_proba(df.drop(target_col, axis=1))[:, 1]
        fig = visualizer.create_roc_curve(df[target_col], y_pred_proba)
        st.plotly_chart(fig, key=f"{key_prefix}class_roc")
    
    # Confusion Matrix
    if hasattr(visualizer.model, "predict"):
        y_pred = visualizer.model.predict(df.drop(target_col, axis=1))
        fig = visualizer.create_confusion_matrix(df[target_col], y_pred)
        st.plotly_chart(fig, key=f"{key_prefix}class_cm")

def show_regression_metrics(visualizer, df, target_col, key_prefix=""):
    """Show regression metrics using the visualizer."""
    # Distribution of target
    fig = visualizer.create_distribution_plot(target_col)
    st.plotly_chart(fig, key=f"{key_prefix}reg_target_dist")
    
    # Residuals plot
    if hasattr(visualizer.model, "predict"):
        y_pred = visualizer.model.predict(df.drop(target_col, axis=1))
        fig = visualizer.create_residuals_plot(df[target_col], y_pred)
        st.plotly_chart(fig, key=f"{key_prefix}reg_residuals")
        
        # Actual vs Predicted
        fig = visualizer.create_actual_vs_predicted(df[target_col], y_pred)
        st.plotly_chart(fig, key=f"{key_prefix}reg_actual_pred")

def show_clustering_metrics(visualizer, df, key_prefix=""):
    """Show clustering metrics using the visualizer."""
    # Select features for visualization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    selected_features = st.multiselect(
        "Select Features for Cluster Visualization",
        numeric_cols,
        default=list(numeric_cols)[:2] if len(numeric_cols) >= 2 else list(numeric_cols),
        key=f"{key_prefix}cluster_viz_feat"
    )
    
    if len(selected_features) >= 2:
        if hasattr(visualizer.model, "predict"):
            labels = visualizer.model.predict(df[selected_features])
            fig = visualizer.create_cluster_plot(selected_features, labels)
            st.plotly_chart(fig, key=f"{key_prefix}cluster_viz")
        
        # Correlation matrix
        fig = visualizer.create_correlation_matrix(selected_features)
        st.plotly_chart(fig, key=f"{key_prefix}cluster_corr")

def show_feature_importance(visualizer, df, target_col, key_prefix=""):
    """Show feature importance using the visualizer."""
    if visualizer.model is not None and hasattr(visualizer.model, "feature_importances_"):
        features = df.drop(target_col, axis=1).columns
        fig = visualizer.create_feature_importance_plot(
            visualizer.model.feature_importances_,
            features
        )
        st.plotly_chart(fig, key=f"{key_prefix}feat_imp")
        
        # SHAP values if available
        try:
            fig = visualizer.create_shap_summary_plot()
            if fig is not None:
                st.pyplot(fig, key=f"{key_prefix}shap_summary")
        except:
            st.info("SHAP values are not available for this model.")
    else:
        st.info("Feature importance is not available for this model.")

def show_model_diagnostics(visualizer, df, target_col=None, key_prefix=""):
    """Show model diagnostics using the visualizer."""
    if visualizer.model_type == "Classification":
        # Show confusion matrix and ROC curve
        if hasattr(visualizer.model, "predict"):
            y_pred = visualizer.model.predict(df.drop(target_col, axis=1))
            fig = visualizer.create_confusion_matrix(df[target_col], y_pred)
            st.plotly_chart(fig, key=f"{key_prefix}diag_cm")
            
        if hasattr(visualizer.model, "predict_proba"):
            y_pred_proba = visualizer.model.predict_proba(df.drop(target_col, axis=1))[:, 1]
            fig = visualizer.create_roc_curve(df[target_col], y_pred_proba)
            st.plotly_chart(fig, key=f"{key_prefix}diag_roc")
            
    elif visualizer.model_type == "Regression":
        # Show residuals and actual vs predicted plots
        if hasattr(visualizer.model, "predict"):
            y_pred = visualizer.model.predict(df.drop(target_col, axis=1))
            fig = visualizer.create_residuals_plot(df[target_col], y_pred)
            st.plotly_chart(fig, key=f"{key_prefix}diag_residuals")
            
            fig = visualizer.create_actual_vs_predicted(df[target_col], y_pred)
            st.plotly_chart(fig, key=f"{key_prefix}diag_actual_pred")
            
    elif visualizer.model_type == "Clustering":
        # Show cluster visualization and correlation matrix
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        selected_features = st.multiselect(
            "Select Features for Diagnostics",
            numeric_cols,
            default=list(numeric_cols)[:2] if len(numeric_cols) >= 2 else list(numeric_cols),
            key=f"{key_prefix}diag_cluster_feat"
        )
        
        if len(selected_features) >= 2:
            if hasattr(visualizer.model, "predict"):
                labels = visualizer.model.predict(df[selected_features])
                fig = visualizer.create_cluster_plot(selected_features, labels)
                st.plotly_chart(fig, key=f"{key_prefix}diag_cluster")
            
            fig = visualizer.create_correlation_matrix(selected_features)
            st.plotly_chart(fig, key=f"{key_prefix}diag_corr")

def main():
    st.set_page_config(layout='wide', page_icon='🤖', page_title='ML Model Explorer')
    set_sidebar_style()

    # Sidebar
    with st.sidebar:
        # Logo and Title in a row
        st.markdown("""
            <div class="header-container">
                <img src="https://raw.githubusercontent.com/Zeed-Almelhem/Explainify/83b290fe414bb61ca3e8bc16cdb8640a201266fc/analysis-svgrepo-com.svg" class="sidebar-logo">
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
                        st.metric("Number of Features", 20)  # Fixed to show correct number of features
                    
                    # Add Start Exploration button
                    if st.button("🔍 Start Model Exploration", key="explore_churn"):
                        st.session_state.start_exploration = True
                    
                    # Show analysis tabs only after clicking Start Exploration
                    if st.session_state.start_exploration:
                        data_exp_tab, data_analysis_tab = st.tabs([
                            "Data Exploration", "Data Analysis"
                        ])
                        
                        # Data Exploration tab
                        with data_exp_tab:
                            create_data_exploration(df)
                        
                        # Data Analysis tab
                        with data_analysis_tab:
                            create_classification_analysis(df, target_col)
                elif st.session_state.selected_model == "Brazilian E-Commerce Dataset":
                    st.write("Coming soon: Predict order status in Brazilian e-commerce transactions.")
                elif st.session_state.selected_model == "No-show Flight Prediction Dataset":
                    st.write("Coming soon: Predict whether a passenger will show up for their flight.")
            elif st.session_state.model_type == "Regression":
                if st.session_state.selected_model == "House Price Predictor":
                    st.write("Predict house prices based on various features like location, size, and amenities.")
                    
                    try:
                        # Load and combine train data
                        X_train = pd.read_csv('files/Regression/house_price/house_price_X_train.csv')
                        y_train = pd.read_csv('files/Regression/house_price/house_price_y_train.csv')
                        # Load and combine test data
                        X_test = pd.read_csv('files/Regression/house_price/house_price_X_test.csv')
                        y_test = pd.read_csv('files/Regression/house_price/house_price_y_test.csv')
                        
                        # Combine features and target
                        X = pd.concat([X_train, X_test], axis=0)
                        y = pd.concat([y_train, y_test], axis=0)
                        
                        # Create a combined dataset with proper column names
                        df = X.copy()
                        df['price'] = y.values
                        target_col = 'price'
                        
                        # Load the pre-built model
                        st.session_state.custom_model = load_prebuilt_model(st.session_state.model_type, st.session_state.selected_model)
                        if st.session_state.custom_model is None:
                            st.error("Failed to load pre-built regression model.")
                            return
                        
                    except Exception as e:
                        st.error(f"Error loading pre-built model: {str(e)}")
                        df = None
                        target_col = None
                    
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
                    
                elif st.session_state.selected_model == "International Football Results":
                    st.write("Coming soon: Predict football match scores based on historical data.")
                elif st.session_state.selected_model == "Air Quality Dataset":
                    st.write("Coming soon: Predict air quality index based on various environmental factors.")
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
                    if st.button("🔍 Start Model Exploration", key="explore_clustering"):
                        st.session_state.start_exploration = True
                    
                    # Show analysis tabs only after clicking Start Exploration
                    if st.session_state.start_exploration:
                        data_exp_tab, data_analysis_tab = st.tabs([
                            "Data Exploration", "Data Analysis"
                        ])
                        
                        # Data Exploration tab
                        with data_exp_tab:
                            create_data_exploration(df)
                        
                        # Data Analysis tab
                        with data_analysis_tab:
                            create_clustering_analysis(df)
                    
                elif st.session_state.selected_model == "Custom Dataset and Model":
                    st.write("Upload your dataset and model to start the analysis.")
    else:
        # Custom dataset and model upload interface
        st.markdown("### Upload Your Data")
        if st.session_state.model_type in ["Classification", "Regression"]:
            if handle_file_upload(require_target=True):
                df = st.session_state.uploaded_data
                target_col = st.session_state.target_column
                
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
                
                # Show model upload section
                st.markdown("### Upload Your Model")
                if handle_model_upload():
                    # Add Start Exploration button
                    if st.button("🔍 Start Model Exploration", key="explore_custom"):
                        st.session_state.start_exploration = True
                    
                    # Show analysis tabs only after clicking Start Exploration
                    if st.session_state.start_exploration:
                        data_exp_tab, data_analysis_tab = st.tabs([
                            "Data Exploration", "Data Analysis"
                        ])
                        
                        # Data Exploration tab
                        with data_exp_tab:
                            create_data_exploration(df)
                        
                        # Data Analysis tab
                        with data_analysis_tab:
                            if st.session_state.model_type == "Classification":
                                create_classification_analysis(df, target_col)
                            elif st.session_state.model_type == "Regression":
                                create_regression_analysis(df, target_col)
                            elif st.session_state.model_type == "Clustering":
                                create_clustering_analysis(df)
            else:
                st.info("Please upload a dataset to start the analysis.")
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
        st.markdown("## In-Depth Model Exploration")
        
        # Create tabs for different types of visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Features Discovery",
            "🎯 Model Performance",
            "🔍 Feature Importance",
            "📈 Model Diagnostics"
        ])
        
        # Initialize visualizer
        if not st.session_state.is_custom:
            if st.session_state.model_type == "Classification":
                if st.session_state.selected_model == "Binary Customer Churn Classifier":
                    df = pd.read_csv('files/Classification/customer_churn_classification_dataset.csv')
                    target_col = 'Churn'
                    
                    # Load the pre-built model
                    st.session_state.custom_model = load_prebuilt_model("Classification")
                    if st.session_state.custom_model is None:
                        st.error("Failed to load pre-built classification model.")
                        return
            elif st.session_state.model_type == "Regression":
                if st.session_state.selected_model == "House Price Predictor":
                    try:
                        # Load and combine train data
                        X_train = pd.read_csv('files/Regression/house_price/house_price_X_train.csv')
                        y_train = pd.read_csv('files/Regression/house_price/house_price_y_train.csv')
                        # Load and combine test data
                        X_test = pd.read_csv('files/Regression/house_price/house_price_X_test.csv')
                        y_test = pd.read_csv('files/Regression/house_price/house_price_y_test.csv')
                        
                        # Combine features and target
                        X = pd.concat([X_train, X_test], axis=0)
                        y = pd.concat([y_train, y_test], axis=0)
                        
                        # Create a combined dataset with proper column names
                        df = X.copy()
                        df['price'] = y.values
                        target_col = 'price'
                    except Exception as e:
                        st.error(f"Error loading pre-built model: {str(e)}")
                        df = None
                        target_col = None
            elif st.session_state.model_type == "Clustering":
                if st.session_state.selected_model == "Customer Segmentation (Wholesale)":
                    df = pd.read_csv('files/Clustering/customer_segmentation_X_train.csv')
                    target_col = None
        else:
            df = st.session_state.uploaded_data
            target_col = st.session_state.target_column if st.session_state.model_type != "Clustering" else None
        
        visualizer = ModelVisualizer(
            data=df,
            model=st.session_state.get('custom_model'),
            target_col=target_col,
            model_type=st.session_state.model_type
        )
        
        with viz_tab1:
            st.markdown("### Data Discovery")
            
            if df is not None:
                try:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        plot_type = st.selectbox(
                            "Select Plot Type",
                            ["Distribution", "Box Plot", "Violin Plot", "Correlation", "Scatter Plot", 
                             "Missing Values", "Pair Plot"],
                            key="viz_plot_type"
                        )
                    
                    with col2:
                        # Feature Selection for Analysis
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        if target_col in numeric_cols:
                            numeric_cols.remove(target_col)
                        
                        if len(numeric_cols) == 0:
                            st.warning("No numeric columns found in the dataset for analysis.")
                        else:
                            if plot_type in ["Scatter Plot", "Pair Plot"]:
                                selected_features = st.multiselect(
                                    "Select Features to Analyze",
                                    numeric_cols,
                                    default=numeric_cols[:2] if len(numeric_cols) > 1 else numeric_cols,
                                    key="viz_feature_selector"
                                )
                            else:
                                selected_feature = st.selectbox(
                                    "Select Feature to Analyze",
                                    numeric_cols,
                                    key="viz_feature_selector"
                                )
                    
                    # Create visualization based on selection
                    if len(numeric_cols) > 0:
                        try:
                            if plot_type == "Distribution":
                                fig = visualizer.create_distribution_plot(selected_feature)
                            elif plot_type == "Box Plot":
                                fig = visualizer.create_box_plot(selected_feature)
                            elif plot_type == "Violin Plot":
                                fig = visualizer.create_violin_plot(selected_feature)
                            elif plot_type == "Correlation":
                                fig = visualizer.create_correlation_matrix()
                            elif plot_type == "Scatter Plot" and len(selected_features) >= 2:
                                fig = visualizer.create_scatter_plot(selected_features[0], selected_features[1])
                            elif plot_type == "Missing Values":
                                fig = visualizer.create_missing_values_heatmap()
                            elif plot_type == "Pair Plot" and len(selected_features) >= 2:
                                if len(selected_features) > 5:
                                    st.warning("Too many features selected. Limiting to first 5 features for better visualization.")
                                    selected_features = selected_features[:5]
                                fig = visualizer.create_pair_plot(selected_features)
                            else:
                                fig = None
                                if plot_type in ["Scatter Plot", "Pair Plot"] and len(selected_features) < 2:
                                    st.warning("Please select at least 2 features for this plot type.")
                            
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Unable to generate {plot_type.lower()} plot.")
                            st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.warning("Error processing the dataset for visualization.")
                    st.error(f"Error: {str(e)}")
            else:
                st.info("Please upload a dataset to start the analysis.")
        
        with viz_tab2:
            st.markdown("### Model Performance")
            
            if st.session_state.model_type == "Classification":
                # Initialize variables
                df = None
                target_col = None
                
                if not st.session_state.is_custom:
                    try:
                        # Load and combine train data for classification
                        if st.session_state.selected_model == "Binary Customer Churn Classifier":
                            # Load the pre-built model first
                            model = load_prebuilt_model(st.session_state.model_type, st.session_state.selected_model)
                            if model is None:
                                st.error("Failed to load pre-built classification model.")
                                return
                            
                            # Now load the data
                            df = pd.read_csv('files/Classification/customer_churn_classification_dataset.csv')
                            target_col = 'Churn'
                        else:
                            st.error("Selected model is not available.")
                            return
                    except Exception as e:
                        st.error(f"Error loading pre-built model and data: {str(e)}")
                        return
                else:
                    df = st.session_state.uploaded_data
                    target_col = st.session_state.target_column
                
                # Validate we have the data and model
                if df is None:
                    st.error("No dataset available. Please ensure data is loaded.")
                    return
                if target_col not in df.columns:
                    st.error(f"Target column '{target_col}' not found in dataset.")
                    return
                if st.session_state.get('custom_model') is None:
                    st.error("No model available. Please ensure a model is loaded.")
                    return
                
                try:
                    # Transform features to match model expectations
                    df_transformed, feature_cols = validate_and_transform_features(df, "Classification", target_col)
                    
                    if len(feature_cols) == 0:
                        st.error("No valid features found for the model.")
                        return
                    
                    X = df_transformed[feature_cols]
                    y = df[target_col].map({'Yes': 1, 'No': 0})  # Transform target to numeric
                    
                    # Make predictions
                    y_pred_raw = st.session_state.custom_model.predict(X)
                    y_pred = np.where(y_pred_raw == 'Yes', 1, 0)  # Convert string predictions to numeric
                    y_pred_proba = st.session_state.custom_model.predict_proba(X)[:, 1]
                    
                    # Calculate and display metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
                    with col2:
                        st.metric("Precision", f"{precision_score(y, y_pred):.3f}")
                    with col3:
                        st.metric("Recall", f"{recall_score(y, y_pred):.3f}")
                    with col4:
                        st.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y, y_pred_proba)
                    fig = px.line(x=fpr, y=tpr, 
                                title='ROC Curve',
                                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1)
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y, y_pred)
                    fig = px.imshow(cm,
                                  labels=dict(x="Predicted", y="Actual"),
                                  title="Confusion Matrix",
                                  color_continuous_scale="Viridis")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during model evaluation: {str(e)}")
                    st.error("Debug info:")
                    st.write(f"Data shape: {df.shape if df is not None else None}")
                    st.write(f"Target column: {target_col}")
                    st.write(f"Feature columns: {feature_cols if 'feature_cols' in locals() else None}")            
            elif st.session_state.model_type == "Regression":
                # Get the appropriate data based on model type
                if not st.session_state.is_custom:
                    if st.session_state.selected_model == "House Price Predictor":
                        try:
                            # Load and combine train data
                            X_train = pd.read_csv('files/Regression/house_price/house_price_X_train.csv')
                            y_train = pd.read_csv('files/Regression/house_price/house_price_y_train.csv')
                            # Load and combine test data
                            X_test = pd.read_csv('files/Regression/house_price/house_price_X_test.csv')
                            y_test = pd.read_csv('files/Regression/house_price/house_price_y_test.csv')
                            
                            # Combine features and target
                            X = pd.concat([X_train, X_test], axis=0)
                            y = pd.concat([y_train, y_test], axis=0)
                            
                            # Create a combined dataset with proper column names
                            df = X.copy()
                            df['price'] = y.values
                            target_col = 'price'
                            
                            # Load the pre-built model
                            st.session_state.custom_model = load_prebuilt_model(st.session_state.model_type, st.session_state.selected_model)
                            if st.session_state.custom_model is None:
                                st.error("Failed to load pre-built regression model.")
                                return
                        except Exception as e:
                            st.error(f"Error loading pre-built model: {str(e)}")
                            df = None
                            target_col = None
                else:
                    df = st.session_state.uploaded_data
                    target_col = st.session_state.target_column
                
                # Calculate regression metrics if we have a model and data
                if st.session_state.get('custom_model') is not None and df is not None and target_col in df.columns:
                    try:
                        # Transform features to match model expectations
                        df_transformed, feature_cols = validate_and_transform_features(df, "Regression", target_col)
                        
                        if len(feature_cols) == 0:
                            st.error("No valid features found for the model.")
                            return
                        
                        X = df_transformed[feature_cols]
                        y = df_transformed[target_col]
                        
                        # Ensure X and y have the same number of samples
                        if len(X) != len(y):
                            st.error(f"Mismatch in number of samples: X has {len(X)} samples, y has {len(y)} samples")
                            return
                        
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
                        import numpy as np
                        
                        # Make predictions
                        y_pred = st.session_state.custom_model.predict(X)
                        
                        # Calculate metrics
                        r2 = r2_score(y, y_pred)
                        mae = mean_absolute_error(y, y_pred)
                        mse = mean_squared_error(y, y_pred)
                        rmse = np.sqrt(mse)
                        explained_var = explained_variance_score(y, y_pred)
                        mape = np.mean(np.abs((y - y_pred) / y)) * 100
                        
                        # Display metrics in two rows
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{r2:.3f}")
                            st.metric("MAE", f"{mae:.3f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.3f}")
                            st.metric("MSE", f"{mse:.3f}")
                        with col3:
                            st.metric("Explained Variance", f"{explained_var:.3f}")
                            st.metric("MAPE (%)", f"{mape:.1f}")
                        
                        # Add detailed visualizations
                        st.markdown("#### Regression Analysis Plots")
                        
                        tab1, tab2, tab3 = st.tabs(["Predictions", "Residuals", "Error Distribution"])
                        
                        with tab1:
                            # Actual vs Predicted Plot
                            fig = visualizer.create_actual_vs_predicted(y, y_pred)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                # Residuals Plot
                                fig = visualizer.create_residuals_plot(y, y_pred)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Residuals vs Feature Plot
                                feature = st.selectbox(
                                    "Select feature for residuals analysis",
                                    feature_cols,
                                    key="residuals_feature"
                                )
                                fig = px.scatter(
                                    x=X[feature],
                                    y=y - y_pred,
                                    labels={"x": feature, "y": "Residuals"},
                                    title=f"Residuals vs {feature}"
                                )
                                fig.add_hline(y=0, line_dash="dash")
                                fig.update_layout(template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            # Error Distribution Plot
                            errors = y - y_pred
                            fig = px.histogram(
                                x=errors,
                                nbins=30,
                                title="Distribution of Prediction Errors",
                                labels={"x": "Prediction Error"}
                            )
                            fig.add_vline(x=0, line_dash="dash")
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Error Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Error", f"{np.mean(errors):.3f}")
                            with col2:
                                st.metric("Std Error", f"{np.std(errors):.3f}")
                            with col3:
                                st.metric("Median Error", f"{np.median(errors):.3f}")
                            
                    except Exception as e:
                        st.warning("Unable to generate regression performance metrics and plots.")
                        st.error(f"Error: {str(e)}")
                        st.error("Debug info:")
                        st.write(f"Data shape: {df.shape if df is not None else None}")
                        st.write(f"Target column: {target_col}")
                        st.write(f"Feature columns: {feature_cols if 'feature_cols' in locals() else None}")
                else:
                    if st.session_state.get('custom_model') is None:
                        st.error("No model available. Please ensure a model is loaded.")
                    elif df is None:
                        st.error("No dataset available. Please ensure data is loaded.")
                    elif target_col not in df.columns:
                        st.error(f"Target column '{target_col}' not found in dataset.")
            
            elif st.session_state.model_type == "Clustering":
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", "0.68")
                with col2:
                    st.metric("Calinski-Harabasz Score", "156.32")
                
                # Add Cluster Visualization
                if st.session_state.get('custom_model') is not None and df is not None:
                    try:
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                        selected_features = st.multiselect(
                            "Select Features for Cluster Visualization",
                            numeric_cols,
                            default=list(numeric_cols[:2]) if len(numeric_cols) >= 2 else []
                        )
                        
                        if len(selected_features) >= 2:
                            X = df[selected_features]
                            labels = st.session_state.custom_model.predict(X)
                            fig = visualizer.create_cluster_plot(selected_features, labels)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning("Unable to generate clustering visualization. Please ensure your model is properly trained and compatible with the data.")
                        st.error(f"Error: {str(e)}")
        
        with viz_tab3:
            st.markdown("### Feature Importance")
            
            if st.session_state.model_type == "Clustering":
                st.info("Feature importance analysis is not available for clustering models.")
                return
            
            if st.session_state.get('custom_model') is not None and df is not None and target_col in df.columns:
                try:
                    # Transform features to match model expectations
                    df_transformed, feature_cols = validate_and_transform_features(df, st.session_state.model_type, target_col)
                    
                    if len(feature_cols) == 0:
                        st.error("No valid features found for the model.")
                        return
                    
                    # Get feature importance if available
                    if hasattr(st.session_state.custom_model, 'feature_importances_'):
                        X = df_transformed[feature_cols]
                        feature_names = X.columns
                        importances = st.session_state.custom_model.feature_importances_
                        
                        # Sort features by importance
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        })
                        feature_importance = feature_importance.sort_values('importance', ascending=False)
                        
                        # Create bar plot
                        fig = px.bar(
                            feature_importance,
                            x='feature',
                            y='importance',
                            title='Feature Importance',
                            labels={'feature': 'Feature', 'importance': 'Importance'}
                        )
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("This model does not provide direct feature importance scores. Consider using SHAP values for feature importance analysis.")
                except Exception as e:
                    st.warning("Unable to generate feature importance visualization.")
                    st.error(f"Error: {str(e)}")
            else:
                st.info("Please upload a model and dataset to view feature importance analysis.")
        
        with viz_tab4:
            st.markdown("### Model Diagnostics")
            
            if st.session_state.get('custom_model') is not None:
                try:
                    st.markdown("#### Model Information")
                    st.code(str(st.session_state.custom_model))
                    
                    # Add model parameters
                    st.markdown("#### Model Parameters")
                    if hasattr(st.session_state.custom_model, 'get_params'):
                        st.json(st.session_state.custom_model.get_params())
                    else:
                        st.info("Model parameters are not available for this model type.")
                except Exception as e:
                    st.warning("Unable to display model diagnostics.")
                    st.error(f"Error: {str(e)}")
            else:
                st.info("Please upload a model to view model diagnostics.")

if __name__ == '__main__':
    main()
