# Explainify: Explainable AI Model Analysis Platform

## Description
Explainify is a powerful Explainable AI (XAI) platform designed to demystify machine learning models through comprehensive visualizations and interactive analysis tools. By bridging the gap between complex ML models and human understanding, Explainify helps data scientists, analysts, and stakeholders gain deeper insights into their models' behavior and decision-making processes.

The platform supports a wide range of ML frameworks including scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM, and CatBoost, making it versatile for different model architectures and use cases. Whether you're working with classification, regression, or clustering models, Explainify provides the tools to understand your model's inner workings and validate its performance.

### Key Benefits
- **Model Transparency**: Understand how your models make decisions
- **Interactive Analysis**: Explore model behavior through dynamic visualizations
- **Framework Agnostic**: Support for major ML libraries and custom models
- **Automated Insights**: Intelligent feature analysis and model diagnostics
- **No-Code Exploration**: User-friendly interface for technical and non-technical users

## Features

### Pre-built Models
- **Binary Customer Churn Classifier**: Analyze customer churn prediction
- **House Price Predictor**: Explore house price predictions
- **Customer Segmentation**: Understand customer segments based on wholesale data
- More models coming soon!

### Analysis Capabilities
- **Data Exploration**:
  - Dataset overview and statistics
  - Missing values analysis
  - Data type information
  - Distribution plots
  - Correlation analysis
  - Feature relationships

- **Data Analysis**:
  - Target distribution visualization
  - Feature analysis by target
  - Numerical and categorical feature insights
  - Interactive visualizations

### Model Types & Framework Support
1. **Classification Models**:
   - Binary and multiclass classification
   - Supported Frameworks:
     - scikit-learn (RandomForest, SVM, etc.)
     - TensorFlow/Keras
     - PyTorch
     - XGBoost
     - LightGBM
     - CatBoost

2. **Regression Models**:
   - Linear and non-linear regression
   - Supported Frameworks:
     - scikit-learn
     - TensorFlow/Keras
     - PyTorch
     - XGBoost
     - LightGBM

3. **Clustering Models**:
   - K-means, hierarchical clustering
   - Density-based clustering
   - Supported Frameworks:
     - scikit-learn
     - Custom clustering implementations

### XAI Capabilities
- **Feature Importance Analysis**:
  - Global feature importance
  - Local feature importance (SHAP, LIME support coming soon)
  - Feature interaction analysis

- **Model Behavior Analysis**:
  - Decision boundary visualization
  - Prediction confidence analysis
  - Model comparison tools

### Custom Model Support
- Upload your own dataset (CSV format)
- Import pre-trained models (pickle/joblib format)
- Automatic feature type detection
- Interactive analysis tools
- Support for custom model implementations

## Technical Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Model Support**: 
  - scikit-learn
  - TensorFlow (coming soon)
  - PyTorch (coming soon)
  - XGBoost (coming soon)
  - LightGBM (coming soon)
  - CatBoost (coming soon)

## Dependencies
streamlit>=1.31.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.4.0
joblib>=1.3.0

## Project Structure
- `app.py`: Main application with UI implementation
- `visualizations.py`: Visualization components and utilities
- `files/`: Pre-built model files and datasets
  - `Classification/`: Classification model files
  - `Regression/`: Regression model files
  - `Clustering/`: Clustering model files

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Using the application:
   - Select a model type (Classification/Regression/Clustering)
   - Choose between pre-built models or upload your own
   - Click "Start Model Exploration" to begin analysis
   - Navigate through Data Exploration and Analysis tabs

## Contributing
Feel free to contribute to this project by:
- Adding new pre-built models
- Enhancing visualization capabilities
- Improving feature detection and analysis
- Adding support for more model types and frameworks
- Implementing additional XAI techniques

## Coming Soon
- Explainability for BERT models built with Google Pytorch tools.


## License
MIT License


