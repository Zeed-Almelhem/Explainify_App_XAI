# Web Pro Identification ML Model App

## Description
This application provides an interactive interface for exploring and understanding machine learning models used to identify web professionals. The system includes two models:
- 0 days model: Identifies web professionals within 12 hours of signup
- 7 days model: Identifies web professionals within 7 days of signup

## Features
- **Main Dashboard**: Overview of the models and their purposes
- **Features Analysis**: Visualize and understand feature importance for each model
- **Interactive Playground**: Test different feature combinations and see their impact on predictions
- **User Investigation**: Look up specific users by UUID to see their web pro probability scores
- **Enhanced Visualizations**:
  - **Data Analysis**: Distribution plots, box plots, violin plots, correlation matrices, scatter plots, missing values heatmaps, and pair plots
  - **Model Performance**: ROC curves, confusion matrices, residual plots, actual vs predicted plots, and clustering visualizations
  - **Feature Importance**: Feature importance bar plots and SHAP value analysis
  - **Model Diagnostics**: Model architecture visualization and parameter inspection

## Technical Stack
- **Frontend**: Streamlit
- **ML Framework**: Custom model implementation with feature importance analysis
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Model Interpretability**: SHAP, LIME
- **Model Evaluation**: Yellowbrick
- **Database Connection**: Wix Trino Client

## Dependencies
- wix-trino-client
- streamlit
- streamlit_option_menu
- pickle
- pandas
- matplotlib
- numpy
- plotly
- seaborn
- shap
- lime
- yellowbrick

## Project Structure
- `app.py`: Main application entry point with Streamlit UI implementation
- `model_class.py`: Core model implementation with feature importance and prediction logic
- `design.py`: UI design elements and text content
- `files/`: Contains model files, training data, and feature descriptions

## Usage
1. Run the application using Streamlit:
```bash
streamlit run app.py
```

2. Navigate through different sections:
   - Use the sidebar to switch between Main, Features, Playground, and User Investigation
   - Explore feature importance visualizations
   - Test predictions with different feature combinations
   - Look up specific users using their UUID

## Model Information
The models analyze various user behaviors and characteristics to determine the likelihood of them being web professionals. Key features include user activity patterns, engagement metrics, and account characteristics.

## Data Privacy
The application connects to production data through Wix Trino Client. Ensure proper authentication and authorization when accessing user data.
