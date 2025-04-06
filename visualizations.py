import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import shap
import lime
import lime.lime_tabular
from yellowbrick.classifier import ROCAUC
from yellowbrick.regressor import ResidualsPlot
import pandas as pd

class ModelVisualizer:
    def __init__(self, data, model=None, target_col=None, model_type=None):
        self.data = data
        self.model = model
        self.target_col = target_col
        self.model_type = model_type
        
    def create_distribution_plot(self, feature):
        """Create distribution plot for a feature"""
        fig = px.histogram(self.data, x=feature, title=f"Distribution of {feature}")
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_box_plot(self, feature):
        """Create box plot for a feature"""
        fig = px.box(self.data, y=feature, title=f"Box Plot of {feature}")
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_violin_plot(self, feature):
        """Create violin plot for a feature"""
        fig = px.violin(self.data, y=feature, title=f"Violin Plot of {feature}")
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_correlation_matrix(self, features=None):
        """Create correlation matrix heatmap"""
        if features is None:
            features = self.data.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = self.data[features].corr()
        fig = px.imshow(corr_matrix,
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu')
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_scatter_plot(self, x_feature, y_feature):
        """Create scatter plot between two features"""
        fig = px.scatter(self.data, x=x_feature, y=y_feature,
                        title=f"{x_feature} vs {y_feature}")
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_missing_values_heatmap(self):
        """Create missing values heatmap"""
        missing = self.data.isnull()
        fig = px.imshow(missing,
                       title="Missing Values Heatmap",
                       color_continuous_scale=['blue', 'red'])
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_pair_plot(self, features):
        """Create pair plot for selected features"""
        fig = px.scatter_matrix(self.data[features],
                              title="Pair Plot")
        fig.update_layout(template="plotly_dark")
        return fig
    
    # Classification specific visualizations
    def create_roc_curve(self, y_true, y_pred_proba):
        """Create ROC curve for classification models"""
        if self.model_type != "Classification":
            return None
            
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random',
                                line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                name='Model'))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template="plotly_dark"
        )
        return fig
    
    def create_confusion_matrix(self, y_true, y_pred):
        """Create confusion matrix heatmap"""
        if self.model_type != "Classification":
            return None
            
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm,
                       title="Confusion Matrix",
                       labels=dict(x="Predicted", y="Actual"),
                       color_continuous_scale='Viridis')
        fig.update_layout(template="plotly_dark")
        return fig
    
    # Regression specific visualizations
    def create_residuals_plot(self, y_true, y_pred):
        """Create residuals plot for regression models"""
        if self.model_type != "Regression":
            return None
            
        residuals = y_true - y_pred
        fig = px.scatter(x=y_pred, y=residuals,
                        title="Residuals Plot",
                        labels={"x": "Predicted Values", "y": "Residuals"})
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_actual_vs_predicted(self, y_true, y_pred):
        """Create actual vs predicted plot for regression"""
        if self.model_type != "Regression":
            return None
            
        fig = px.scatter(x=y_true, y=y_pred,
                        title="Actual vs Predicted",
                        labels={"x": "Actual Values", "y": "Predicted Values"})
        fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)],
                                y=[min(y_true), max(y_true)],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(dash='dash')))
        fig.update_layout(template="plotly_dark")
        return fig
    
    # Clustering specific visualizations
    def create_cluster_plot(self, features, labels):
        """Create cluster visualization"""
        if self.model_type != "Clustering":
            return None
            
        if len(features) == 2:
            fig = px.scatter(self.data, x=features[0], y=features[1],
                           color=labels,
                           title="Cluster Visualization")
        elif len(features) == 3:
            fig = px.scatter_3d(self.data, x=features[0], y=features[1], z=features[2],
                              color=labels,
                              title="3D Cluster Visualization")
        else:
            # Use PCA for higher dimensions
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(self.data[features])
            fig = px.scatter(x=coords[:, 0], y=coords[:, 1],
                           color=labels,
                           title="Cluster Visualization (PCA)")
        
        fig.update_layout(template="plotly_dark")
        return fig
    
    # Feature importance visualizations
    def create_feature_importance_plot(self, importance_scores, feature_names):
        """Create feature importance bar plot"""
        fig = px.bar(x=feature_names, y=importance_scores,
                    title="Feature Importance",
                    labels={"x": "Features", "y": "Importance"})
        fig.update_layout(template="plotly_dark")
        return fig
    
    def create_shap_summary_plot(self):
        """Create SHAP summary plot"""
        if self.model is None:
            return None
            
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.data)
        
        # Convert SHAP plot to plotly
        plt.figure()
        shap.summary_plot(shap_values, self.data, show=False)
        fig = plt.gcf()
        plt.close()
        
        return fig
