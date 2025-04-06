#Description of what each explain component is 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

predictions = []

class models:
    def __init__(self, name):
        self.name = name
        # Define which models are coming soon
        self.coming_soon_models = [
            'Brazilian E-Commerce Dataset',
            'No-show Flight Prediction Dataset',
            'International Football Results',
            'Air Quality Dataset',
            '20 Newsgroups Classification'
        ]
        
        # Map model names to their respective paths
        model_paths = {
            # Classification models
            'Binary Customer Churn Classifier': {
                'data': 'files/Classification/customer_churn_X_train.csv',
                'model': 'files/Classification/customer_churn_model.pkl'
            },
            
            # Regression models
            'House Price Predictor': {
                'data': 'files/Regression/house_price/house_price_X_train.csv',
                'model': 'files/Regression/house_price/house_price_model.pkl'
            },
            
            # Clustering models
            'Customer Segmentation (Wholesale)': {
                'data': 'files/Clustering/customer_segmentation_X_train.csv',
                'model': 'files/Clustering/customer_segmentation_model.pkl'
            }
        }
        
        # Set paths based on selected model
        if self.name in self.coming_soon_models:
            self.data_file = None
            self.model_file = None
            self.is_coming_soon = True
        elif self.name in model_paths:
            self.data_file = model_paths[self.name]['data']
            self.model_file = model_paths[self.name]['model']
            self.is_coming_soon = False
        else:
            # Fallback for custom models
            self.data_file = None
            self.model_file = None
            self.is_coming_soon = False

        self.train_data = self.get_train_data()
        self.encoders = self.create_encoders()
        self.encoded_data = self.encode_data(self.train_data)
        self.model = self.create_fallback_model()  # Always use fallback model for demo
        self.features_file = 'files/feature_discriptions.csv'
        
        self.feature_mapping = self.create_feature_mapping(self.train_data.columns)
        self.feature_importance_dict = self.create_feature_importance_dict()
        self.unique_x_train_values = self.get_unique_values_as_dict()
        self.numeric_columns, self.categorical_columns, self.string_columns = self.data_types()
        self.predictions = []

    def get_train_data(self):
        try:
            x_train = pd.read_csv(self.data_file)
            return x_train
        except FileNotFoundError:
            st.error(f"Training data file not found: {self.data_file}")
            # Create dummy data for demonstration
            return pd.DataFrame({
                'numeric_feature': np.random.randn(100),
                'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
                'binary_feature': np.random.choice([0, 1], 100),
                'text_feature': ['text_' + str(i) for i in range(100)]
            })

    def create_encoders(self):
        """Create label encoders for categorical columns"""
        encoders = {}
        for column in self.train_data.columns:
            if self.train_data[column].dtype == 'object':
                le = LabelEncoder()
                le.fit(self.train_data[column].astype(str))
                encoders[column] = le
        return encoders

    def encode_data(self, data):
        """Encode categorical data for model training/prediction"""
        encoded = data.copy()
        for column, encoder in self.encoders.items():
            encoded[column] = encoder.transform(data[column].astype(str))
        return encoded

    def create_fallback_model(self):
        """Create a simple RandomForest model for demonstration"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = self.encoded_data
        y = np.random.randint(0, 2, size=len(X))  # Random binary labels for demo
        model.fit(X, y)
        return model

    def get_unique_values_as_dict(self):
        unique_values_dict = {}
        for column in self.train_data.columns:
            unique_values_dict[column] = self.train_data[column].unique().tolist()
        return unique_values_dict

    def data_types(self):
        numeric_columns = []
        string_columns = []
        categorical_columns = []

        for column in self.train_data.columns:
            if self.train_data[column].dtype in ['int64', 'float64']:
                numeric_columns.append(column)
                self.unique_x_train_values[column] = [int(x) if not math.isnan(x) else float('nan') for x in self.unique_x_train_values[column]]
            elif self.train_data[column].dtype == 'object':
                categorical_columns.append(column)
            else:
                string_columns.append(column)
        return numeric_columns, categorical_columns, string_columns

    def create_feature_importance_dict(self):
        """Create feature importance for model"""
        feature_importance = self.model.feature_importances_
        feature_importance_dict = {col: imp for col, imp in zip(self.train_data.columns, feature_importance)}
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features

    def create_feature_mapping(self, table):
        feature_names = table
        feature_mapping = {}
        for feature in feature_names:
            feature_mapping[feature] = feature.replace('_', ' ').title()
        return feature_mapping

    def create_avg_df(self):
        # Calculate mean for numeric columns
        new_data = self.train_data.iloc[0, :].copy()
        for feature in self.train_data.columns:
            if feature in self.numeric_columns:
                new_data[feature] = self.train_data[feature].median()
            else:
                new_data[feature] = self.train_data[feature].mode().iloc[0]
        new_data = pd.DataFrame([new_data])
        return new_data

    def playground_predict(self, special_features):
        X = self.create_avg_df()
        for feature in special_features.keys():
            X[feature] = special_features[feature]
        X_encoded = self.encode_data(X)
        y_predict_proba = float(self.model.predict_proba(X_encoded)[:, 1][0])
        return y_predict_proba

    def investigate_predict(self, uuid):
        """Return a random prediction for demo purposes"""
        return np.random.random()

    def plot_feature_importance(self, n):
        self.top_x_features_importance = self.feature_importance_dict[:n][::-1]
        self.top_x_features = [(self.feature_mapping[feature[0]], feature[1]) for feature in self.top_x_features_importance]
        self.top_x_features = pd.DataFrame(self.top_x_features).rename({0: 'Feature', 1: 'Importance'}, axis=1)

        # Create a horizontal bar chart with white bars and white axes
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.barh(self.top_x_features['Feature'], self.top_x_features['Importance'], color='white')

        # Customize the appearance of the chart
        ax.set_xlabel('Importance', fontsize=7, color='white')
        ax.set_ylabel('Feature', fontsize=7, color='white')
        ax.xaxis.set_tick_params(labelsize=7, colors='white')
        ax.yaxis.set_tick_params(labelsize=7, colors='white')

        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        st.pyplot(fig)

    def feature_descriptions(self, tab):
        try:
            self.feature_descriptions = pd.read_csv(self.features_file).rename({'feature': 'Feature', 'description': 'Description'}, axis=1)
        except FileNotFoundError:
            # Create dummy feature descriptions
            self.feature_descriptions = pd.DataFrame({
                'Feature': self.train_data.columns,
                'Description': [f'Description for {col}' for col in self.train_data.columns]
            })

        if tab == 'features':
            feature_mapping_table = pd.DataFrame(self.top_x_features_importance).rename({0: 'Feature', 1: 'Importance'}, axis=1)
            self.feature_descriptions = pd.merge(self.feature_descriptions, feature_mapping_table, left_on='Feature', right_on='Feature', how='inner')
            self.feature_descriptions['Feature'] = self.feature_descriptions['Feature'].map(self.feature_mapping)
            st.table(self.feature_descriptions[['Feature','Description','Importance']].sort_values(by='Importance', ascending=False).set_index('Feature'))
        elif tab == 'playground':
            combined_data = {}
            for index, row in self.feature_descriptions.iterrows():
                feature_name = row['Feature']
                if feature_name in self.feature_mapping:
                    fixed_feature_name = self.feature_mapping[feature_name]
                    description = row['Description']
                    combined_data[feature_name] = (fixed_feature_name, description)
            self.feature_mapping = combined_data
