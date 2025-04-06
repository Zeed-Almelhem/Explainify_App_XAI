import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create folder for air quality dataset
folder_name = "Air_Quality"
os.makedirs(folder_name, exist_ok=True)

# Load the dataset
air_quality_df = pd.read_csv("air_quality_regression_datatset.csv", delimiter=';')

# Handle missing values and convert data types
air_quality_df = air_quality_df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
air_quality_df = air_quality_df.dropna()

# Split the data into training and testing sets
X = air_quality_df.drop(columns=['CO(GT)'])
y = air_quality_df['CO(GT)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Air Quality Model - MSE: {mse}, R2: {r2}")

# Save the model and datasets to the folder
joblib.dump(model, os.path.join(folder_name, "air_quality_model.pkl"))
X_train.to_csv(os.path.join(folder_name, "air_quality_X_train.csv"), index=False)
X_test.to_csv(os.path.join(folder_name, "air_quality_X_test.csv"), index=False)
y_train.to_csv(os.path.join(folder_name, "air_quality_y_train.csv"), index=False)
y_test.to_csv(os.path.join(folder_name, "air_quality_y_test.csv"), index=False)
