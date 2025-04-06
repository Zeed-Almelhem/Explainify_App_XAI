import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create folder for house price dataset
folder_name = "House_Price"
os.makedirs(folder_name, exist_ok=True)

# Load the dataset
house_price_df = pd.read_csv("house_price_regression_dataset.csv")

# Handle categorical columns
house_price_df['mainroad'] = house_price_df['mainroad'].map({'yes': 1, 'no': 0})
house_price_df['guestroom'] = house_price_df['guestroom'].map({'yes': 1, 'no': 0})
house_price_df['basement'] = house_price_df['basement'].map({'yes': 1, 'no': 0})
house_price_df['hotwaterheating'] = house_price_df['hotwaterheating'].map({'yes': 1, 'no': 0})
house_price_df['airconditioning'] = house_price_df['airconditioning'].map({'yes': 1, 'no': 0})
house_price_df['prefarea'] = house_price_df['prefarea'].map({'yes': 1, 'no': 0})
house_price_df['furnishingstatus'] = house_price_df['furnishingstatus'].map({'furnished': 1, 'unfurnished': 0, 'semi-furnished': 2})

# Split the data into training and testing sets
X = house_price_df.drop(columns=['price'])
y = house_price_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"House Price Model - MSE: {mse}, R2: {r2}")

# Save the model and datasets to the folder
joblib.dump(model, os.path.join(folder_name, "house_price_model.pkl"))
X_train.to_csv(os.path.join(folder_name, "house_price_X_train.csv"), index=False)
X_test.to_csv(os.path.join(folder_name, "house_price_X_test.csv"), index=False)
y_train.to_csv(os.path.join(folder_name, "house_price_y_train.csv"), index=False)
y_test.to_csv(os.path.join(folder_name, "house_price_y_test.csv"), index=False)