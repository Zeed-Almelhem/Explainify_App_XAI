import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create folder for football results dataset
folder_name = "Football_Results"
os.makedirs(folder_name, exist_ok=True)

# Load the dataset
football_results_df = pd.read_csv("international_football_results_regression_dataset.csv")

# Handle categorical columns and date column
football_results_df['date'] = pd.to_datetime(football_results_df['date'])
football_results_df['year'] = football_results_df['date'].dt.year
football_results_df['month'] = football_results_df['date'].dt.month
football_results_df['day'] = football_results_df['date'].dt.day

# Encode categorical columns (home_team, away_team, tournament, country, neutral)
football_results_df = pd.get_dummies(football_results_df, columns=['home_team', 'away_team', 'tournament', 'country'], drop_first=True)

# Create a new target variable: score difference (home_score - away_score)
football_results_df['score_diff'] = football_results_df['home_score'] - football_results_df['away_score']

# Drop original columns that are not needed for regression
football_results_df = football_results_df.drop(columns=['date', 'home_score', 'away_score', 'neutral'])

# Split the data into training and testing sets
X = football_results_df.drop(columns=['score_diff'])
y = football_results_df['score_diff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Football Results Model - MSE: {mse}, R2: {r2}")

# Save the model and datasets to the folder
joblib.dump(model, os.path.join(folder_name, "football_results_model.pkl"))
X_train.to_csv(os.path.join(folder_name, "football_results_X_train.csv"), index=False)
X_test.to_csv(os.path.join(folder_name, "football_results_X_test.csv"), index=False)
y_train.to_csv(os.path.join(folder_name, "football_results_y_train.csv"), index=False)
y_test.to_csv(os.path.join(folder_name, "football_results_y_test.csv"), index=False)
