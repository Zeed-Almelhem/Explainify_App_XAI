import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Create folder for customer churn dataset
folder_name = "Customer_Churn"
os.makedirs(folder_name, exist_ok=True)

# Load the dataset (replace with the actual file path)
churn_df = pd.read_csv("customer_churn_classification_dataset.csv")

# Preprocess the data (Assuming 'Churn' is the target column)
# Handle categorical columns by encoding them to numeric
churn_df['gender'] = churn_df['gender'].map({'Male': 1, 'Female': 0})
churn_df['Partner'] = churn_df['Partner'].map({'Yes': 1, 'No': 0})
churn_df['Dependents'] = churn_df['Dependents'].map({'Yes': 1, 'No': 0})
churn_df['PhoneService'] = churn_df['PhoneService'].map({'Yes': 1, 'No': 0})
churn_df['MultipleLines'] = churn_df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
churn_df['InternetService'] = churn_df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
churn_df['OnlineSecurity'] = churn_df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
churn_df['OnlineBackup'] = churn_df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
churn_df['DeviceProtection'] = churn_df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
churn_df['TechSupport'] = churn_df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
churn_df['StreamingTV'] = churn_df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
churn_df['StreamingMovies'] = churn_df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
churn_df['Contract'] = churn_df['Contract'].map({'Month-to-month': 1, 'One year': 2, 'Two year': 3})
churn_df['PaperlessBilling'] = churn_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
churn_df['PaymentMethod'] = churn_df['PaymentMethod'].map({'Electronic check': 1, 'Mailed check': 2, 'Bank transfer (automatic)': 3, 'Credit card (automatic)': 4})

# Handle missing values for numeric columns only
numeric_columns = churn_df.select_dtypes(include=[np.number]).columns
churn_df[numeric_columns] = churn_df[numeric_columns].fillna(churn_df[numeric_columns].mean())

# Handle missing values for categorical columns (using mode)
categorical_columns = churn_df.select_dtypes(exclude=[np.number]).columns
for column in categorical_columns:
    churn_df[column] = churn_df[column].fillna(churn_df[column].mode()[0])

# Convert 'TotalCharges' to numeric, handling any non-numeric entries (like empty strings)
churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce')

# Handle any remaining missing values after converting 'TotalCharges' (if any)
churn_df['TotalCharges'] = churn_df['TotalCharges'].fillna(churn_df['TotalCharges'].mean())

# Drop unnecessary columns (e.g., 'customerID')
churn_df = churn_df.drop(columns=['customerID'])

# Target column 'Churn' (binary: 0 = No, 1 = Yes)
X = churn_df.drop(columns=['Churn'])
y = churn_df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f"Customer Churn Model - Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save the model and datasets to the folder
joblib.dump(model, os.path.join(folder_name, "customer_churn_model.pkl"))
X_train.to_csv(os.path.join(folder_name, "customer_churn_X_train.csv"), index=False)
X_test.to_csv(os.path.join(folder_name, "customer_churn_X_test.csv"), index=False)
y_train.to_csv(os.path.join(folder_name, "customer_churn_y_train.csv"), index=False)
y_test.to_csv(os.path.join(folder_name, "customer_churn_y_test.csv"), index=False)

print("Customer churn model and data saved successfully.")