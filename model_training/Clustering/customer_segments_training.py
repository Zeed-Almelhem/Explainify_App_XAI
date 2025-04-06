import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt

# Create folder for customer segmentation dataset
folder_name = "Customer_Segmentation"
os.makedirs(folder_name, exist_ok=True)

# Load the dataset (replace with the actual file path)
segmentation_df = pd.read_csv("wholesale_customers_clustering_dataset.csv.csv")

# Drop any unnecessary columns (e.g., customer ID)
# In this case, it seems there is no 'CustomerID', so we won't drop anything specific
# But if there is any such column, we would drop it like this:
# segmentation_df = segmentation_df.drop(columns=['CustomerID'])

# Feature set X (no target variable since clustering is unsupervised)
X = segmentation_df

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (useful for visualization/interpretation)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Optional: Use the Elbow Method to determine the optimal number of clusters
# Calculate distortions (inertia) for a range of cluster numbers
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (for example, 3 clusters)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

# Train the KMeans clustering model
kmeans.fit(X_train)

# Predict the cluster labels for the test set
y_pred = kmeans.predict(X_test)

# Save the model and datasets to the folder
joblib.dump(kmeans, os.path.join(folder_name, "customer_segmentation_model.pkl"))
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_train_df.to_csv(os.path.join(folder_name, "customer_segmentation_X_train.csv"), index=False)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df.to_csv(os.path.join(folder_name, "customer_segmentation_X_test.csv"), index=False)

# Optional: Save the predicted labels to the original dataset
segmentation_df['cluster'] = kmeans.predict(X_scaled)
segmentation_df.to_csv(os.path.join(folder_name, "customer_segmentation_labels.csv"), index=False)

print("Customer segmentation model and data saved successfully.")

