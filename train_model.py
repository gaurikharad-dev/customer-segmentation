
import pandas as pd

# Load the dataset
df = pd.read_excel("Online Retail.xlsx")

# Display the original dataset information
print("Original dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

# Remove rows with missing customer IDs
df = df.dropna(subset=["CustomerID"])

# Remove duplicate records
df = df.drop_duplicates()

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep only positive quantities and prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Convert invoice dates to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate total amount for each transaction
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Display cleaned dataset information
print("\nCleaned dataset shape:", df.shape)
print("\nCleaned data preview:")
print(df.head())

# Save the cleaned dataset
df.to_csv("cleaned_retail_data.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned data saved to cleaned_retail_data.csv")

# Step 9: Create RFM features

# Set the reference date to one day after the last transaction
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Calculate RFM values for each customer
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum")
)

# Display customer-level RFM data
print("\nRFM dataset preview:")
print(rfm.head())

print("\nTotal number of customers:", len(rfm))

# Save the RFM dataset
rfm.to_csv("customer_rfm.csv")

print("\nRFM data saved to customer_rfm.csv")

# Step 10: Train the K-Means clustering model

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Select RFM features
features = rfm[["Recency", "Frequency", "Monetary"]]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train the K-Means model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(scaled_features)

# Evaluate the model
score = silhouette_score(scaled_features, rfm["Cluster"])

print("\nK-Means Model Training Completed!")
print("Number of clusters:", rfm["Cluster"].nunique())
print("Silhouette Score:", round(score, 3))

# Display customer segments
print("\nCustomer segments:")
print(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2))

# Save the model and scaler
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save customer data with cluster labels
rfm.to_csv("customer_rfm.csv")

print("\nModel saved as kmeans_model.pkl")
print("Scaler saved as scaler.pkl")
print("Cluster labels saved in customer_rfm.csv")

