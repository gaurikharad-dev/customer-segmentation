
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI-Based Customer Segmentation")
st.write("Analyze customer purchasing behavior using K-Means clustering.")

# Load saved data and model
@st.cache_data
def load_data():
    return pd.read_csv("customer_rfm.csv")

@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

rfm = load_data()
kmeans, scaler = load_model()

# Remove the CSV index column if present
rfm = rfm.loc[:, ~rfm.columns.str.startswith("Unnamed")]

# Summary metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(rfm):,}")
col2.metric("Customer Segments", rfm["Cluster"].nunique())
col3.metric("Total Customer Revenue", f"£{rfm['Monetary'].sum():,.2f}")

st.divider()

# Customer segment distribution
st.subheader("Customer Segment Distribution")

cluster_counts = rfm["Cluster"].value_counts().sort_index()

fig, ax = plt.subplots()
cluster_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Customers")
ax.set_title("Customers in Each Segment")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

# RFM profile
st.subheader("Customer Segment Profiles")

profile = rfm.groupby("Cluster")[
    ["Recency", "Frequency", "Monetary"]
].mean().round(2)

st.dataframe(profile, use_container_width=True)

# Monetary comparison
st.subheader("Average Spending by Segment")

fig, ax = plt.subplots()
profile["Monetary"].plot(kind="bar", ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Average Spending (£)")
ax.set_title("Average Customer Spending by Segment")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

# View customer records
st.subheader("Explore Customer Data")

selected_cluster = st.selectbox(
    "Select a customer segment",
    ["All Segments"] + sorted(rfm["Cluster"].unique().tolist())
)

if selected_cluster == "All Segments":
    filtered_data = rfm
else:
    filtered_data = rfm[rfm["Cluster"] == selected_cluster]

st.dataframe(filtered_data, use_container_width=True)

# Predict segment for a new customer
st.divider()
st.subheader("Predict a New Customer's Segment")

with st.form("customer_form"):
    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency (number of invoices)",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Monetary (total spending in £)",
        min_value=0.01,
        value=500.0
    )

    submitted = st.form_submit_button("Predict Segment")

if submitted:
    customer = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )

    scaled_customer = scaler.transform(customer)
    predicted_cluster = kmeans.predict(scaled_customer)[0]

    st.success(f"This customer belongs to Cluster {predicted_cluster}.")
