import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Clustering + Regression", layout="wide")
st.title("Customer Clustering & Prediksi Belanja")

# load data hasil colab
rfm = pd.read_csv("rfm_customer.csv")

# ===== retrain ringan (dataset kecil, aman) =====
features = ["Recency", "Frequency", "Monetary", "AvgQuantity", "UniqueItems"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

kmeans = KMeans(n_clusters=rfm["Cluster"].nunique(), random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

reg_model = RandomForestRegressor(n_estimators=300, random_state=42)
reg_model.fit(
    rfm[["Recency","Frequency","AvgQuantity","UniqueItems","Cluster"]],
    rfm["Monetary"]
)

# ===== INPUT =====
st.subheader("Input Manual Customer")

recency = st.number_input("Recency (hari)", 0, 1000, 30)
frequency = st.number_input("Frequency", 0, 500, 5)
avg_qty = st.number_input("Avg Quantity", 0.0, 1000.0, 10.0)
unique_items = st.number_input("Unique Items", 0, 500, 20)
monetary = st.number_input("Monetary (dummy utk clustering)", 0.0, 1e7, 200.0)

X_input = pd.DataFrame([{
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary,
    "AvgQuantity": avg_qty,
    "UniqueItems": unique_items
}])

cluster = kmeans.predict(scaler.transform(X_input))[0]

X_reg = pd.DataFrame([{
    "Recency": recency,
    "Frequency": frequency,
    "AvgQuantity": avg_qty,
    "UniqueItems": unique_items,
    "Cluster": cluster
}])

pred_monetary = reg_model.predict(X_reg)[0]

st.success(f"Cluster Customer: {cluster}")
st.info(f"Prediksi Monetary: {pred_monetary:,.2f}")
