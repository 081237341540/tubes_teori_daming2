import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Clustering + Regression", layout="wide")
st.title("Customer Clustering dan Prediksi Belanja")

@st.cache_data
def load_data():
    return pd.read_csv("rfm_customer.csv")

rfm = load_data()

# =========================
# TRAIN MODEL RINGAN DI APP
# =========================
features_cluster = ["Recency", "Frequency", "Monetary", "AvgQuantity", "UniqueItems"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features_cluster])

# jumlah cluster ambil dari kolom Cluster di CSV (hasil dari Colab)
n_clusters = int(rfm["Cluster"].nunique()) if "Cluster" in rfm.columns else 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA untuk visualisasi 2D clustering
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Regression: prediksi Monetary dari fitur lain + cluster
Xr = rfm[["Recency", "Frequency", "AvgQuantity", "UniqueItems", "Cluster"]]
yr = rfm["Monetary"]

X_train, X_test, y_train, y_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
reg_model.fit(X_train, y_train)

pred_test = reg_model.predict(X_test)

mae = float(mean_absolute_error(y_test, pred_test))
rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
r2 = float(r2_score(y_test, pred_test))

# =========================
# METRICS DI ATAS
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Jumlah Cluster", n_clusters)
m2.metric("MAE", f"{mae:,.2f}")
m3.metric("RMSE", f"{rmse:,.2f}")
m4.metric("R2", f"{r2:.4f}")

# =========================
# VISUALISASI
# =========================
st.subheader("Visualisasi")

colA, colB = st.columns(2)

with colA:
    st.write("Clustering: PCA 2D")
    fig = plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rfm["Cluster"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D untuk Cluster")
    st.pyplot(fig)

with colB:
    st.write("Regression: Actual vs Predicted (Monetary)")
    fig2 = plt.figure()
    plt.scatter(y_test, pred_test)
    plt.xlabel("Actual Monetary")
    plt.ylabel("Predicted Monetary")
    plt.title("Actual vs Predicted")
    st.pyplot(fig2)

st.subheader("Ringkasan tiap cluster (mean)")
summary = (
    rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary", "AvgQuantity", "UniqueItems"]]
    .mean()
    .round(2)
)
st.dataframe(summary, use_container_width=True)

# =========================
# INPUT STREAMLIT
# =========================
st.subheader("Prediksi dari Customer ID atau input manual")

tab1, tab2 = st.tabs(["Pilih Customer ID", "Input Manual"])

with tab1:
    st.write("Pilih Customer ID untuk lihat cluster dan prediksi Monetary.")
    cust_id = st.selectbox("Customer ID", rfm["Customer_ID"].astype(str).tolist())
    row = rfm[rfm["Customer_ID"].astype(str) == cust_id].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Recency (hari)", int(row["Recency"]))
    b.metric("Frequency", int(row["Frequency"]))
    c.metric("Cluster", int(row["Cluster"]))

    X_reg_one = pd.DataFrame([{
        "Recency": row["Recency"],
        "Frequency": row["Frequency"],
        "AvgQuantity": row["AvgQuantity"],
        "UniqueItems": row["UniqueItems"],
        "Cluster": row["Cluster"],
    }])

    pred_money = float(reg_model.predict(X_reg_one)[0])
    st.success(f"Prediksi Monetary: {pred_money:,.2f}")

    with st.expander("Detail baris customer"):
        st.dataframe(pd.DataFrame([row]), use_container_width=True)

with tab2:
    st.write("Input manual untuk prediksi cluster dan Monetary.")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (hari)", min_value=0, value=30, step=1)
    frequency = c2.number_input("Frequency (invoice unik)", min_value=0, value=5, step=1)
    avg_qty = c3.number_input("AvgQuantity", min_value=0.0, value=10.0, step=1.0)

    c4, c5 = st.columns(2)
    unique_items = c4.number_input("UniqueItems", min_value=0, value=20, step=1)
    monetary_dummy = c5.number_input("Monetary (dummy untuk clustering)", min_value=0.0, value=200.0, step=10.0)

    X_cluster_one = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary_dummy,
        "AvgQuantity": avg_qty,
        "UniqueItems": unique_items,
    }])

    cluster_pred = int(kmeans.predict(scaler.transform(X_cluster_one))[0])

    X_reg_one = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "AvgQuantity": avg_qty,
        "UniqueItems": unique_items,
        "Cluster": cluster_pred,
    }])

    pred_money = float(reg_model.predict(X_reg_one)[0])

    st.success(f"Prediksi Cluster: {cluster_pred}")
    st.info(f"Prediksi Monetary: {pred_money:,.2f}")

st.caption("Data input: rfm_customer.csv (hasil dari Colab). Visual: PCA clustering dan Actual vs Predicted regression.")
