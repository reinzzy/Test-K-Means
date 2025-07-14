import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="K-Means Clustering UI", layout="wide")
st.title("K-Means Clustering")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

# Parameter K dan max_iter
k = st.slider("Pilih jumlah kluster (k):", min_value=1, max_value=10, value=3)
max_iter = st.slider("Maksimal jumlah iterasi:", min_value=1, max_value=100, value=10)

# Load Data
data = None
data_index = None
feature_columns = None
original_df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    original_df = df.copy()

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    start_col = 'Age'
    end_col = 'Cleanliness'
    cols = list(df.columns)
    if start_col in cols and end_col in cols:
        start_idx = cols.index(start_col)
        end_idx = cols.index(end_col) + 1
        feature_columns = cols[start_idx:end_idx]
        df = df[feature_columns]
    else:
        st.error("Kolom 'Age' sampai 'Cleanliness' tidak ditemukan dalam file.")

    st.write("## Data yang Digunakan untuk Clustering:", df.head())
    data = df.values
    data_index = original_df.index + 1

if data is not None:
    st.subheader("📋 Proses Iterasi K-Means")

    np.random.seed(42)
    initial_indices = np.random.choice(len(data), size=k, replace=False)
    centroids = data[initial_indices]

    converged = False

    for iter_num in range(1, max_iter + 1):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        cluster_headers = [f"Centroid {i + 1}" for i in range(k)]
        iter_table = pd.DataFrame(distances, columns=cluster_headers)
        iter_table.insert(0, "Data ke", data_index)
        iter_table["Terdekat"] = labels
        iter_table["Cluster"] = labels + 1

        st.markdown(f"### Iterasi ke-{iter_num}")
        st.dataframe(iter_table.style.format(precision=6))

        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)
        ])

        if np.allclose(new_centroids, centroids):
            st.info("Centroid tidak berubah. Iterasi dihentikan.")
            converged = True
            break

        centroids = new_centroids

    if not converged:
        st.warning("Iterasi telah mencapai batas maksimal.")

    # Tampilkan hasil akhir
    final_labels = labels
    if feature_columns:
        df_result = original_df[feature_columns].copy()
    else:
        df_result = pd.DataFrame(data, columns=[f"Fitur {i + 1}" for i in range(data.shape[1])])

    df_result['Cluster'] = final_labels + 1
    st.subheader("HASIL AKHIR K-MEANS")
    st.dataframe(df_result)

    # Visualisasi
    st.subheader("Hasil Clustering KMeans")

    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        reduced_centroids = pca.transform(centroids)
    else:
        reduced_data = data
        reduced_centroids = centroids

    cluster_colors = [plt.cm.viridis(i / k) for i in range(k)]

    fig, ax = plt.subplots(figsize=(5, 4))

    for cluster_id in range(k):
        cluster_points = reduced_data[final_labels == cluster_id]
        ax.scatter(cluster_points[:, 1], cluster_points[:, 0],
                   color=cluster_colors[cluster_id],
                   label=f"Cluster {cluster_id + 1}", alpha=0.6)

    ax.scatter(reduced_centroids[:, 1], reduced_centroids[:, 0],
               color='red', s=60, marker='X', label='Centroid')

    ax.set_title("Visualisasi Klaster (Orientasi Horizontal)", fontsize=10)
    ax.set_xlabel("Komponen PCA 2 (Horizontal)", fontsize=9)
    ax.set_ylabel("Komponen PCA 1 (Vertikal)", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(loc='upper right', fontsize=8)

    st.pyplot(fig)

    # Elbow Method
    st.subheader("Elbow Method")

    sse_list = []
    max_k = min(10, len(data))
    for i in range(1, max_k + 1):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(data)
        sse_list.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, max_k + 1), sse_list, marker='o')
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xlabel("Jumlah Cluster (k)")
    ax_elbow.set_ylabel("WCSS")
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)

else:
    st.info("Silakan upload file CSV untuk memulai proses clustering.")
