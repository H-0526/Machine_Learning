import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 数据集路径
DATA_PATH = r"G:\Pycharm\Machine_Learning\data\processed_data\standardized_selected_features_data_combined_wine.csv"

# 输出文件夹路径
PCA_DIR = "PCA_Results"
TSNE_DIR = "tSNE_Visualization"
CLUSTER_DIR = "Clustering_Results"

# 创建输出文件夹
os.makedirs(PCA_DIR, exist_ok=True)
os.makedirs(TSNE_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)

# 加载数据
print("Loading dataset...")
data = pd.read_csv(DATA_PATH)
X = data.iloc[:, :-1]  # 特征矩阵
y = data.iloc[:, -1]   # 标签（质量分级）

# 降维：PCA
print("Performing PCA...")
pca = PCA(n_components=0.95, random_state=RANDOM_SEED)  # 保留95%方差信息
X_pca = pca.fit_transform(X)
pca_variance_ratio = pca.explained_variance_ratio_

# 保存 PCA 结果
pca_results_path = os.path.join(PCA_DIR, "pca_results.csv")
pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]).to_csv(pca_results_path, index=False)
print(f"PCA results saved to {pca_results_path}")

# 绘制 PCA 方差解释率
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_variance_ratio), marker="o")
plt.title("Cumulative Explained Variance Ratio by PCA Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.grid()
plt.savefig(os.path.join(PCA_DIR, "pca_variance_ratio.png"))
plt.close()

# 聚类：KMeans
print("Performing KMeans clustering...")
kmeans = KMeans(n_clusters= 3, random_state=RANDOM_SEED, n_init=10, max_iter=300)
kmeans_labels = kmeans.fit_predict(X_pca)

# 计算轮廓系数，用来选择合适的 K 个簇
silhouette_avg = silhouette_score(X_pca, kmeans_labels)
print(f"KMeans silhouette score: {silhouette_avg:.4f}")

# 保存 KMeans 结果
kmeans_results_path = os.path.join(CLUSTER_DIR, "kmeans_results.csv")
pd.DataFrame({"Cluster": kmeans_labels}).to_csv(kmeans_results_path, index=False)
print(f"KMeans results saved to {kmeans_results_path}")

# 可视化 KMeans 聚类结果
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", legend="full")
plt.title("KMeans Clustering Results (PCA-reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.savefig(os.path.join(CLUSTER_DIR, "kmeans_clustering.png"))
plt.close()

# 聚类：DBSCAN
print("Performing DBSCAN clustering...")
dbscan = DBSCAN(eps=1.0, min_samples=3)
dbscan_labels = dbscan.fit_predict(X_pca)

# 保存 DBSCAN 结果
dbscan_results_path = os.path.join(CLUSTER_DIR, "dbscan_results.csv")
pd.DataFrame({"Cluster": dbscan_labels}).to_csv(dbscan_results_path, index=False)
print(f"DBSCAN results saved to {dbscan_results_path}")

# 可视化 DBSCAN 聚类结果
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette="viridis", legend="full", s=50, alpha=0.7)
plt.title("DBSCAN Clustering Results (PCA-reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster", loc="upper right", bbox_to_anchor=(1.2, 1))
plt.savefig(os.path.join(CLUSTER_DIR, "dbscan_clustering.png"))
plt.close()

print("Dimensionality reduction and clustering completed.")