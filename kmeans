import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster



# Load dataset
df = pd.read_csv('sales_data_sample.csv', encoding='latin1')

# Display first few rows
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)


# Select only numeric columns for clustering
num_df = df.select_dtypes(include=['int64', 'float64']).copy()

# Drop rows with missing values
num_df.dropna(inplace=True)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

print("Scaled data shape:", scaled_data.shape)


inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia / WCSS')
plt.grid(True)
plt.show()


# Assuming optimal k = 3 (you can change based on the plot)
k_opt = 3

kmeans = KMeans(n_clusters=k_opt, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to original dataframe
df['Cluster'] = clusters
print(df['Cluster'].value_counts())


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=clusters, cmap='viridis')
plt.title('K-Means Clustering Visualization (2D PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


#Hierarchical clusttering(_optional)

linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Example: Create 3 clusters
hier_clusters = fcluster(linked, 3, criterion='maxclust')
df['Hier_Cluster'] = hier_clusters


# Cluster summary statistics
print(df.groupby('Cluster').mean(numeric_only=True))

