from sklearn.cluster import  KMeans,AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

kmeans.fit(x)

print(kmeans.labels_)

print(kmeans.cluster_centers_)


hc = AgglomerativeClustering(
    n_clusters=3
)

y = hc.fit_predict(x)
print(y)

Z = linkage(x, method= 'ward')

dendrogram(Z)
plt.show()



#Both K-Means and Agglomerative Clustering successfully identified three major groups in the Iris dataset. The cluster labels differed because cluster numbers are assigned arbitrarily by each algorithm
#Generally, K-Means is expected to perform slightly better because the Iris clusters are relatively compact and well-suited to centroid-based clustering.