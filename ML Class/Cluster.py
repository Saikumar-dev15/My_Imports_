#Clustering is unsupervised data label
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

kmeans.fit(x)

#print(kmeans.labels_)

#print(kmeans.cluster_centers_)



#Exercise


import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.array([[90,85], [92,80], [40,45], [38,58], [88,79], [35,40]])
df = pd.DataFrame(data, columns =['Maths', 'Science'])

#print(df)

kmeans = KMeans(n_clusters=2)
kmeans.fit(df)
df['Cluster'] = kmeans.labels_
print(df)

plt.scatter(df["Maths"], df["Science"], c=df["Cluster"])
plt.xlabel('Maths')
plt.ylabel('Science')
plt.title('Student Clusters')
plt.show()

#Heirarchical Clustering 
#1.Agglomerative Hierarchical Clustering (Bottom-up)
#Ex: if a classs has 60 members then form a group is called Agglomerative Cluster

#2.Divisive Heirarchical Clustering (Top-Down)
#Ex : Same but here groups are divided into individuals..

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data

hc = AgglomerativeClustering(
    n_clusters=3
)

cluster = hc.fit_predict(x)

#print(cluster)

Z = linkage(x, method= 'ward')

dendrogram(Z)
plt.show()