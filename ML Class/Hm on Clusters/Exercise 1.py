import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
data = {
    "Age" : [21, 25, 30, 35, 40],
    "Salary" : [50000, 60000, 70000, 80000, 90000],
    "Spending Score": [30000, 35000, 40000, 45000, 50000]
    
}
df = pd.DataFrame(data)
#print(df)

scalar = StandardScaler()
scaled_data = scalar.fit_transform(df)

kmeans = KMeans(
    n_clusters=2 , random_state=42
)
kmeans.fit(df)
df["Cluster"] = kmeans.labels_
#print(df)

k = kmeans.fit_predict(df)
print("KMeans: ", k)


hc = AgglomerativeClustering(
    n_clusters=2 
)

hc.fit(df)
z = hc.fit_predict(df)

print("Agglomerative: ", z)

plt.scatter(df["Age"], df["Salary"], c=df["Cluster"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

#conclusion for this exercise KMeans:  [1 1 0 0 0], Agglomerative:  [0 0 0 1 1]

#for second next exercise i change No.of clusters =3 then i got
#output as Kmeans = [00011],  agglomerative = [11200]

from sklearn.metrics import adjusted_rand_score

score = adjusted_rand_score(k, z)

print(score)