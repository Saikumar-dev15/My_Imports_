import pandas as pd 
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df =pd.read_csv("ML Class/HM on Clusters/IncomeSpending.csv")

#print(df)

hc = AgglomerativeClustering(
    n_clusters=3  
)

df["cluster"] = hc.fit_predict(df)
print(df)

print(df.groupby("cluster")[["Income", "Spending"]].mean())


Z = linkage(df, method= 'ward')

dendrogram(Z)
plt.show()