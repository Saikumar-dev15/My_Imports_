#pca  focus on Data and its unsupervised data
#LDA focus on Labels and its supervised data. used for classification purpose only...


import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = {
    'Height' : [150, 160, 170, 180, 175],
    'Weight' : [50,60,70,80,75],
    'Age'    : [20,22,24,26,25]
}

df = pd.DataFrame(data)

scalar = StandardScaler()
scaled_data = scalar.fit_transform(df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
#print(pca_data)

#print(pca.explained_variance_ratio_)




#LDA 

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()

x = iris.data
y = iris.target

lda = LinearDiscriminantAnalysis(n_components=2)

x_lda = lda.fit_transform(x,y)

print(x_lda.shape)

