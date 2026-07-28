import pandas as pd
import matplotlib.pyplot as plt
import time 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets  import load_iris
from sklearn.metrics import accuracy_score

df = load_iris()

x = df.data
y = df.target

scalar = StandardScaler()
scaled_df = scalar.fit_transform(x)
#print(scaled_df)

pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(scaled_df)
#print(pca_scaled)

lda = LinearDiscriminantAnalysis(n_components=2)
lda_scaled = lda.fit_transform(x,y)
#print(lda_scaled)

plt.scatter(pca_scaled[:,0], pca_scaled[:, 1], c=y )
plt.xlabel("Principle component 1")
plt.ylabel("Principle component 2")
#plt.show()


plt.scatter(lda_scaled[:,0], lda_scaled[:, 1], c=y )
plt.xlabel("LDA 1")
plt.ylabel("LDA 2")
#plt.show()

x_train , x_test , y_train , y_test = train_test_split(
    x,y, test_size=0.2 , random_state=42
)

model = LogisticRegression()
start = time.time()
model.fit(x_train, y_train)
end = time.time()
y_pred = model.predict(x_test)
print(f"y_pred_original: {y_pred}")

acc_score_original = accuracy_score(y_test, y_pred)
#print(f"acc_score_original: {acc_score_original}")
time_original = end-start
#print(f"time_original: {time_original}")

#Pca-Transformed data using Logestic reg

xp_train , xp_test , yp_train , yp_test = train_test_split(
    pca_scaled,y, test_size=0.2 , random_state=42
)

model = LogisticRegression()
start = time.time()
model.fit(xp_train, yp_train)
end = time.time()
y_pred_pca = model.predict(xp_test)
print(f"y_pred_lda: {y_pred_pca}")

acc_score_pca = accuracy_score(yp_test, y_pred_pca)
#print(f"acc_score_pca: {acc_score_pca}")
time_pca = end -start
#print(f"time_pca: {time_pca}")


#LDA-Transformed data using Logestic reg

xl_train , xl_test , yl_train , yl_test = train_test_split(
    lda_scaled,y, test_size=0.2 , random_state=42
)

model = LogisticRegression()
start = time.time()
model.fit(xl_train, yl_train)
end = time.time()
y_pred_lda = model.predict(xl_test)
print(f"y_pred_lda: {y_pred_lda}")

acc_score_lda = accuracy_score(yl_test, y_pred_lda)
#print(f"acc_score_lda: {acc_score_lda}")
time_lda = end -start
#print(f"time_lda: {time_lda}")


results = pd.DataFrame({
    "Dataset": ["Original", "PCA", "LDA"],
    "Accuracy": [acc_score_original, acc_score_pca, acc_score_lda],
    "Features": [x.shape[1], pca_scaled.shape[1], lda_scaled.shape[1]],
    "Training Time": [time_original, time_pca, time_lda]
})

print(results)


print("\n==============Conclusion===================")

if acc_score_lda >= acc_score_original and acc_score_lda >= acc_score_original :
    print("LDA is the best method for the Iris dataset ")
    print("Reasons: ")
    print("- Achieved 100% accuracy.")
    print("- Reduced the number of features from 4 to 2.")
    print("- Training time is lower than the original dataset.")
    print("- LDA uses class labels, so it separates the classes effectively.")
    
elif acc_score_original > acc_score_pca:
    print("Original dataset performed the best because it retained all features")
    
else :
    print("PCA performed well by reducing the features and maintaining good accuracy ")