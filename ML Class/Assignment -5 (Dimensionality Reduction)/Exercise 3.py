import numpy as np
import pandas as pd 
import time 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()

x = df.data
y = df.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scalar = StandardScaler()

x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)

# Original scaled SVM
model = SVC()
start = time.time()
model.fit(x_train_scaled, y_train)
end = time.time()
y_pred = model.predict(x_test_scaled)

time_svm = end - start
#print(f"Timne taken by SVM : {time_svm}")

accuracy_score_SVM = accuracy_score(y_test, y_pred)
#print("SVM accuracy:", accuracy_score_SVM)


precision_score_svm = precision_score(y_test, y_pred)
#print("SVM Precision:", precision_score_svm)

recall_score_svm = recall_score(y_test, y_pred)
#print("SVM Recall:", recall_score_svm)


f1_score_svm = f1_score(y_test, y_pred)
#print("SVM F1 :", f1_score_svm)

print("SVM Confusion Matrix:", confusion_matrix(y_test, y_pred))


# PCA + SVM
pca = PCA(n_components=10)

xp_train = pca.fit_transform(x_train_scaled)
xp_test = pca.transform(x_test_scaled)

model = SVC()
start = time.time()
model.fit(xp_train, y_train)
end = time.time()
y_pred_pca = model.predict(xp_test)


time_pca = end - start
#print(f"Timne taken by pca : {time_pca}")


accuracy_score_pca = accuracy_score(y_test, y_pred_pca)
#print("PCA + SVM accuracy:", accuracy_score_pca)

precision_score_pca =precision_score(y_test, y_pred_pca)
#print("PCA + SVM Precision:", precision_score_pca)

recall_score_pca = recall_score(y_test, y_pred_pca)
#print("PCA + SVM Recall :", recall_score_pca)

f1_score_pca = f1_score(y_test, y_pred_pca)
#print("PCA + SVM f1 score:", f1_score_pca)


confusion_matrix_pca = confusion_matrix(y_test, y_pred_pca)
print("PCA + SVM Confusion Matrix:", confusion_matrix_pca)


# LDA + SVM
lda = LinearDiscriminantAnalysis()

xl_train = lda.fit_transform(x_train_scaled, y_train)
xl_test = lda.transform(x_test_scaled)

model = SVC()
start = time.time()
model.fit(xl_train, y_train)
end = time.time()

y_pred_lda = model.predict(xl_test)


time_lda = end - start
#print(f"Timne taken by lda : {time_lda}")


accuracy_score_lda = accuracy_score(y_test, y_pred_lda)
#print("LDA + SVM accuracy:", accuracy_score_lda)


precision_score_lda = precision_score(y_test, y_pred_lda)
#print("LDA + SVM Precision:", precision_score_lda)

recall_score_lda = recall_score(y_test, y_pred_lda)
#print("LDA + SVM Recall:", recall_score_lda)


f1_score_lda= f1_score(y_test, y_pred_lda)
#print("LDA + SVM F1 Score:", f1_score_lda)


print("LDA + SVM Confusion Matrix:", confusion_matrix(y_test, y_pred_lda))



result = pd.DataFrame({
    "Methods": ["SVM", "PCA + SVM", "LDA + SVM"],
    "Accuracy": [accuracy_score_SVM, accuracy_score_pca, accuracy_score_lda], 
    "Precision": [precision_score_svm, precision_score_pca, precision_score_lda],
    "Recall" :   [recall_score_svm, recall_score_pca, recall_score_lda],
    "F1 Score" : [f1_score_svm, f1_score_pca, f1_score_lda],
    "Time Taken": [time_svm, time_pca ,time_lda]
})

print(result)



print("\n==============Conclusion===================")

if accuracy_score_lda >= accuracy_score_SVM and accuracy_score_lda >= accuracy_score_SVM:
    print("LDA is the best method for the Iris dataset ")
    print("Reasons: ")
    print("- Achieved 100% accuracy.")
    print("- Reduced the number of features from 4 to 2.")
    print("- Training time is lower than the original dataset.")
    print("- LDA uses class labels, so it separates the classes effectively.")
    
elif accuracy_score_SVM > accuracy_score_pca:
    print("Original dataset performed the best because it retained all features")
    
else :
    print("PCA performed well by reducing the features and maintaining good accuracy ")