import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()

x = df.data
y = df.target

scalar = StandardScaler()
scaled = scalar.fit_transform(x)


lda = LinearDiscriminantAnalysis()
xl = lda.fit_transform(scaled,y)
#print(xl)

x_train , x_test , y_train , y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#print(y_pred)

classification_report_svm = classification_report(y_test, y_pred)
print("=========================Classification Report of SVM===============================")
print(classification_report_svm)

acc_score_original = accuracy_score(y_test, y_pred)
print(f"acc_score_SVM: {acc_score_original}")


confusion_matrix_svm = confusion_matrix(y_test, y_pred)
print(f"Confuion Matrix of SVM: {confusion_matrix_svm}")






xl_train , xl_test , yl_train , yl_test = train_test_split(
    xl, y, test_size=0.2, random_state=42
)

model = SVC()
model.fit(xl_train, yl_train)
y_pred_lda = model.predict(xl_test)
#print(y_pred_lda)

classification_report_LDA = classification_report(y_test, y_pred_lda)
print("=========================Classification Report of LDA===============================")
print(classification_report_LDA)

acc_score_lda= accuracy_score(y_test, y_pred_lda)
print(f"acc_score_LDA: {acc_score_lda}")

confusion_matrix_lda = confusion_matrix(y_test, y_pred_lda)
print(f"Confuion Matrix of lda: {confusion_matrix_lda}")