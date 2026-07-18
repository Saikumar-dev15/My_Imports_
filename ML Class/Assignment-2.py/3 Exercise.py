#Using the Breast Cancer dataset:
#Train Logistic Regression.
#Tune using Grid Search.
#Predict on test data.
#Display:
#Confusion Matrix
#Accuracy
#Precision
#Recall
#F1 Score
#Compare metrics before and after tuning.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df= load_breast_cancer()

x = df.data
y = df.target

x_train , x_test , y_train , y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")

f1score = f1_score(y_test, y_pred)
print(f"f1 score: {f1score}")


cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: {cm}")

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

params = {
    'C' : [0.01, 0.1, 1, 10, 100],
    'max_iter' : [100, 200, 500]
}

grid =GridSearchCV(
    estimator=model,
    param_grid=params,
    cv =3,
    scoring="accuracy"
)

grid.fit(x_train, y_train) 
print("Best Model from Grid: ", grid.best_estimator_)
print("Best Score from Grid: ", grid.best_score_)


grid_pred = grid.predict(x_test)


accuracy = accuracy_score(y_test, grid_pred)
print(f"Accuracy: {accuracy}")

precision = precision_score(y_test, grid_pred)
print(f"Precision: {precision}")

recall = recall_score(y_test, grid_pred)
print(f"Recall: {recall}")

f1score = f1_score(y_test, grid_pred)
print(f"f1 score: {f1score}")


cm = confusion_matrix(y_test, grid_pred)
print(f"Confusion Matrix: {cm}")
