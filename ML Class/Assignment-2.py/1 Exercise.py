#Using the Breast Cancer dataset, build a complete classification pipeline that includes data loading, train-test split, 
#model training using Random Forest, evaluation using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix, 
#followed by hyperparameter tuning using both GridSearchCV and RandomizedSearchCV. 
#Compare the performance of the baseline model with the tuned models and identify the best-performing hyperparameter combination.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
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

model = RandomForestClassifier()

params = {
    'n_estimators':[100,150,200,250],
    'max_depth' :[10,15,20,25],
    'min_samples_split': [2,5,7]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv =3,
    scoring="accuracy"
)

grid.fit(x_train, y_train) 
print("Best Model from Grid: ", grid.best_estimator_)
print("Best Score from Grid: ", grid.best_score_)

rand = RandomizedSearchCV(model, params,cv=3, n_iter=5, scoring="accuracy")

rand.fit(x_train, y_train)
print("Best model from Rand: ", rand.best_estimator_)
print("Best score from Rand: ", rand.best_score_)

best_model = grid.best_estimator_         # I choose a best model from gridsearch

y_pred = best_model.predict(x_test)

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

