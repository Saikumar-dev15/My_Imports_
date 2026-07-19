from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import numpy as np

wine = load_wine()

x = wine.data
y = wine.target

x_train , x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test) 
print(f"Class labels from model: {y_pred}")

scalar = StandardScaler()

x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled  = scalar.transform(x_test)

model.fit(x_train_scaled, y_train)
y_pred_knn = model.predict(x_test_scaled)
print(f"Class labels of Scaled data: {y_pred_knn}")

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"]
}

grid = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    cv= 3,
    scoring='accuracy'
)

grid.fit(x_train_scaled, y_train)
print(f"Model Best Parameters : {grid.best_estimator_}")
print(f"Model Accuracy: {grid.best_score_}")

#retrain data by standard scalar+KNN model
accuracy = accuracy_score(y_test,y_pred_knn)
print(f"Retrain Accuracy score by scaled  data: {accuracy}")


#Retrain data by griedsearch + knn
best_knn = grid.best_estimator_

y_pred_best = best_knn.predict(x_test_scaled)
#print(y_pred_best)

accuracy = accuracy_score(y_test, y_pred_best)
print(f"Retrain Accuracy score from best parameter: {accuracy}")



print(np.array_equal(y_pred_knn, y_pred_best))