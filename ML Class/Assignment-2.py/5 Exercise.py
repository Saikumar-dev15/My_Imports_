#Load the Iris dataset and:
#Split the data into training and testing sets.
#Train a DecisionTreeClassifier.
#Use GridSearchCV to find the best:
#max_depth = [2, 3, 4, 5, 6]
#criterion = ['gini', 'entropy']
#Print:
#Best parameters
#Best cross-validation score
#Test accuracy


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd 

iris = load_iris()

x = iris.data
y = iris.target

x_train , x_test , y_train , y_test = train_test_split(
    x,y, test_size=0.2 , random_state=42
)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Test accuracy of Model: {y_pred}")

params ={
    'max_depth' : [2,3,4,5,6],
    'criterion' : ['gini', 'entropy']
}

grid = GridSearchCV(
    estimator= model,
    param_grid=params,
    cv=3,
    scoring='accuracy'
)

grid.fit(x_train, y_train)
print(f"Model Best Parameters : {grid.best_estimator_}")
print(f"Model Accuracy: {grid.best_score_}")

grid_pred = grid.predict(x_test)
print(f"Test accuracy of Grid: {grid_pred}")