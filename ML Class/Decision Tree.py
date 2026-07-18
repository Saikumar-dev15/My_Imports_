#Hyper parameter Tuning

#DecisionTreeClassifier(max_depth=5)
#| HYPERPARAMETER  | Meaning                                     |
#| Learning Rate   | Control How fast the model learns           |
#| Epochs          | No.of training iterations                   |
#| Batch Size      | No.of samples processed ar once             |
#| Max Depth       | Maximumm depth of a tree                    |
#| n_estimators    | No.of  trees in Random Forest               |
#| k               | No of Neighbors in KNN                      |


#Hyperparameter  Tuning
#1.Grid Search     #2.Random Search       #3.Bayes in optimaization

#Grid Search = 
n_estimators = 50,100,200
max_depth = 5,10,15

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd

df = pd.read_csv("ML Class/students.csv")

df["Pass"] = (df["Final_Marks"] >= 50).astype(int)

x = df[["Previous_Marks"]]
y = df["Pass"]

x_train, x_test, y_train , y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

paras = {
    "max_depth": [2, 4, 6, 8],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=paras,
    cv=3,)                            #our data is divided  into 3 parts. (2- training and 1- testing)
grid.fit(x_train, y_train)

#print(grid.best_params_)
#print(grid.best_score_)





from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#print(X.shape)   # (569, 30)
#print(len(y))    # 569


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

# Parameters
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "criterion": ["gini", "entropy"]
}

# Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

#print("Best Parameters:", grid.best_params_)
#print("Best Score:", grid.best_score_)

y_pred = grid.predict(X_test)
#print(f"y_pred: ",y_pred)
#print("Accuracy:", accuracy_score(y_test, y_pred))




#Random Search 

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble   import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test , y_train , y_test = train_test_split(
    X,y, test_size=0.2 , random_state=42
) 

model = RandomForestClassifier()

params ={
    'n_estimators': [50,100,150,200],
    'max_depth': [5,10,15,20],
    'min_samples_split': [2,5,10]
}

rand = RandomizedSearchCV(model, params, cv =3, n_iter=5, scoring="accuracy" )

rand.fit(X_train, y_train)

#print(rand.best_estimator_)
#print(rand.best_params_)
print(rand.best_score_)

