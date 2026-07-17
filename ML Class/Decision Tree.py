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

x="max_depth"
y= "criterion"

x_train, x_test, y_train , y_test_reg = train_test_split(
    x,y, train_size=0.8, random_state=42
)

model = DecisionTreeClassifier()

paras = {
    "max_depth": [2, 4, 6, 8],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(model, paras, cv=3)
grid.fit(x_train, y_train)

print(grid.best_params_)
print(grid.best_score_)