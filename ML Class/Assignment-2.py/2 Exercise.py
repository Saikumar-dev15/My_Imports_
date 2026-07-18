#Build a classifier for titanic dataset and:
#Perform EDA.
#Handle missing values (if any).
#Split data.
#Train baseline model.
#Evaluate accuracy.
#Apply Grid Search.
#Apply Random Search.
#Compare all three models:
#Baseline
#Grid Search Tuned
#Random Search Tuned
#Present findings in a table.


import seaborn  as sns
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
) 
from sklearn.ensemble import RandomForestClassifier

df =sns.load_dataset("titanic")

#print(df.head(3))
print(df.drop_duplicates)

df = df[["pclass",
        "age",
        "sex",
        "survived",
        "sibsp",
        "fare"]]

df["age"] = df["age"].fillna(df["age"].median())   #Handling missing values

df["sex"] = df["sex"].map ({                       #convert character into number to count
    "male": 0,
    "female": 1
})

x =df[["pclass",
       "sex",
       "age",
       "sibsp"]]

y = df["survived"]

x_train , x_test, y_train ,y_test = train_test_split(
    x,y, test_size=0.2 , random_state=42
)

model = RandomForestClassifier()

params = {
    'n_estimator' : [100,150,200,250],
    'max_depth' : [10,15,20,25],
    "min_split_data" : [2,7,9]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=3,
    scoring="accuracy"
)

print("Best model from Grid: ", grid.best_estimator_)
print("Best Score from Grid: ", grid.best_score_)