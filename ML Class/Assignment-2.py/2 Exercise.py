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
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df =sns.load_dataset("titanic")

#print(df.head(3))
#print(df.drop_duplicates.sum())

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
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#print(f"Baseline of Random Forest Classifier: {y_pred}")


params = {
    'n_estimators': [100,150,200,250],
    'max_depth' : [10,15,20,25],
    "class_weight" : [2,7,9],
    "criterion": ["gini", "entropy"],
    "class_weight": [
        None,
        "balanced",
        "balanced_subsample",
        {0: 1, 1: 9}]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=3,
    scoring="accuracy"
)

grid.fit(x_train, y_train)
grid_pred = grid.predict(x_test)

#print("Best Parameters:", grid.best_estimator_)
#print("Best accuracy score: ", grid.best_score_)


rand = RandomizedSearchCV(model, params, cv=3, n_iter=5, scoring="accuracy", random_state=42)

rand.fit(x_train, y_train)
rand_pred = rand.predict(x_test)
#print("Best parameters from Rand search: ", rand.best_estimator_)
#print("Best Scoring from Rand: ", rand.best_score_)


results = []

results.append({
    "model": "Baseline Random Forest",
    "Best Parameters": "Default",
    "Accuracy": accuracy_score(y_test, y_pred)})

results.append({
    "model": "Grid Search Random Forest",
    "Best Parameters": grid.best_params_,
    "Accuracy": accuracy_score(y_test, grid_pred)})

results.append({
    "model": "Random seaech random forest",
    "Best parameters" : rand.best_params_,
    "Accuracy": accuracy_score(y_test, rand_pred)
})

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

sns.countplot(data=df, x="survived")
plt.title("Survived Count")
plt.show()


sns.histplot(
    data=df,
    x="age",
    kde=True
)

plt.title("Age Distribution")
plt.show()
