import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("ML Class/Students Performance.csv")

#print(df.head(4))


df = df[["Reading_Score",
        "Writing_Score",
        "Placement_Score",
        "Club_Join_Date",
        "Math_Score"]]

df["Math_Score"] = df["Math_Score"].fillna(df["Math_Score"].median())

#print(df["Math_Score"])

df = df.fillna(df.median())

#print(df)
#df = df.drop_duplicates
#print(df)

q1 = df["Writing_Score"].quantile(0.25)
q3 = df["Reading_Score"].quantile(0.75)

IQR = q3 -q1

#print(IQR)

lower = q1 - 1.5* IQR 
upper = q3 + 1.5*IQR

#print(upper)

outlier = df[(df["Writing_Score"] < lower)| (df["Reading_Score"] > upper) ]
print(outlier)

data = np.array(df["Math_Score"])

print("Mean: ", np.mean(data))
print("Median: ", np.median(data))
print("Range: ", np.ptp(data))
print("Variance: ", np.var(data))
print("Standard Deviation: ", np.std(data))

df["Pass"] = (df["Math_Score"] >= 70).astype(int)
#print(df["Pass"])


x =df[["Placement_Score",
      "Reading_Score",
      "Writing_Score",
      "Club_Join_Date"]]

y = df["Math_Score"]

x_train, x_test, y_train , y_test = train_test_split(
    x, y , test_size=0.2 , random_state=42
)


model_1 = LinearRegression()
model_1.fit(x_train, y_train)
y_pred1= model_1.predict(x_test)
print("Slope(m): ", model_1.coef_[0])
print("Intercept: ", model_1.intercept_)

model_2 = GradientBoostingRegressor(
    random_state=42
)

#Grid Search
param_grid = {
    "n_estimators": [50,100,150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search_reg = GridSearchCV(
    estimator = model_2,
    param_grid = param_grid,
    cv =3,
    scoring = "r2",
    n_jobs = -1
)

grid_search_reg.fit(x_train,y_train)
print("Grid Search Score: ",grid_search_reg.best_score_)


#Rand search

params ={
    'n_estimators': [50,100,150,200],
    'max_depth': [5,10,15,20],
    'min_samples_split': [2,5,10]
}

rand = RandomizedSearchCV(model_2, params, cv =3, n_iter=30, scoring="r2" , random_state=42 )
rand.fit(x_train,y_train)
print("Rand Search Score: ",rand.best_score_)


corr = df.corr(numeric_only=True)
print("Correlation: ",corr)
sns.heatmap(corr, annot=True, cmap="rainbow")
plt.show()


X = df["Math_Score"]
Y= df["Placement_Score"]

plt.scatter(range(len(df)), X , color="blue", label="Math_Score")
plt.scatter(range(len(df)), Y , color="Red", label="Placement_Score")
plt.title("Maths_score vs Placement_Score")
plt.xlabel("Maths_score")
plt.ylabel("Placement_Score")
plt.show()


X = df["Math_Score"]
Y= df["Club_Join_Date"]
plt.scatter(X,Y , color="Green")
plt.title("Maths_score vs Club_Join_Date")
plt.xlabel("Maths_score")
plt.ylabel("Club_Join_Date")
plt.show()


X= df["Writing_Score"]
plt.hist(X,    color= "Violet",
               bins= 50,
               edgecolor="black")
plt.title("Score board")
plt.xlabel("Writing_Score")
plt.show()

X = df["Reading_Score"]
sns.lineplot(X, color= "orange")
plt.title("Reading_Score")
plt.xlabel("Reading_Score")
plt.show()

sns.pairplot(df, hue="Pass")
plt.show()


best_model = grid_search_reg.best_estimator_

y_pred = best_model.predict(x_test)
print("Test R2 Score for GridSearch:", r2_score(y_test, y_pred))

best_rand_model = rand.best_estimator_

y_pred_rand = best_rand_model.predict(x_test)
print("Test R2 Score for RandSearch:", r2_score(y_test, y_pred_rand))