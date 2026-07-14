import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


df = pd.read_csv("ML Class/Students Performance.csv")
#print(df.head())

#print(df.shape)
#print(df.columns)
#print(df.dtypes)
#print(df.describe)

duplicate = df[df.duplicated()]
#print(duplicate)
#print(df.duplicated().sum())
#print(df.drop_duplicates)

q1 = df["Writing_Score"].quantile(0.25)
q3 = df["Reading_Score"].quantile(0.75)

IQR = q3 - q1

lower = q1 - 1.5*IQR
upper = q3 + 1.5*IQR

outlier = df[(df["Writing_Score"]< lower) | (df["Reading_Score"]> upper)]

x = df[["Math_Score",
        "Writing_Score",
        "Placement_Score","Club_Join_Date"
        ]]
columns = MinMaxScaler()
scalar = columns.fit_transform(x)
#print(scalar)

x = df[["Reading_Score",
        "Writing_Score",
       "Placement_Score",
       "Club_Join_Date"]]

y = df[["Math_Score"]]

x_train, x_test, y_train , y_test_reg = train_test_split(
    x,y, train_size=0.8, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred_reg = model.predict(x_test)
print(y_pred_reg)

df["Pass"] = (df["Math_Score"] >= 65).astype(int)
print(df["Pass"].value_counts())

x = df[["Reading_Score",
        "Writing_Score",
       "Placement_Score",
       "Club_Join_Date"]]

y = df["Pass"]

x_train, x_test, y_train , y_test = train_test_split(
    x,y, train_size=0.8, random_state=42 , stratify=y
)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_pred)
print(y_train.value_counts())
print(y_test.value_counts())


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

precision = precision_score(y_test, y_pred)
print(f"Precision_score: {precision}")

recall = recall_score(y_test, y_pred)
print(f"Recall score: {recall}")

f1score = f1_score(y_test, y_pred)
print(f"f1 score: {f1score}")

confusion  = confusion_matrix(y_test, y_pred)
print(f"confusion_matrix: {confusion}")


print(f"MAE: ", mean_absolute_error(y_test_reg, y_pred_reg))
print(f"MSE: ", mean_squared_error(y_test_reg, y_pred_reg))
print(f"RMSE: ", root_mean_squared_error(y_test_reg, y_pred_reg))
print(f"f_score: ", r2_score(y_test_reg, y_pred_reg))


X = df["Math_Score"]
Y= df["Placement_Score"]

plt.bar(X,Y, color= "Blue",
             alpha=0.4,
             )
plt.title("Maths_score vs Placement_Score")
plt.xlabel("Maths_score")
plt.ylabel("Placement_Score")
plt.show()


X = df["Math_Score"]
Y= df["Club_Join_Date"]
plt.scatter(X,Y , color="orange")
plt.title("Maths_score vs Club_Join_Date")
plt.xlabel("Maths_score")
plt.ylabel("Club_Join_Date")
plt.show()


X= df["Writing_Score"]
plt.hist(X,    color= "orange",
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
