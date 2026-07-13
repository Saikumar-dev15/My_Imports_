import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
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

x_train, x_test, y_train , y_test = train_test_split(
    x,y, train_size=0.8, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


df["Pass"] = (df["Math_Score"] >= 75).astype(int)
print(df["Pass"].value_counts())

x = df[["Reading_Score",
        "Writing_Score",
       "Placement_Score",
       "Club_Join_Date"]]

y = df["Pass"]

x_train, x_test, y_train , y_test = train_test_split(
    x,y, test_size=0.2, random_state=42 , stratify=y
)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_test, y_test)

x_pred = model.predict(x_test)

print(x_pred)
print(y_train.value_counts())
print(y_test.value_counts())

