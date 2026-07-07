import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df = pd.read_csv("ML Class/Datasheet.csv")

#print(df.head(4))
#print(df.tail(5))
#print(df.shape)
#print(df.columns)
#print(df.info)
#print(df.describe())

#print(df.isnull().sum())
#print(df.drop_duplicates())

q1 = df["Price"].quantile(0.25)
q3 = df["Price"].quantile(0.75)

IQR = q3 - q1 

#print(IQR)

lower = q1 -1.5*IQR
upper = q3 + 1.5*IQR

df["Price"] = df["Price"].clip (lower= lower , upper = upper)
#print(df["Price"])

#print(df.loc[:, "Price"])


