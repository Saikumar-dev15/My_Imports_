# Seriesa = Series is a one-dimensional array-like object that can hold many data types, including objects, floats, and integers. It is similar to a column in a DataFrame.

import  pandas as pd  # type: ignore
data = [101, 102, 103]
series = pd.Series(data, index = ["a", "b", "c"])
#print(series)
#print(series.loc["a"])             #to access the value of the index "a"
series.loc["a"] = 259               #to to change the value of the index "a"
#print(series)

series = pd.Series(data, index = ["a","b", "c"])
#print(series.iloc[0])                #to acces the values of indexes by their integer location


Marks = [89, 93, 69, 47, 77]
series = pd.Series(Marks)
#print(series[series > 70])


#Dictinoary

calories = {"Day1": 420, "Day2": 380, "Day3": 390} 

series = pd.Series(calories)
#print(series)
series["Day3"] += 500
print(series)
