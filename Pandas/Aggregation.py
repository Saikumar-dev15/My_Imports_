import pandas as pd

#Aggregation = Combining multiple values into a single value

df =pd.read_csv("Pandas/Data.csv")

#print(df.mean(numeric_only =True))
#print(df.sum(numeric_only =True))
#print(df.count())

#single column aggregation
#print(df["Marks"].mean())
#print(df["Marks"].sum())


group = df.groupby("Age") 
print(group["Age"].mean())
print(group["Age"].sum())
print(group["Age"].count())
