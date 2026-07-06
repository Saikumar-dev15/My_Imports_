import pandas as pd

df = pd.read_csv("Pandas/Data.csv")

#Filtering = Keeping the rows that match a condition

#Top_marks = df[df["Marks"] >= 90]                #it will filter the rows where the Marks are greater than or equal to 90

ff_List = df[(df["Age"] == 22) | 
            (df["Marks"]== 88)]

#print(Top_marks)
#print(ff_List)