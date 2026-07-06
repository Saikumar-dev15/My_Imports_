#Data cleaning = the process of fixing or removing:
#                incomplete, incorrect, inaccurate or irrelevant data
#                75% of work done with pandas is data cleaning

import pandas as pd

df = pd.read_csv("Pandas/Data.csv")

#  1. Drop irrelevant columns
df = df.drop(columns =["City"])

#print(df)


#  2. Handle missing data
#df = df.dropna(subset=["Marks"])          #drop the rows where the Marks are missing
#df= df.fillna({"Marks": "None"})
#print(df)

#  3.  fix inconsistent data

df["Department"] = df["Department"].replace({"CSE": "DATA SCIENCE"})

#print(df.to_string())


#   4. Standardize text
df["Name"] = df["Name"].str.lower()          #convert the first letter of each word to lowercase
#print(df.to_string())


#  5 . Fix data types
df["Department"] =df["Department"].astype(bool)
#print(df.to_string())



#  6. Remove Duplicates 

df = df.drop_duplicates()
print(df.to_string())