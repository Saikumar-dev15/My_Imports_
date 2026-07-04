import pandas as pd

df = pd.read_csv("Pandas/Data.CSV")

#print(df.shape)
print(df.to_string())                       #to print the entire data frame without truncation

