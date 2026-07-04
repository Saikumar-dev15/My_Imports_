import pandas as pd

df = pd.read_csv ("Pandas/Data.csv")

#Selection by Columns
print(df["Name"].to_string())