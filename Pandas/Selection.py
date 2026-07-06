import pandas as pd

df = pd.read_csv ("Pandas/Data.csv", index_col="Name") 

#Selection by Columns
#print(df["Name"].to_string())
#print(df["Age"].to_string())
#print(df[["Age", "City"]].to_string())
#print(df)


#Selection by Rows
#print(df.loc["Priya":"Sneha", ["Age","City"]])                  # Selects the row with index "Priya"
#print(df.iloc[0:11:2, 0:3])

Name = input("Enter the name of the person: ")

try:
    print(df.loc[Name])
except KeyError:
    print(f"{Name} not found in the DataFrame.")
    