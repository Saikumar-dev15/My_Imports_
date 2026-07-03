import pandas as pd 

data = {"Name": ["Sai", "Mohit", "Gowtham"],
        "Age": [21, 22, 23],
}

df =pd.DataFrame(data, index = ["Employee1", "Employee2", "Employee3"])
#print(df.loc["Employee1"])
#print(df.iloc[0])

#add new column
df["Job"] = ["Data Analyst", "Data Scientist", "Data Engineer"]
#print(df)

#Add new row
new_rows = pd.DataFrame([{"Name": "Shiva", "Age": 24, "Job": "Java Developer"},
                        {"Name": "Jack", "Age": 25, "Job": "Machine Learning Engineer"}],
                        index = ["Employee 4", "Employee 5"])
df = pd.concat([df, new_rows])             #nothing but append the new row to the existing DataFrame
print(df)