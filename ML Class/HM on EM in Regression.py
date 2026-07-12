import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection  import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

df = pd.read_csv("ML Class/students.csv")

#print(df.head(5))
#print(df.tail(5))

#print(df.shape)
#print(df.dtypes)
#print(df.info)
#print(df.describe())

#print(df.drop_duplicates())

q1 = df["Hours_Studied"].quantile(0.25)
q3 = df["Hours_Studied"].quantile(0.75)

IQR = q3 -q1

lower = q1 - 1.5*IQR
upper = q3 + 1.5*IQR

outlier = df[(df["Hours_Studied"] < lower) | (df["Hours_Studied"] > upper)]

#print(outlier)

x= df[["Student_ID",
       "Hours_Studied",
       "Attendance",
       "Assignments_Completed",
       "Previous_Marks",
       "Final_Marks"]]

scalar = MinMaxScaler()
df_scaled = scalar.fit_transform(x)

#print(df_scaled)


plt.hist(df["Final_Marks"],bins=5,
                        color ="green",
                        edgecolor = "Black",
                        alpha = 0.5)

plt.show()

plt.boxplot(df["Hours_Studied"])
plt.xlabel("Hours studied")
plt.show()

plt.scatter(df["Hours_Studied"], df["Final_Marks"])
plt.title("Hours vs Marks")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()


#Linear regression 

x= df[["Hours_Studied",
       "Student_ID",
       "Attendance",
       "Assignments_Completed",
       "Previous_Marks"]]
y= df["Final_Marks"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8 , random_state=42
)


model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

print("MAE: ", mean_absolute_error(df["Final_Marks"],df["Previous_Marks"]))
print("MSE: ", mean_squared_error(df["Final_Marks"],df["Previous_Marks"]))
print("RMSE: ", root_mean_squared_error(df["Final_Marks"],df["Previous_Marks"]))
print("R2_score: ", r2_score(df["Final_Marks"],df["Previous_Marks"]))