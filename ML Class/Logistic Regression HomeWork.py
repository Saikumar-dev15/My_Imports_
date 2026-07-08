import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    "Study_hours" : [1,2,3,4,5,6],
    "pass" : [0,0,0,1,1,1]
}

x = np.array(data["Study_hours"]).reshape(-1,1)
y = np.array(data["pass"])

x_train , x_test ,y_train, y_test = train_test_split(
    x, y, train_size=0.8 , random_state = 42
)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict([[5]])
#print(y_pred)

prob_pred = model.predict_proba([[5]])
#print(prob_pred)











from sklearn.linear_model import LogisticRegression


Data = {
    "Age" : [31,32,33,34,35,39,37,38],
    "Salary" : [10000,20000,30000,40000,50000,60000,70000,80000],
    "Pass"  : [0,0,0,0,1,1,1,1]
}

# Create feature matrix with shape (n_samples, n_features)
# Use column_stack to get shape (8,2) instead of (2,8)
x = np.column_stack((Data["Age"], Data["Salary"]))
y = np.array(Data["Pass"])

x_train , x_test, y_train , y_test = train_test_split(
    x,y, test_size= 0.25 , random_state = 42
    )

model = LogisticRegression()
model.fit(x_train, y_train)

customer = [[36, 60000]]
predict = model.predict(customer)
probability = model.predict_proba(customer)
#print("Predicted Class:", predict[0])
#print("Probability of not purchasing:", probability[0][0])
#print("Probability of purchasing:", probability[0][1])




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("ML Class/BC Dataset.csv")

#print(df.columns)
print(df.columns.tolist())
#print(df)

x = df[["Previous_Marks", "Study_Hours"]]
y = df["Pass"]

x_train , x_test, y_train , y_test = train_test_split(
    x,y, train_size=0.8 , random_state = 42
    )

model = LogisticRegression()
model.fit(x_train , y_train)

predict = model.predict(x_test)
probability = model.predict_proba(x_test)
print("Predicted data: ",predict)
print("Predicted probability of data: ",probability)
