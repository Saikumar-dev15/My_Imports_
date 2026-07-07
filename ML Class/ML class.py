#print("Hello, Machine Learning!.... ")

#1.Write a line of code that prints your name and your hobby
#print("Laxmi sai Kumar", "Reading books")

#2.Can you all create Yor own two variables-name and your hobby
Name = "Laxmi sai kumar"
hobby = "Reading books"
#print(Name)
#print(hobby)

#3.Now try combining your namne and your hobby into one sentence
Name = "Laxmi sai kumar"
hobby = "Reading books"
#print(f"My Name is {Name} and My hobby is {hobby}")


#4. Write a small program that Stores your name , age, and hobby - and print them all in one Sentence.
Name = "Laxmi sai kumar"
Age = int(25)
hobby = "Reading books"
#print(f"My namee is {Name} and My age is {Age} and my favourite hobby is {hobby}")



Movies = ['RRR', 'BAHUBALI', 'DEVARA']
#print(Movies)
Movies.append('Bahubali-2')
#print(Movies)


Student = ("Rahul", 21 , "Python")
#print(Student[2])
#Student[1] = 22
#print(Student)

colors = {'Red', 'Orange', 'Blue', 'Yellow', 'Green'}
#print(colors)
colors.add('Red')
#print(colors)

Names = {"Rahul", "Priya", "Aman", "Aman", "Ajay"}
Names.add("Neha")
Names.remove("Aman")
#print(Names)

student = {
    "name": "Sai Kumar",
    "Age": 19,
    "course": "AIML"
}
#print(student["name"])
student["marks"] = 85
#print(student)
student.update({"Age": 21})
#print(student)
student.pop("course", None)
#print(student)
  
  



My_self = {
    "Name": "Laxmi Sai Kumar",
    "Age": 19,
    "Hobby": "I want to be Full Stack Developer",
}
#print(My_self)

number = 1
for number in range (1,6):
    nums = number*number 
    #print(nums)
    
    
#num= 1
#for num in range(1,11):
    #print(num)  
 
 
 
     
#print("Hello, World!")

def add(a,b):
    return a+b

def sub(a,b):
    return a-b

def mul(a,b):
    return a*b

def divi(a,b):
    return a/b

a= 4
b=2

#print("1:add")
#print("2.sub")
#print("3.mul")
#print("4.div")

#choose = input("select one operation {1/2/3/4}: ")

#if choose == "1":
    #print("Results: ", add(a,b))
#elif choose == "2":
    #print("Results: ", sub(a,b))
#elif choose == "3":
    #print("Results: ", mul(a,b))
#elif choose == "4" :
    #print("Results: ", divi(a,b)) 
#else :
    #print("Invalid error")
         
         

import numpy as np
marks = [75, 80, 90, 85, 95]
avg= np.mean(marks)
highest = np.max(marks)
new_marks = np.array(marks) + 10
new_Mul_Marks = np.array(marks) * 5
New_sub_marks = np.array(marks) - 2

#print(New_sub_marks)


#def Table(num):
#    for i in range (1,11):
       # print(f"{num} x {i} = {num*i}")
        
#Table(5)



import pandas as pd
import numpy as np
data = {
    "Name": ["Rahul", "Priya", "Aman", "Ajay"],
    "Age": [21, np.nan, 20, 23],
    "Marks": [85, 90, np.nan, 95]
}
df = pd.DataFrame(data)
###print(df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Age"] = df["Age"].fillna(df["Age"].median())
#df["City"] = df["City"].fillna(df["City"].mode()[0])
df.isnull().sum()
df_drop = df.dropna()
###print(df_drop)
df_fill_constant = df.fillna(0)
###print(df_fill_constant)


df_fill_mode = df.copy()
df_fill_mode["Age"] = df_fill_mode["Age"].fillna(df_fill_mode["Age"].mode()[0])
###print(df_fill_mode)

df.duplicated()             #to check for duplicates
df.duplicated().sum()       #to count the number of duplicates
df =  df.drop_duplicates()  #to eliminate duplicates
###print(df)



#Normalization 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Age" : [21, 25, 30, 35, 40],
    "Salary" : [50000, 60000, 70000, 80000, 90000]
}
df =pd.DataFrame(data)
#print(df)

scalar = MinMaxScaler()
df_scaled = scalar.fit_transform(df)

#Normalization using MinMaxScaler
df_scaled = pd.DataFrame(
    df_scaled,
    columns=df.columns
    )
#print(df_scaled)





#Standardization
#formula for standardization is (x - mean) / std

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Age" : [21, 25, 30, 35, 40],
    "Salary" : [50000, 60000, 70000, 80000, 90000]
    
}
df = pd.DataFrame(data)

scalar = StandardScaler()
scaled_data = scalar.fit_transform(df)

scaled_df = pd.DataFrame( 
    scaled_data,
    columns=df.columns
)
#print(scaled_df)

        
         
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1,1)
marks = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

model = LinearRegression()
model.fit(hours, marks)

predicted_marks = model.predict([[11]])
#print("Predicted marks for 11 hours of study:", predicted_marks)

#plt.scatter(hours, marks, color='blue', label='Actual Marks')
#plt.plot(hours, model.predict(hours), color='red', label='Regression Line')  
#plt.xlabel('Hours of Study')
#plt.ylabel('Marks')
#plt.title('Linear Regression: Hours of Study vs Marks')
#plt.show()




#this code will train and test a linear regression model on a dataset of hours studied and marks obtained. It will also predict the marks for 11 hours of study and visualize the results using a scatter plot and regression line.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1,1)
marks = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

data = pd.DataFrame({'Hours': hours.flatten(), 'Marks': marks})
#print(data)

x = data[['Hours']]   #independent variable
y = data['Marks']      #dependent variable

x_train , x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
    )

x_train, x_test, y_train, y_test

model = LinearRegression()
model.fit(x_train, y_train)

#print("Slope(m): ", model.coef_[0])
#print("Intercept(c): ", model.intercept_)

y_pred = model.predict(x_test)
#print("Predicted marks for test data: ", y_pred)
#print("Actual marks for test data: ", y_test.values)

#plt.scatter(x_test, y_test, color='blue', label='Actual Marks')
#plt.plot(x_test, y_pred, color='red', label='Regression Line')
#plt.xlabel('Hours of Study')
#plt.ylabel('Marks')
#plt.title('Linear Regression: Hours of Study vs Marks')
#plt.legend()
#plt.show()



#logistic  regression
from sklearn.linear_model import LogisticRegression

data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'attendance' : [40,50,55,60,70,75,80,85,90,95],
    'passed_exam': [0,0,0,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

x = df[['study_hours', 'attendance']]
y = df[['passed_exam']]

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)
print(y_pred)


