# scatter graph = shows the relationship between two variables 
#                 Helps to identify a correlation (+,-, None)
#                 Ex: Study hours vs Test Scores

import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0,1,1,2,3,4,5,6,7,8])             #hours studied
y1 = np.array([55,60,65,74,79,98,97,87,81,93])  #grade

x2 = np.array([0,1,2,2,3,4,7,5,6,7,8])             #hours studied
y2 = np.array([55,87,80,95,74,79,68,87,77,51,93]) 

plt.scatter(x1,y1,color="skyblue",
            alpha = 0.5,
            s= 200)

plt.scatter(x2,y2,color="red",
            alpha = 0.5,
            s= 200)

plt.title("Test Score")
plt.xlabel("Hours Studied")
plt.ylabel("Grade")

plt.show()