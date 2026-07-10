import matplotlib.pyplot as plt
import numpy as np

scores = np.random.normal(loc=80, scale=10, size= 100)
scores = np.clip(scores, 0,100)
plt.hist(scores, bins=10,
                 color= "lightgreen",
                 edgecolor="black",)
plt.show()




#Home Work

Marks = np.array([45, 50, 55, 60, 62, 65, 68, 70, 72, 75, 78,80 ,85,90,92,95])

plt.hist(Marks, bins=6,
                color="lightgreen",
                edgecolor="black",
                alpha=0.9,)

plt.title("Exam Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()