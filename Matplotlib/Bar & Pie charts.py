import matplotlib.pyplot as plt
import numpy as np 

categories = np.array(["Grains", "Fruits", "Vegetables", "protein", "Dairy"])
values = np.array([4,5,7,3,2])

#plt.bar(categories, values, color = "skyblue")

#plt.title("Daily consumption")
#plt.xlabel("Food")
#plt.ylabel("Quantity")

#plt.show()



#pie chart

Strength = ["Freshmen", "sophomeres", "Juniors", "Seniors"]
values = np.array([300, 250, 275,225])
colors = ["red", "yellow", "blue", "green"]

plt.pie(values, labels= Strength,
        autopct="%1.1f%%",
        colors = colors,
        explode=[0, 0, 0, 0.2],
        shadow = True,
        startangle= 90)

plt.show()