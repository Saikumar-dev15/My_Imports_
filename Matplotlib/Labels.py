import matplotlib.pyplot as plt
import numpy as np 

x = np.array([2023, 2024, 2025, 2026])
y1 = np.array([15, 25, 32, 28])
y2 = np.array([17, 23, 38, 5])
y3 = np.array([13,15,17,18])

plt.title("Class Size", fontsize=25,
          family= "Arial",
          fontweight = "bold",
          color= "Darkred") 

plt.xlabel("year", fontsize=20,
           family= "Arial",
           fontweight = "bold",
           color="green")

plt.ylabel("year", fontsize=20,
           family= "Arial",
           fontweight = "bold",
           color="green")

plt.tick_params(axis="both",
                colors="purple")

plt.plot(x,y1)
plt.plot(x,y2)
plt.show()