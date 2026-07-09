import matplotlib.pyplot as plt
import numpy as np 

x = np.array([2023, 2024, 2025, 2026])
y1 = np.array([15, 25, 32, 28])
y2 = np.array([17, 23, 38, 5])

#plt.plot(x,y1, marker=".", 
#              markersize =30,            #ms
#              markerfacecolor= "orange", #mfc
#              markeredgecolor= "orange",  #mec
#              linestyle="solid",
#              linewidth=4,
#              color = "Darkblue") 

line_style = dict(marker=".", 
              markersize =30,            #ms
              markerfacecolor= "orange", #mfc
              markeredgecolor= "orange",  #mec
              linestyle="solid",
              linewidth=4,
              color = "Darkblue") 
plt.plot(x,y1, **line_style)
plt.plot(x,y2, **line_style)
plt.show()