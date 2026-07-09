import matplotlib 

#print(matplotlib.__version__)

import matplotlib.pyplot as plt
import numpy as np 

x = np.array([2023, 2024, 2025, 2026])
y = np.array([15, 25, 32, 28])

plt.plot(x,y)
plt.show()