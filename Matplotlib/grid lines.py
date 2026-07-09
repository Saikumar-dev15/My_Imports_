import matplotlib.pyplot as plt

#grid() = Helps make plots easier to read by adding reference line

x = [1,2,3,4,5]
y = [5,10,15,20,25]

plt.grid(axis = "y",
        linewidth = 2,
        color ="lightgray",
        linestyle = "dashed")

plt.plot(x,y)
plt.show()