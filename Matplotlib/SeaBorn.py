import seaborn as sns 
import matplotlib.pyplot as plt 

tips = sns.load_dataset("tips")

sns.scatterplot(data=tips, x="total_bill", y="tip", color= "red")

plt.show()


#Line plot

sns.lineplot(x="size", y="tip", data=tips)
plt.show()