import seaborn as sns 
import matplotlib.pyplot as plt 

tips = sns.load_dataset("tips")

#sns.scatterplot(data=tips, x="total_bill", y="tip", color= "red")

#plt.show()


#Line plot

#sns.lineplot(x="size", y="tip", data=tips)

#sns.countplot(data=tips, x="sex")

#sns.countplot(
#    x="day",
#    data=tips,
#    hue="sex",
#    palette="Set2"                     #Colors
#    )
#.show()


#sns.histplot(
#    data=tips,
#    x="total_bill", bins=10, color="skyblue", edgecolor= "black", kde=True,  alpha=0.8,                 
#)
#plt.show()

sns.boxplot(data=tips , y="total_bill", 
            hue="sex",
            palette="Set2")
#plt.figure(figsize=(8,5))
plt.show()


#HeatMap
tips = sns.load_dataset("tips")

corr = tips.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap of Tips Dataset")

plt.show()




#Pairplot
sns.pairplot(tips, hue="sex")

plt.show()