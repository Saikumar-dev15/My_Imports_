#Advance classification Algrothims: Ml which only identify only patterns in dataset.
# we need this because logistic reg will assume inputfeatures.

#Gradient Boosting : It is most power ful ML  and it is Ensemble. It will Gather all weak Tree and Merge with Strong Trees.
#Boosting : Training into strong trees


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

df = load_iris()
x = df.data
y = df.target

x_train , x_test, y_train , y_test =  train_test_split(
    x, y , test_size=0.2 , random_state=42
)


model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1, 
    random_state=42
)


model.fit(x_train, y_train)

pred = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, pred))
print("\n Clasification Report: ")
print(classification_report(y_test, pred))

