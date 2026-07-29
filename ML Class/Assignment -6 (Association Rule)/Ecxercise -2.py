import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori , association_rules

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Jam'],
    ['Milk', 'Jam'],
    ['Milk', 'Bread', 'Jam'],
    ['Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread']
]


te =  TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(transactions, columns=te.columns_)
#print(df)

df = df.fillna(0)
df = df.astype(bool)

frequent_itemset = apriori(
    df, 
    min_support= 0.3,
    use_colnames= True
)

#print(frequent_itemset)


rules = association_rules(
    frequent_itemset, 
    metric= "confidence",
    min_threshold=0.6
)

#print(rules)

print("Top Rules")

for i, row in rules.iterrows():
    print("-"*60)
    print("Rule:", set(row['antecedents']), "->", set(row['consequents']))
    print("Support   :", round(row['support'],3))
    print("Confidence:", round(row['confidence'],3))
    print("Lift      :", round(row['lift'],3))
    
    


plt.figure(figsize=(6,5))
plt.scatter(rules["support"], rules["confidence"])

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
#plt.show()


top_rules = rules.sort_values(by="lift", ascending=False).head(10)

plt.bar(range(len(top_rules)), top_rules["lift"])
plt.xlabel("Top Rules")
plt.ylabel("Lift")
plt.title("Top Association Rules")
#plt.show()


print("\nBusiness Recommendations")

for i in range(len(top_rules)):
    print("If customer buys",
          list(top_rules.iloc[i]["antecedents"]),
          "recommend",
          list(top_rules.iloc[i]["consequents"]))
    