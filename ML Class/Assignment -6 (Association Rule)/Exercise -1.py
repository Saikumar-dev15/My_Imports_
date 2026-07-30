import pandas as pd
import matplotlib.pyplot as plt
import time

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Milk","Bread","Butter"],
    ["Milk","Eggs","Bread"],
    ["Bread","Butter","Jam"],
    ["Milk","Bread","Eggs"],
    ["Bread","Butter"],
    ["Milk","Butter"],
    ["Bread","Jam"],
    ["Milk","Bread","Butter","Eggs"],
    ["Milk","Cheese"],
    ["Bread","Butter","Cheese"],
    ["Milk","Bread"],
    ["Milk","Jam"],
    ["Butter","Eggs"],
    ["Milk","Bread","Butter"],
    ["Cheese","Bread"],
    ["Milk","Bread","Jam"],
    ["Eggs","Bread"],
    ["Milk","Butter","Cheese"],
    ["Milk","Bread","Butter","Jam"],
    ["Juice","Bread"],
    ["Juice","Milk"],
    ["Juice","Bread","Butter"],
    ["Coffee","Milk"],
    ["Coffee","Bread"],
    ["Coffee","Milk","Bread"],
    ["Milk","Eggs","Cheese"],
    ["Bread","Butter","Eggs"],
    ["Milk","Bread","Juice"],
    ["Milk","Coffee","Butter"],
    ["Bread","Cheese","Jam"]
]


te = TransactionEncoder()

encoded = te.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded, columns=te.columns_)

#print("One-Hot Encoded Data")
#print(df.head(5))



#print("\nNumber of Transactions :", len(transactions))
#print("Number of Products :", len(df.columns))
#print("Products :", list(df.columns))


#print("\n========== APRIORI ==========")

support_values = [0.2, 0.3, 0.4]

for s in support_values:

    print("\nMinimum Support:", s)

    start = time.time()

    frequent_itemsets = apriori(
        df,
        min_support=s,
        use_colnames=True
    )

    end = time.time()

    print(frequent_itemsets)

    print("Number of Frequent Itemsets :", len(frequent_itemsets))
    print("Execution Time :", round(end-start,6), "seconds")



frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets")
print(frequent_itemsets)



rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print("\nAssociation Rules")

print(rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
])



top_rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("\nTop Rules")

print(top_rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
].head())




plt.figure(figsize=(8,5))

plt.bar(
    range(len(top_rules.head(10))),
    top_rules["lift"].head(10)
)

plt.xlabel("Top Rules")
plt.ylabel("Lift")
plt.title("Apriori - Top Association Rules")

plt.show()


print("\nBusiness Recommendations\n")

for i in range(len(top_rules.head(5))):

    print("---------------------------")

    print(
        "Buy :",
        list(top_rules.iloc[i]["antecedents"])
    )

    print(
        "Recommend :",
        list(top_rules.iloc[i]["consequents"])
    )

    print(
        "Support :",
        round(top_rules.iloc[i]["support"],2)
    )

    print(
        "Confidence :",
        round(top_rules.iloc[i]["confidence"],2)
    )

    print(
        "Lift :",
        round(top_rules.iloc[i]["lift"],2)
    )
    
    
    

print("\n========== ECLAT ==========")

start = time.time()


item_tid = {}

for tid, transaction in enumerate(transactions):

    for item in transaction:

        if item not in item_tid:
            item_tid[item] = set()

        item_tid[item].add(tid)


min_support = 0.3
min_count = int(min_support * len(transactions))

frequent_items = []

for item in item_tid:

    if len(item_tid[item]) >= min_count:

        frequent_items.append(
            (item, len(item_tid[item]))
        )

end = time.time()

print("\nFrequent Items")

for item in frequent_items:

    print(item)

print("\nNumber of Frequent Itemsets :", len(frequent_items))
print("Execution Time :", round(end-start,6), "seconds")



print("\n========== COMPARISON ==========")

# Apriori values
start = time.time()

apriori_items = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

end = time.time()

apriori_time = end - start

# ECLAT values
start = time.time()

item_tid = {}

for tid, transaction in enumerate(transactions):

    for item in transaction:

        if item not in item_tid:
            item_tid[item] = set()

        item_tid[item].add(tid)

eclat_items = []

for item in item_tid:

    if len(item_tid[item]) >= min_count:

        eclat_items.append(item)

end = time.time()

eclat_time = end - start


print("\nAlgorithm Comparison")
print("------------------------------------------")
print("Apriori")
print("Frequent Itemsets :", len(apriori_items))
print("Execution Time :", round(apriori_time,6))
print("Ease of Implementation : Easy")

print()

print("ECLAT")
print("Frequent Itemsets :", len(eclat_items))
print("Execution Time :", round(eclat_time,6))
print("Ease of Implementation : Moderate")



print("\nBusiness Recommendations")

print("--------------------------------")
print("1. Place Milk and Bread close together.")

print("2. Offer Butter as a recommendation when customers buy Bread.")

print("3. Create combo offers for Milk, Bread and Butter.")

print("4. Keep frequently purchased products near each other.")

print("5. Use association rules for online product recommendations.")