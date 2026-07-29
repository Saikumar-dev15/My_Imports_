import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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

start = time.time()
frequent_itemset = apriori(
    df, 
    min_support= 0.3,
    use_colnames= True
)
end = time.time()
#print(frequent_itemset)
Time_of_apriori = end - start 
#print(f"Time taken by Apriori: {Time_of_apriori}")


rules = association_rules(
    frequent_itemset, 
    metric= "confidence",
    min_threshold=0.6
)

#print(rules)

start = time.time()
frequent_itemsets_eclat = apriori(
    df , 
    min_support= 0.3,
    use_colnames= True,
    max_len = 2
)
end = time.time()
#print(frequent_itemsets_eclat)
time_of_Eclat = end - start
#print(f"Time Taken by Eclat: {time_of_Eclat}")

frequent_itemsets = len(frequent_itemset)
#print("Number of Frequent Itemsets by Apriori:", frequent_itemsets) 


frequent_itemsets_eclats = len(frequent_itemsets_eclat)
#print("Number of Frequent Itemset by ECLAT: ", frequent_itemsets_eclats)


result =pd.DataFrame({
    "Modes" : ["Apriiori" , "ECLAT"],
    "Number of frequent itemsets" : [frequent_itemsets, frequent_itemsets_eclats],
    "Execution time" : [Time_of_apriori, time_of_Eclat],
    "Ease of implementation" : ["Easy" , "Moderate"]
}) 

print(result)

if Time_of_apriori <= time_of_Eclat :
    print("Apriori is More Efficient")
else :
    print("ECLAT is More Efficient")