#If a customer buys item A, What other items are they likely to buy

#Example 
# if x --> happen then y --> is gone to  happen
#if a person buy bread then hw will go for Butter

# Support : if 20 0ut of 100 transactions contain bread and butter
#support  = 20/100 

# Confidence = If 25 Customers bought bread and 20 of them also bought butter,
#Confidence = 20/25 = 80%

# Lift =  Confidence / Support
# > 1 ----> Positive relationship 
# = 1 ----> No Relationship
# < 1 ----> Negative Relationship


# Apriori Algorithm
# | Transcation id    |  Items       |
# |  T1               | Butter, Milk |
# |  T2               | Butter,, Bread , Milk | 
# |  T3               | Milk , Bread          |
# |  T4               | Bread , Milk          |


import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = [
    ['Bread' , 'Milk'],
    ['Bread', 'Diaper', 'Beer', 'Eggs'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk', 'Diaper', 'Cola']
]

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_arrays = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_arrays, columns = te.columns_)
#print(df)

frequent_itemset = apriori(df, min_support=0.4, use_colnames=True)
#print(frequent_itemset)

rules = association_rules(frequent_itemset, metric ="confidence", min_threshold=0.6)
#print(rules)





# ECLAT Algorithm 
# | Item     |  Transcation Ids   |
# | Bread    | T1, T2, T3, T4, T5 |
# | Milk     | T1, T2, T3         | 
# | Butter   | T2, T3, T4         |


from mlxtend.frequent_patterns import apriori

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Jam'],
    ['Milk', 'Jam'],
    ['Milk', 'Bread', 'Jam']
]

te = TransactionEncoder()
te_arrays = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_arrays, columns = te.columns_)
#print(df)

frequent_itemsets = apriori(
    df, min_support=0.3, use_colnames=True
)

#print(frequent_itemsets)


from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets, 
    metric = "confidence",
    min_threshold= 0.6
)
#print(rules)

frequent_itemsets_eclat = apriori(
    df , 
    min_support= 0.3,
    use_colnames= True,
    max_len = 2
)

print(frequent_itemsets_eclat)
