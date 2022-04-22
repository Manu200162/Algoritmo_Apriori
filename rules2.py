import numpy as np
import pandas as pd
from apyori import apriori

store_data= pd.read_csv('Day1.csv',header=None)

# print(store_data)
# print(store_data.shape)

records = []
for i in range (0,22):
    records.append([str(store_data.values[i,j]) for j in range(0,6)])

association_rules = apriori(records, min_support=0.5,min_confidence=0.7,min_lift=1.2,min_length=2)
association_results = list(association_rules)

# min support = Num of transactions containing "Milk,Bread,Butter" divided by total Num of transactions = 11/22 = 0.5
# confidence = 84.6% of transactions that contains"Milk, Bread" contains also "Butter"
# Butter is 1.241 times more likely to be bought by the customers who buy both "Milk,Butter" compared to the default likelihood sale of "Butter".

def inspect(output):
    lhs         = [tuple(result[2][0][0]) for result in output]
    rhs         = [tuple(result[2][0][1]) for result in output]
    support    = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift       = [result[2][0][3] for result in output]
    return list(zip(lhs, rhs, support, confidence, lift))
output_DataFrame = pd.DataFrame(inspect(association_results), columns = ['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'])

from tabulate import tabulate
print(tabulate(output_DataFrame, headers = 'keys', tablefmt = 'psql'))
print(association_results)
