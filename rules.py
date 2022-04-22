import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


Data= pd.read_csv('Market_Basket_Optimisation.csv',header = None)

transacts = [] 

for i in range(0,7501):
     transacts.append([str(Data.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rule = apriori(transactions = transacts, 
               min_support = 0.003, 
               min_confidence = 0.2, min_lift = 3, 
               min_length = 2, 
               max_length = 2)

results = list(rule) # returns a non-tabular output
# putting output into a pandas dataframe
def inspect(output):
    lhs         = [tuple(result[2][0][0])[0] for result in output]
    rhs         = [tuple(result[2][0][1])[0] for result in output]
    support    = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift       = [result[2][0][3] for result in output]
    return list(zip(lhs, rhs, support, confidence, lift))
output_DataFrame = pd.DataFrame(inspect(results), columns = ['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'])

from tabulate import tabulate
print(tabulate(output_DataFrame, headers = 'keys', tablefmt = 'psql'))