import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
# Apriori
from apyori import apriori

# datasets 
crimes = pd.read_csv("./Crime+Corpus/crime_cleaned_v3.csv").drop('Unnamed: 0',axis=1)
demographic = pd.read_csv('./Demographic/demographic_cleaned_v1.csv').drop('Unnamed: 0',axis=1)

# Sampling 1000 records
crime_sampled = crimes.sample(n=1000)

crime_sampled['Date'] = pd.to_datetime(crime_sampled['Date'])
print('Sample DataFrame Info')
print(crime_sampled.info())

# Changing Date into DateTime Format
crime_sampled['abs_date'] = crime_sampled['Date'].dt.strftime('%m-%d-%Y')

# Transactional Formatting
fortransaction = crime_sampled[['abs_date','Primary Type']]
fortransaction = fortransaction.reset_index(drop=True)
fortransaction['Ctype'] = fortransaction['Primary Type']

# lets see maximum number of patterns
store_lengths = {}
index = fortransaction.reset_index()['abs_date']
for idx in index:
    store_lengths[idx] = fortransaction.loc[idx].shape[0]
print('Item Sets by Date')
print(store_lengths)

# Creating Trasactionally Formatted Data ( Lists of Lists )
dictionary_of_items = {}
miss_count = 0 # counting crime types with no associations in the data ( meaning no same day, dual reports )
for date in crime_sampled['abs_date'].unique():
    if len(crime_sampled[crime_sampled['abs_date'] == date]['Primary Type']) >= 2:
        dictionary_of_items[date] = crime_sampled[crime_sampled['abs_date'] == date]['Primary Type'].to_list()
    else:
        miss_count += 1
        
print(f'Total Sample-Data: {crime_sampled.shape[0]}, Missed Associations {miss_count}')

# Transactions
transactions = []
for lt in pd.Series(dictionary_of_items.values()):
    transactions.append(lt)
print(transactions[:5])


# APRIORI Python
association_rules = apriori(transactions, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=4) 
association_results = list(association_rules)


print('PRINTING ASSOCIATION RULES')
for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " => " + items[1])
    #second index of the inner list
    print("Support: " + str(item[1])) #third index of the list located at 0th
    #of the third index of the inner list
    print("Confidence: " + str(item[2][0][2])) 
    print("Lift: " + str(item[2][0][3])) 
    print("=====================================")
