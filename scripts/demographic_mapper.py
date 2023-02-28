# Structuring Data and Mapping Demographic

# libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# datasets 
crimes = pd.read_csv("./Crime+Corpus/crime_cleaned_v2.csv").drop('Unnamed: 0',axis=1)
demographic = pd.read_csv('./Demographic/demographic_cleaned_v1.csv').drop('Unnamed: 0',axis=1)

# CATEGORIZING DEMOGRAPHIC QUANTITIES
df = demographic.copy(deep=True)

# define the weights for each variable in the composite index
w1 = 0.2 # weight for Percent of Housing Crowded
w2 = 0.3 # weight for Percent Households Below Poverty
w3 = 0.2 # weight for Percent Aged 16+ Unemployed
w4 = 0.2 # weight for Percent Aged 25+ Without High School Diploma
w5 = 0.1 # weight for Percent Aged Under 18 or Over 64

# create the composite index
df['Composite Index'] = (w1 * df['PERCENT OF HOUSING CROWDED']) + (w2 * df['PERCENT HOUSEHOLDS BELOW POVERTY']) + \
                        (w3 * df['PERCENT AGED 16+ UNEMPLOYED']) + (w4 * df['PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA']) + \
                        (w5 * df['PERCENT AGED UNDER 18 OR OVER 64'])

# divide the community areas into high, moderate, and low-income categories based on the composite index quartiles
low_income = df[df['Composite Index'] <= df['Composite Index'].quantile(0.25)]
moderate_income = df[(df['Composite Index'] > df['Composite Index'].quantile(0.25)) & (df['Composite Index'] <= df['Composite Index'].quantile(0.75))]
high_income = df[df['Composite Index'] > df['Composite Index'].quantile(0.75)]

# print the number of community areas in each category
print('Low-income areas:', len(low_income))
print('Moderate-income areas:', len(moderate_income))
print('High-income areas:', len(high_income))

# CREATING A MAPPER
# checking if there are overlaps 
li = low_income['COMMUNITY AREA NAME'].to_list()
hi = high_income['COMMUNITY AREA NAME'].to_list()
mi = moderate_income['COMMUNITY AREA NAME'].to_list()

print(set(li).intersection(set(hi)))
print(set(li).intersection(set(mi)))
print(set(mi).intersection(set(hi)))
print(set(mi).intersection(set(hi)))
# no intersections


# creating a mapper from the above descritization
demographic_mapper = {}
for df, label in zip([low_income, moderate_income, high_income], ['li','mi','hi']):
    for cas in df['COMMUNITY AREA NAME']:
        if label == 'li':
            demographic_mapper[cas] = 'Low Income'
        elif label == 'mi':
            demographic_mapper[cas] = 'Moderate Income'
        else:
            demographic_mapper[cas] = 'High Income'

crimes['SocioEconomic-Status']=crimes['Community Area Name'].map(demographic_mapper)crimes.to_csv('./Crime+Corpus/crime_cleaned_v3.csv') # saving
crimes.to_csv('./Crime+Corpus/crime_cleaned_v3.csv') # saving
