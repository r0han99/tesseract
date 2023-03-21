# import packages 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

# Load Dataset
crimes = pd.read_csv('Crime+Corpus/crime_cleaned_v3.csv').drop('Unnamed: 0',axis=1)

# Decretizing Date
crimes['Date'] = pd.to_datetime(crimes['Date'])
crimes['Hour'] = crimes['Date'].dt.hour
crimes['Day'] = crimes['Date'].dt.day
crimes['Month'] = crimes['Date'].dt.month

# Saving 
crimes.to_csv('./Crime+Corpus/crimes_binary_v1.csv')