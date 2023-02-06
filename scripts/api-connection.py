
import pandas as pd 
import requests


URL = 'https://data.cityofchicago.org/resource/crimes.json' # API Endpoint, Chicago Crimes 

response = requests.get(URL) # GET, will fetch the JSON Data Format from the 
print(response) # To check if there's a successful transaction of Data

# For Json Text
print(response.json())

