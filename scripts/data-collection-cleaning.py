# reading libraries 
import pandas as pd 
import numpy as np 
import requests
import missingno as msno 

# Formats json formated text to Pandas DataFrame. 
def create_dataframe_from_json(json):
    json_items = []
    for item in range(len(json)):
        json_items.append(pd.DataFrame(pd.Series(resp.json()[item])).T)

    return pd.concat(json_items, axis=0)

# Fetches Data from API and creates a Pandas DataFrame 
def create_police_stations_data(api_end_point):
    required_keys = ['district', 'district_name', 'address', 'city', 'state', 'zip', 'website', 'x_coordinate', 'y_coordinate', 'latitude', 'longitude', 'location', ]
   # Get Data
    resp = requests.get(api_end_point)
  
  
    police_stations = create_dataframe_from_json(resp.json())
    return police_stations 

# Data Collection and Organisation.
police_stations_api_endpoint = 'https://data.cityofchicago.org/resource/z8bn-74gv.json?$query=SELECT%0A%20%20%60district%60%2C%0A%20%20%60district_name%60%2C%0A%20%20%60address%60%2C%0A%20%20%60city%60%2C%0A%20%20%60state%60%2C%0A%20%20%60zip%60%2C%0A%20%20%60website%60%2C%0A%20%20%60phone%60%2C%0A%20%20%60fax%60%2C%0A%20%20%60tty%60%2C%0A%20%20%60x_coordinate%60%2C%0A%20%20%60y_coordinate%60%2C%0A%20%20%60latitude%60%2C%0A%20%20%60longitude%60%2C%0A%20%20%60location%60%2C%0A%20%20%60%3A%40computed_region_rpca_8um6%60%2C%0A%20%20%60%3A%40computed_region_vrxf_vc4k%60%2C%0A%20%20%60%3A%40computed_region_6mkv_f3dw%60%2C%0A%20%20%60%3A%40computed_region_bdys_3d7i%60%2C%0A%20%20%60%3A%40computed_region_43wa_7qmu%60%2C%0A%20%20%60%3A%40computed_region_awaf_s7ux%60'

# API CSV ENDPOINTS
CRIME_CSV_ENDPOINT = 'https://data.cityofchicago.org/resource/ijzp-q8t2.csv'
DEMOGRAPHIC_CSV_ENDPOINT = 'https://data.cityofchicago.org/resource/kn9c-c2s2.csv'

# reading data
police_stations = create_police_stations_data(police_stations_apiendpoint)
crime = pd.read_csv(CRIME_CSV_ENDPOINT).drop('Unnamed: 0',axis=1)
environmental = pd.read_csv('Environmental/Map_of_Street_Lights_All_Out.csv')
demographic = pd.read_csv(DEMOGRAPHIC_CSV_ENDPOINT) 


# missing value visualisation 
msno.matrix(police_stations)
msno.matrix(crime)
msno.matrix(environmental)

# Quantified Look at the Missing Values
print('POLICE STATIONS NULL VALUES')
print(police_stations.isna().sum())
print('CRIMES NULL VALUES')
print(crime.isna().sum())
print('ENVIRONMENTAL NULL VALUES')
print(environmental.isna().sum())
print('DEMOGRAPHIC NULL VALUES')
print(demographic.isna().sum())

# RELATIONAL LINKAGE BETWEEN CRIMES AND DEMOGRAPHIC DATA
community_area_num = crime['Community Area'].fillna(0).astype(int)

# creating a mapper.
mapper = dict(demographic[['Community Area Number','COMMUNITY AREA NAME']].values)

# Compensating the 0 that we filled above 
mapper[0] = 'No Area' # this can be addressed later.

crime['Community Area'] = community_area_num
print('Mapped Community Areas')
crime['Community Area Name'] = crime['Community Area'].map(mapper)

# RELATIONAL LINKAGE BETWEEN CRIMES AND POLICE_STATIONS On DISTRICT
police_stations['district'] = police_stations['district'].replace('Headquarters','0')
police_stations['district'] = police_stations['district'].astype('int')

compound_mapper = police_stations[['district','district_name','zip']].iloc[1:]
compound_mapper.head(2)

district_mapper  = dict(compound_mapper[['district','district_name']].values)

# adding 31 to the mapper
district_mapper[31] = 'West Northfield'
district_mapper

# Type casting
crime['District'] = crime['District'].astype('int')

# District Mapping
print('Mapped District Names')
crime['District'] = crime['District'].astype('int')

# SAVING PARTIALLY CLEANED AND LINKED DATASETS TO FOLDERS as Version1s.
print("Datasets Saved.!")
crime.to_csv('Crime+Corpus/crime_cleaned_v1.csv')
environmental.to_csv('Environmental/environmental_cleaned_v1.csv')
demographic.to_csv('Demographic/demographic_cleaned_v1.csv')
police_stations.to_csv('Police+Stations/police_stations_cleaned.csv')
