# Reading libraries 
import pandas as pd 
import numpy as np 
import requests
import missingno as msno 

# Visualisation libraries 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.io
from plotly.offline import iplot

plt.style.use('ggplot')

# Reading the Partially Cleaned DataFrames. (Version1s)
crimes = pd.read_csv('Crime+Corpus/crime_cleaned_v1.csv').drop('Unnamed: 0',axis=1)
demographic = pd.read_csv('Demographic/demographic_cleaned_v1.csv').drop('Unnamed: 0',axis=1)
environmental = pd.read_csv('Environmental/environmental_cleaned_v1.csv').drop('Unnamed: 0',axis=1)
police_stations = pd.read_csv('Police+Stations/police_stations_cleaned.csv').drop('Unnamed: 0',axis=1)
temperature_2018_jan = pd.read_csv('Temperature/Temperature_2018_Jan.csv').drop('Unnamed: 0',axis=1)
# correcting columns
temperature_2018_jan.columns = temperature_2018_jan.iloc[0].to_list()
temperature_2018_jan = temperature_2018_jan.drop(0,axis=0)

print('Converting crimes Date to datetime')
crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date will help us a lot later on


# Checking for Duplicates
print('Dataset Shape before drop_duplicate : ', crimes.shape)
crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('Dataset Shape after drop_duplicate: ', crimes.shape)

# Simple Bar Chart 
print('Aggregation of Crime Types over the years (2018-2022)')
plt.figure(figsize=(15,4))
sns.countplot(x='Primary Type',data=crimes,palette='seismic')
plt.xticks(rotation=90)
plt.xlabel('Total of Number of Recorded Crimes Types (2018-2022)')
plt.ylabel('Crime Types Recorded')
plt.title('Aggregation of Crime Types over the years (2018-2022)')
plt.show()

# Bar Charts grouped Yearly
print('Aggregation of Crime Types over the years (2018-2022)')
plt.figure(figsize=(15,7))
sns.countplot(x='Primary Type',data=crimes,hue='Year',palette='seismic')
plt.xticks(rotation=90)
plt.xlabel('Total of Number of Recorded Crimes Types (2018-2022)')
plt.ylabel('Crime Types Recorded')
plt.title('Aggregation of Crime Types over the years (2018-2022)')
plt.show()

# Spatial Heat-Maps 
# District
print('Spatial Map of Chicago with Crime Density by District')
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
data.plot(kind='scatter',x='X Coordinate', y='Y Coordinate', c='District', cmap='inferno_r')
plt.title('Spatial Map of Chicago with Crime Density by District')
plt.show()

# Community Area
print('Spatial Map of Chicago with Crime Density by Community Area')
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
data.plot(kind='scatter',x='X Coordinate', y='Y Coordinate', c='Community Area', cmap='seismic')
plt.title('Spatial Map of Chicago with Crime Density by Community Area')
plt.show()


# Joint Plot of Magnitude in Spatial Format
print('Joint Plot of Magnitude in Spatial Format')
plt.figure(figsize=(12,12))
sns.jointplot(x=data['X Coordinate'].values, y=data['Y Coordinate'].values, kind='hex',
              palette='inferno')
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()

# Subplots
# we are only looking at the top 6 crime types
print('Spatial Representation of the Crime Types Illustration of Arrests Made.')
cols = ['Date','Primary Type','Arrest','Domestic','District','X Coordinate','Y Coordinate']
multiple_crimes = crimes[cols]
multiple_crimes = multiple_crimes[multiple_crimes['Primary Type']\
                  .isin(['HOMICIDE','THEFT','NARCOTICS','WEAPONS VIOLATION'])]
# clean some rouge (0,0) coordinates
multiple_crimes = multiple_crimes[multiple_crimes['X Coordinate']!=0]
multiple_crimes.head()
g = sns.lmplot(x="X Coordinate",
               y="Y Coordinate",
               col="Primary Type",
               hue='Arrest',
               data=multiple_crimes.dropna(), 
               col_wrap=2, fit_reg=False, 
               sharey=False,
               	palette='Set1',
               scatter_kws={"marker": "D",
                            "s": 10},legend=True)
plt.suptitle('Spatial Representation of the Crime Types Illustration of Arrests Made.')
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()

# Feature Engineering
print('Feature Engineering on a Subset DataFrame of Crime')
subdf = crimes[['Date','Primary Type','Community Area','Community Area Name','District','Dristrict-Name','Year']]
print('Generating Month Numbers from Date')
subdf['Month'] = subdf['Date'].dt.month
print('Generating Days from Date')
subdf['Day'] = subdf['Date'].dt.day

# Tree Maps
# Community Area
print('Tree Map of Community Areas and Magnitude of Criminal Event Records')
treemap_df = subdf['Community Area'].value_counts() # subset for treemap

squarify.plot(sizes=treemap_df.values, label=treemap_df.index, alpha=0.8, )
plt.axis('off')
plt.title('Tree Map of Community Areas and Magnitude of Criminal Event Records')
plt.show()

# District Wise
print('Tree Map of Districts and Magnitude of Criminal Event Records')
treemap_df = subdf['District'].value_counts()
# plot it


squarify.plot(sizes=treemap_df.values, label=treemap_df.index, alpha=.8 )
plt.axis('off')
plt.title('Tree Map of Districts and Magnitude of Criminal Event Records')
plt.show()

# Interactive Tree Map
print('Interactive Tree Map of Crime Types Hirarchically Arranged in Nested Boxes which represent Community Areas.')
td = subdf.groupby(['Community Area Name','Primary Type']).count()['Community Area']
td = td.reset_index()

td["Chicago"] = "Chicago" # in order to have a single root node
fig = px.treemap(td, path=['Chicago', 'Community Area Name', 'Primary Type'], values='Community Area', width = 2000, height = 1000, color = "Community Area",
                 color_continuous_scale='delta')
fig.update_traces(root_color="black")
#fig.update_layout(margin = dict(t=50, l=25, r=25))

fig.show()


# Bar Chart of Crimes Recorded vs Arrest Made in the Coloration.
print('Districtwise Crimes Recorded Vs Arrests Made')
district_list = []

for i in range(0,crimes.shape[0]):
    district = crimes.iloc[i].District
    arrest = crimes.iloc[i].Arrest
    get_index = -1
    
    for j in range(0, len(district_list)):
        if (district_list[j][0] == district):
            get_index = j
            if arrest:
                district_list[j][1]+=1
            else:
                district_list[j][2]+=1
    
    if get_index == -1:
        if arrest:
            district_list.append([district, 1, 0])
        else:
            district_list.append([district, 0, 1])


get_district = pd.DataFrame(columns=['district','arrest','not_arrest'], data=district_list) 
get_district['Total'] = get_district.apply(lambda x: x.arrest+x.not_arrest, axis=1)

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 15))



sns.set_color_codes("pastel")
sns.barplot(x="Total", y="district", data=get_district,
            label="Total", color="b", orient='h')

sns.set_color_codes("muted")
sns.barplot(x="arrest", y="district", data=get_district,
            label="Arrest", color="b", orient='h')

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="",
       xlabel="Total cases vs arrest")
sns.despine(left=True, bottom=True)

plt.title('Districtwise Crimes Recorded Vs Arrests Made')
plt.show()

print('TEMPORAL PLOTS')
print('Monthly Aggregation of Crimes Recorded in the Year 2021')
ts = subdf[subdf['Year']==2021]
ts = pd.DataFrame(ts.sort_values(by='Date')['Date'].value_counts())

fig, ax = plt.subplots(figsize=(12,5))
sns.violinplot(x = ts.index.month,
                y = ts['Date'], 
                ax = ax)
plt.title('Monthly Aggregation of Crimes Recorded in the Year 2021')
plt.xlabel('Months')
plt.show()

# Temporal Rolling Sum of Crime Type 
print('Temporal Rolling Sum of Crime Types in Chicago from (2018 - 2022) ')
crimes.index = pd.DatetimeIndex(crimes.Date)
crimes_count_date = crimes.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=crimes.index.date, fill_value=0)
crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)
plo = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
plt.suptitle('Temporal Rolling Sum of Crime Types in Chicago from (2018 - 2022) ')
plt.subplots_adjust(top=0.958)
plt.show()

# For temperature vs crime
crime_2018_jan = subdf[(subdf['Month']==1) & (subdf['Year']==2018)]
time_temp = pd.DataFrame(crime_2018_jan.groupby(['Day']).count()['Date'])
time_temp['Temperature_Max'] = temperature_2018_jan['Maximum']
time_temp['Temperature_Min'] = temperature_2018_jan['Minimum']

print('Temperature (°C) vs Crimes Reported for the Month of Junuary 2018.')
plt.figure(figsize=(15,5))
sns.lineplot(data=time_temp, x='Temperature_Max',y='Date',markers='x',)
#sns.kdeplot(data=time_temp, x='Temperature_Min',y='Date',)
plt.legend()
plt.xticks(rotation=90)
plt.title('Temperature (°C) vs Crimes Reported for the Month of Junuary 2018.')
plt.xlabel('Temperature °C')
plt.ylabel('Number of Crimes Reported.')
plt.show()


# Folium Map of Community Areas.
print('Folium Map of Community Areas')
latlong = {}
for cm in crimes['Community Area Name'].unique()[:-1]:
	latlong[cm] = crimes[crimes['Community Area Name']==cm].dropna()[['Latitude','Longitude']].iloc[0].values

locs = pd.DataFrame(pd.Series(latlong)).reset_index
locs.columns = ['Commnunity Area','Coords']

from folium import plugins

m = folium.Map([41.8781, -87.6298], zoom_start=11)
for name, coord in locs.values:
    folium.Marker(
        location=[coord[0], coord[1]],
        popup=name,
    ).add_to(m)
m

# end.


