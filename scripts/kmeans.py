# KMEANS SCRIPT 
# Load required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode()
from datetime import datetime


plt.style.use('seaborn')

# READ Data
crimes = pd.read_csv('./Crime+Corpus/crime_cleaned_v3.csv').drop('Unnamed: 0',axis=1)



df = crimes[['Longitude','Latitude','SocioEconomic-Status']].sample(1000)

# Drop any missing values
df.dropna(inplace=True)

# Encode the labels using label encoder
label_encoder = LabelEncoder()
df['SocioEconomic-Status'] = label_encoder.fit_transform(df['SocioEconomic-Status'])


# Create a dictionary to map the encoded labels to real strings
label_dict = {
    0: 'Low Income',
    1: 'Moderate Income',
    2: 'High Income'
}

# Create a feature matrix
X = df[['Longitude', 'Latitude']].values

# Scale the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)


# Map the cluster labels to real strings
labels = [label_dict[label] for label in y_pred]


# Plot the results
plt.figure(figsize=(10,7))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10')
plt.title('K-means clustering on Chicago crime dataset')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.legend(labels=list(label_dict.values()),bbox_to_anchor=(1.05, 1.0), loc='upper left')
legend_elements = [plt.Line2D([0], [1], marker='o', color='green', label='{}'.format(label_dict[la]), markerfacecolor=c) for i,la,c in zip(range(kmeans.n_clusters), np.unique(kmeans.labels_),['cyan','dodgerblue','darkgoldenrod'])]
plt.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

# Plotly Script 
fig = px.scatter(df, x="Longitude", y="Latitude", color="kmeans_labels",title='Kmeans Clustering of Spatial Data')
fig.show()

# DF for Plotly
df['o_label'] = df['SocioEconomic-Status'].map(label_dict)
df['k_label'] = df['kmeans_labels'].map(label_dict)
df


# Create subplots - Plotly
fig = make_subplots(rows=1, cols=2, subplot_titles=("Kmeans", "Original"))

# Create two bar charts in the subplots
fig.add_trace(px.scatter(df, x="Longitude", y="Latitude", color="kmeans_labels", hover_name="k_label").data[0], row=1, col=1)
fig.add_trace(px.scatter(df, x="Longitude", y="Latitude", color="SocioEconomic-Status",hover_name='o_label').data[0], row=1, col=2)


fig.update_layout(title="Kmeans vs Original Clusters",
                  paper_bgcolor='#1C4586', 
                  template='ggplot2',
                  showlegend=False, title_font_family="Chakra Petch",
    title_font_color="white",)
fig.update_xaxes(color='white') 
fig.update_yaxes(color='white') 

# Show the plot
fig.write_html('./Interactive_charts_html/kmeansvsorginal_k4.html')
fig.show()


# Silhouette
visualizer = KElbowVisualizer(kmeans, k=(2,10),metric='silhouette')

visualizer.fit(X) # Fit the data to the visualizer
visualizer.show() # Finalize and render the figure

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    '''
    Create KMeans instances for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='jet', ax=ax[q-1][mod])
    visualizer.fit(X) 
    
# 3 Dimensional KMEANS ANALYSIS
data = pd.read_csv('./Crime+Corpus/crime_cleaned_v3.csv').drop('Unnamed: 0',axis=1)
# dropping unrelated columns 
data.drop(['X Coordinate', 'Y Coordinate', 'Updated On', 'Location', 'Beat'], axis=1, inplace=True)

# Manufacturing time column
data['Date'] = pd.to_datetime(data.Date) 
data['date'] = [d.date() for d in data['Date']]
data['time'] = [d.time() for d in data['Date']]

data['time'] = data['time'].astype(str)
empty_list = []
for timestr in data['time'].tolist():
    ftr = [3600,60,1]
    var = sum([a*b for a,b in zip(ftr, map(int,timestr.split(':')))])
    empty_list.append(var)
    
data['seconds'] = empty_list

# Subsetting 
sub_data = data[['Ward', 'IUCR', 'District']]
sub_data = sub_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
sub_data['IUCR'] = sub_data.IUCR.str.extract('(\d+)', expand=True).astype(int)
sub_data.head()

# Recursive KMeans for finding optimal model
N = range(1, 20)
kmeans = [KMeans(n_clusters=i,n_init='auto') for i in N]
score = [kmeans[i].fit(sub_data).score(sub_data) for i in range(len(kmeans))]

# plotting the elbow
pl.figure(figsize=(8,5))
pl.plot(N,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show() 

# fitting and predicting
km = KMeans(n_clusters=4)
km.fit(sub_data)
y = km.predict(sub_data)
labels = km.labels_
sub_data['Cluster'] = y

sub_sample = sub_data.sample(1000)

# 3 dimensional plot for k=4
fig = px.scatter_3d(sub_sample, x='IUCR', y='Ward', z='District',
              color='Cluster', symbol='Cluster' )
fig.update_layout(title="3 Dimensional Clusters K=4",paper_bgcolor='#01143D', 
                  template='plotly_dark', showlegend=False)

fig.write_html("./Interactive_charts_html/3dKmean4.html")
fig.show()

# Modifying data
sub_data['IUCR'] = (sub_data['IUCR'] - sub_data['IUCR'].min())/(sub_data['IUCR'].max()-sub_data['IUCR'].min())
sub_data['Ward'] = (sub_data['Ward'] - sub_data['Ward'].min())/(sub_data['Ward'].max()-sub_data['Ward'].min())
sub_data['District'] = (sub_data['District'] - sub_data['District'].min())/(sub_data['District'].max()-sub_data['District'].min())

# elbow for modified
pl.plot(N,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


km = KMeans(n_clusters=3)
km.fit(sub_data)
y = km.predict(sub_data)
labels = km.labels_
sub_data['Clusters'] = y

# 3 dimensional plot for k=3
fig = px.scatter_3d(sub_sample, x='IUCR', y='Ward', z='District',
              color='Clusters', symbol='Clusters' )
fig.update_layout(title="3 Dimensional Clusters K=3",paper_bgcolor='#01143D', 
                  template='plotly_dark', showlegend=False)
fig.write_html("./Interactive_charts_html/3dKmean3.html")
fig.show()


## n-Dimensional KMEANS

# Load the Chicago crime dataset into a pandas DataFrame
chicago_crime_df = crimes.copy(deep=True)

# Convert the Date/Time column to a datetime object
chicago_crime_df['Date'] = pd.to_datetime(chicago_crime_df['Date'])

# Extract the hour of the day from the datetime object
chicago_crime_df['Hour'] = chicago_crime_df['Date'].apply(lambda x: x.hour)

# Create a pivot table of the crime counts by hour of the day and crime type
crime_counts = pd.pivot_table(chicago_crime_df, values='ID', index=['Primary Type'], columns=['Hour'], aggfunc=np.count_nonzero, fill_value=0)

# Apply K-means clustering to the crime counts data
kmeans = KMeans(n_clusters=5)
kmeans.fit(crime_counts)

# Get the cluster labels for each crime type
labels = kmeans.predict(crime_counts)

# Add the cluster labels as a new column in the original DataFrame
crime_counts['Cluster'] = labels

# Print the number of crime types in each cluster
print(crime_counts['Cluster'].value_counts())

# Elbow on the above
visualizer = KElbowVisualizer(kmeans, k=(2,10))
visualizer.fit(X) # Fit the data to the visualizer
visualizer.show() # Finalize and render the figure


x = crime_counts[crime_counts.columns[:-1]]

# PRINCIPAL COMPONENT ANALYSIS to Reduce Dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

# Plotting reduced Df
X = principalDf.values
# Plot the results
plt.figure(figsize=(10,7))
plt.scatter(X[:, 0], X[:, 1], c=crime_counts['Cluster'], cmap='tab10')
plt.title('K-means clustering on Chicago crime dataset')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
#plt.legend(labels=list(label_dict.values()),bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

# 3 dimensional version of the same 

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

fig = px.scatter_3d(principalDf, x='principal component 1', y='principal component 2', z='principal component 3',
              color=crime_counts['Cluster'], symbol=crime_counts['Cluster'] )
fig.update_layout(title="3 Dimensional Clusters",template="plotly_dark", showlegend=False)
#fig.write_html("./Plotly_Chart_Html/Phase-2-3DKmean.html")
fig.show()

