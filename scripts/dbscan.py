# DBSCAN - Code Lifted from Geeks for Geeks @ all rights belongs to geeks for geeks

# importing the library
from sklearn.neighbors import NearestNeighbors 
from sklearn.cluster import DBSCAN 
neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(X) # fitting the data to the object
distances,indices=nbrs.kneighbors(X) # finding the nearest neighbours

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show() # showing the plot

from sklearn.cluster import DBSCAN
# cluster the data into five clusters
dbscan = DBSCAN(eps = 8, min_samples = 4).fit(X) # fitting the model
labels = dbscan.labels_ # getting the labels

# Plot the clusters
plt.scatter(X[:, 0], X[:,1], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("Latitude") # X-axis label
plt.ylabel("Logitude") # Y-axis label
plt.show() # showing the plot

from sklearn.neighbors import NearestNeighbors # importing the library
neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(X) # fitting the data to the object
distances,indices=nbrs.kneighbors(X) # finding the nearest neighbours


# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show() # showing the plot


# Plotly Subplots
# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("DBscan", "Original"))

# Create two bar charts in the subplots
fig.add_trace(px.scatter(df, x="Longitude", y="Latitude", color="Dbscan_labels",).data[0], row=1, col=1)
fig.add_trace(px.scatter(df, x="Longitude", y="Latitude", color="SocioEconomic-Status").data[0], row=1, col=2)

# Update layout
fig.update_layout(title="DBScan vs Original Clusters",template="ggplot2", showlegend=False)


# Show the plot
fig.write_html('./Interactive_charts_html/dbscanvsorginal.html')
fig.show()

# STATIC Plots
fig , (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))

ax1.set_title('DBScan')
ax1.scatter(X[:, 0], X[:,1], c = labels, cmap= "tab10") 

ax2.set_title('Original')
ax2.scatter(X[:, 0], X[:, 1],c=df['SocioEconomic-Status'].values, cmap='tab10')

plt.legend()

