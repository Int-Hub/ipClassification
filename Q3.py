import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("coursework1.csv")

# Define data frame
df = DataFrame(data, columns=['sourceIP', 'destIP'])
# print(df)

# Defining data frame array for destIP
df_destIP = df['destIP'].value_counts()
df_destIP = df_destIP.tolist()
df_destIP = pd.DataFrame(df_destIP, columns=["Destination_IP Frequencies"])
df_destIP["Zeros"] = 0

# Defining data frame array for sourceIP
df_sourceIP = df['sourceIP'].value_counts()
df_sourceIP = df_sourceIP.tolist()
df_sourceIP = pd.DataFrame(df_sourceIP, columns=["Source_IP Frequencies"])
df_sourceIP["Zeros"] = 0

# Initialising kmeans for source and destination IP

kmeans_sip = KMeans(n_clusters=4).fit(df_sourceIP)
centroids_sip = kmeans_sip.cluster_centers_
# print(centroids_sip)

kmeans_dip = KMeans(n_clusters=2).fit(df_destIP)
centroids_dip = kmeans_dip.cluster_centers_
#print(centroids_dip)

# Plot showing data clusters for Source and Destination Entries
plt.scatter(df_sourceIP["Source_IP Frequencies"], df_sourceIP["Zeros"], c=kmeans_sip.labels_.astype(float), s=50,
            alpha=0.5)
plt.scatter(centroids_sip[:, 0], centroids_sip[:, 1], c='black', s=50)
plt.ylabel('Frequency')
plt.xlabel('IP address Entries')
plt.title('Source IP clusters')
plt.show()

plt.scatter(df_destIP["Destination_IP Frequencies"], df_destIP["Zeros"], c=kmeans_dip.labels_.astype(float), s=50,
            alpha=0.5)
plt.scatter(centroids_dip[:, 0], centroids_dip[:, 1], c='c', s=50)
plt.ylabel('Frequency')
plt.xlabel('IP address Entries')
plt.title('Destination IP clusters')
plt.show()

# Create the Elbow plot for source_IP

# Define Within_clusters_Sum_of_Squares to find optimal K cluster from 1-8
wcss = []
K = range(1, 8)

# Initialise elbow plot for sourceIP
for k in K:
    kmeans_sip = KMeans(n_clusters=k).fit(df_sourceIP)
    wcss.append(sum(np.min(cdist(df_sourceIP,
                                 kmeans_sip.cluster_centers_,
                                 'euclidean') ** 2, axis=1)) / df_sourceIP.shape[0])
# Plot the elbow
plt.figure()
plt.plot(K, wcss)
plt.plot(K, wcss, 'k')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Within clusters Sum of Squares')
plt.title('The Elbow Method highlighting Optimal Number of Clusters (SourceIP)')
plt.show()

# Initialise elbow plot for DestinationIP
wcss = []
K = range(1, 4)
for k in K:
    kmeans_dip = KMeans(n_clusters=k).fit(df_destIP)
    wcss.append(sum(np.min(cdist(df_destIP,
                                 kmeans_dip.cluster_centers_,
                                 'euclidean') ** 2, axis=1)) / df_destIP.shape[0])

# Plot elbow
plt.figure()
plt.plot(K, wcss)
plt.plot(K, wcss, 'b')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Within clusters Sum of Squares')
plt.title('The Elbow Method highlighting Optimal Number of Clusters (DestIP)')
plt.show()
