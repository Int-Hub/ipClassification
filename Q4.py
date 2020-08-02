import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

data = pd.read_csv("coursework1.csv")


# Function to separate sourceIPs into clusters
def sourceIP_cluster(data):
    # Converting data to data frame object and partitioning into numpy array
    df_sourceIP = DataFrame(data.sourceIP.value_counts())
    df_sourceIP["Frequency"] = np.array(data.sourceIP.value_counts())
    df_sourceIP["sourceIP"] = df_sourceIP.index
    df_sourceIP.reset_index(level=0, inplace=True, drop=True)

    # Divide sourceIP into four distinct clusters
    cluster = []
    for x in df_sourceIP['Frequency']:
        if x <= 21:
            cluster.append('cluster 1')
        elif 20 < x < 201:
            cluster.append('cluster 2')
        elif 200 < x < 401:
            cluster.append('cluster 3')
        else:
            cluster.append('cluster 4')

    df_sourceIP['Clusters'] = cluster

    source_IP = {}
    for k in range(len(df_sourceIP)):
        a = df_sourceIP['sourceIP'][k]
        b = df_sourceIP['Clusters'][k]
        source_IP[a] = b

    frequency_sourceIP = []
    for k in data.sourceIP:
        frequency_sourceIP.append(source_IP[k])
    return frequency_sourceIP


# Function to separate destinationIPs into clusters
def destIP_cluster(data):
    # Divide sourceIP into four distinct clusters
    df_destIP = pd.DataFrame(data.destIP.value_counts())
    df_destIP["Frequency"] = np.array(data.destIP.value_counts())
    df_destIP["destIP"] = df_destIP.index
    df_destIP.reset_index(level=0, inplace=True, drop=True)

    # Divide sourceIP into four distinct clusters
    cluster = []
    for x in df_destIP['Frequency']:
        if x <= 41:
            cluster.append('cluster 1')
        elif 40 < x < 101:
            cluster.append('cluster 2')
        elif 100 < x < 401:
            cluster.append('cluster 3')
        else:
            cluster.append('cluster 4')

    df_destIP['Clusters'] = cluster

    dest_IP = {}
    for k in range(len(df_destIP)):
        a = df_destIP['destIP'][k]
        b = df_destIP['Clusters'][k]
        dest_IP[a] = b

    frequency_destIP = []
    for k in data.destIP:
        frequency_destIP.append(dest_IP[k])
    return frequency_destIP

 # Create table to depict distinct source and individual Ips into clusters
data_freq = data.copy()
data_freq['sourceIP cluster'] = sourceIP_cluster(data)
data_freq['destIP cluster'] = destIP_cluster(data)
data_freq = data_freq[['sourceIP', 'sourceIP cluster', 'destIP', 'destIP cluster']]
print(data_freq.head())


# Declaring conditional probability function for 16 distinct probabilities
# Declaring sourceip and destination ip clusters from above
def con_prob(data, x, y, positional=True):
    data = data[[x, y]]

    # Declaring initial probability count
    prob_count = 0

    # Initialising individual clusters
    row = data[x].unique()
    column = data[y].unique()
    cluster_width = data[x].unique().shape[0]
    cluster_height = data[y].unique().shape[0]

    # Creating empty probability matrix
    prob_array = np.zeros((cluster_width, cluster_height))

    # Creating probability arrays based on cluster probabilities
    # Arrays consist of 16 probabilities
    for i in range(cluster_width):
        for k in range(cluster_height):

            num = data[(data[x] == row[i]) & (data[y] == column[k])].shape[0]
            prob_count += num

            if positional:
                num = num / data[(data[x] == row[i])].shape[0]

            elif not positional:
                num = num / data.shape[0]

            prob_array[i, k] = num
            #print(prob_array)
    prob_matrix = DataFrame(prob_array, row, column)
    return prob_matrix


# Plotting Heat map depicting conditional probability between sourceIP and destinationIP clusters
prob_correlation = con_prob(data_freq, 'destIP cluster', 'sourceIP cluster', positional=True)
plot = sns.heatmap(prob_correlation, cmap="Blues", annot=True, center=True)
plt.xlabel('sourceIP clusters')
plt.ylabel('destIP clusters')
plt.title('Conditional Probability between SourceIp and DestinationIP')
plt.show()
