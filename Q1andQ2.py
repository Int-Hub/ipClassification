import pandas as pd
import matplotlib.pyplot as plt

# defining data frame
data = pd.read_csv("coursework1.csv")


# [Q2]
df_sourceIP = data['sourceIP']
df_destIP = data['destIP']
classf = data['classification']

# Finding unique entires in source, dest IPs and classification
df_sourceIP = len(data['sourceIP'].unique())
print(df_sourceIP)
df_destIP = len(data['destIP'].unique())
print(df_destIP)
classf = len(data['classification'].unique())
print(classf)

# [Q2]
# plotting histogram

# sourceIP histogram plot
hist_plot = data['sourceIP'].hist(bins=100, grid=False, edgecolor='black')
hist_plot.set_title('SourceIP Histogram')
hist_plot.set_xticklabels([])
hist_plot.set_ylabel('Frequency')
plt.xticks(
    rotation=90,
    fontweight='heavy',
    fontsize='medium',
)
plt.show()

# destIP histogram plot
hist_plot = data['destIP'].hist(bins=70, grid=False, edgecolor='black')
hist_plot.set_title('destIP Histogram')
hist_plot.set_xticklabels([])
hist_plot.set_ylabel('Frequency')
plt.xticks(
    rotation=90,
    fontweight='light',
    fontsize='medium',
)
plt.show()
