import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\DELL\Downloads\Mall_Customers.csv")
X= dataset.iloc[:,[3,4]].values

from sklearn.cluster import DBSCAN

# Fitting DBSCAN
dbscan = DBSCAN(eps=5, min_samples=5)
y_dbscan = dbscan.fit_predict(X)
# Plot DBSCAN Clusters
plt.figure(figsize=(8, 5))
plt.scatter(X[y_dbscan == 0, 0], X[y_dbscan == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_dbscan == 1, 0], X[y_dbscan == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_dbscan == -1, 0], X[y_dbscan == -1, 1], s=100, c='black', label='Outliers')

plt.title('DBSCAN Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()