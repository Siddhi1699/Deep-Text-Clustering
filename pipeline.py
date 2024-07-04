
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from minisom import MiniSom
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

def get_user_input():
    doc_index = int(input("Enter the document index (0 to 17999): "))
    num_neighbors = int(input("Enter the number of neighbors: "))
    return doc_index, num_neighbors

def fit_knn_model(features):
    knn = NearestNeighbors(n_neighbors=features.shape[0])  # Use maximum number of neighbors
    knn.fit(features)
    return knn

def get_nearest_neighbors(knn, features, doc_index, num_neighbors):
    distances, indices = knn.kneighbors(features.iloc[doc_index].values.reshape(1, -1), n_neighbors=num_neighbors + 1)
    # Exclude the first neighbor as it is the document itself
    return indices.flatten()[1:], distances.flatten()[1:]



df = pd.read_csv('./data/merged.csv')
df.drop('Index', axis=1, inplace=True)
df = df.fillna(0)
df.isnull().sum()
knn_model = fit_knn_model(df)
doc_index, num_neighbors = get_user_input()
neighbor_indices, neighbor_distances = get_nearest_neighbors(knn_model, df, doc_index, num_neighbors)
neighbors_df = df.iloc[neighbor_indices].copy()
neighbors_df['Distance'] = neighbor_distances
k = int(input("Enter the number of clusters: "))
som_shape = (1,k)
features_for_som = neighbors_df.drop(columns=['Distance']).values
som = MiniSom(som_shape[0], som_shape[1], features_for_som.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)
som.train_batch(features_for_som, 1000, verbose=True)
win_map = som.win_map(features_for_som)
cluster_index = np.zeros(features_for_som.shape[0], dtype=int)
for i, x in enumerate(features_for_som):
    w = som.winner(x)
    cluster_index[i] = w[1]
neighbors_df['cluster'] = cluster_index
plt.figure(figsize=(8, 6))
for c in np.unique(cluster_index):
    plt.scatter(features_for_som[cluster_index == c, 0],
                features_for_som[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)

# Plotting centroids
centroids = som.get_weights().reshape(-1, features_for_som.shape[1])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
            s=100, linewidths=3, color='k', label='centroid')
plt.legend()
plt.show()
empty_clusters = [c for c in range(k) if np.sum(cluster_index == c) == 0]
print(f"Empty clusters: {empty_clusters}")

if len(empty_clusters) == 0:  # Compute only if there are no empty clusters
    silhouette_avg = silhouette_score(features_for_som, cluster_index)
    print(f"Silhouette Coefficient: {silhouette_avg}")
else:
    print("Cannot compute silhouette score due to empty clusters.")

if len(empty_clusters) == 0:  # Compute only if there are no empty clusters
    ch_score = calinski_harabasz_score(features_for_som, cluster_index)
    print(f"Calinski-Harabasz Index: {ch_score}")
else:
    print("Cannot compute Calinski-Harabasz Index due to empty clusters.")

features = neighbors_df.drop(columns=['cluster','Distance'])
cluster_labels = neighbors_df['cluster']
X_Train, X_Test, Y_Train, Y_Test = train_test_split(features, cluster_labels, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_Train, Y_Train)
random_forest_preds = clf.predict(X_Test)


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(features)
shap_values_ = shap_values.transpose((2,0,1))
for i in range(k):
    print(f"Cluster {i}")
    shap.summary_plot(shap_values_[i], features)
    plt.show()
for i in range(k):
    print(f"Cluster {i}")
    shap.summary_plot(shap_values_[i], features, plot_type='bar')
    plt.show()

