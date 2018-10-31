from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize,scale
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
import pickle


df = pd.read_csv("../data/train.csv").drop("id",axis=1)
df_test = pd.read_csv("../data/test.csv").drop("id",axis=1)
# df_test_id = df_test.loc[df_test['id']]

labels = df["target"].values
print(type(labels))
print(labels.shape)

unique_labels = np.unique(labels)
nb_labels = len(unique_labels)


df_train = df.drop("target", axis=1)

X = df_train.values
X_test = df_test.values

print(X.shape)
print(X_test.shape)

y = []
for label in labels:
    y.append(int(label[-1:]))
y = np.asarray(y)


print("Construction des centroïdes")
centroids = []
for label in unique_labels:
    same_labels = df_train.loc[df['target'] == label]
    same_labels = same_labels.values
    centroids.append(same_labels.sum(axis = 0))
centroids = np.asarray(centroids)


from sklearn.metrics.pairwise import pairwise_distances
print("Construction de la représentation dcDistance")

X_dc = pairwise_distances(X,centroids)
X_test_dc = pairwise_distances(X_test, centroids)

print("Construction terminée \n")

start = time.time()

k_neighbors = 2

y_train = y

print("Apprentissage données en représentation dcDistance")

X_train_dc = scale(normalize(X_dc))

knn = KNeighborsClassifier(n_neighbors = k_neighbors)
knn.fit(X_train_dc, y_train)

pred_proba = knn.predict_proba(X_test_dc)

print(pred_proba[:10])

pickle.dump(pred_proba, open('../data/dc.csv', 'wb'))

print('done')
