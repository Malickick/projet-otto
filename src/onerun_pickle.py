
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("../data/train.csv").drop("id",axis=1)
df_test = pd.read_csv("../data/test.csv").drop("id",axis=1)
# df_test_id = df_test.loc[df_test['id']]

labels = df["target"].values
print(type(labels))
print(labels.shape)

unique_labels = np.unique(labels)
nb_labels = len(unique_labels)


df_train = df.drop("target", axis=1)

# df_train

X = df_train.values
X_test = df_test.values

print(X.shape)
print(X_test.shape)

y = []
for label in labels:
    y.append(int(label[-1:])-1)
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

X_dc = pairwise_distances(X,centroids, metric="l1")
X_test_dc = pairwise_distances(X_test, centroids, metric="l1")

print("Construction terminée \n")


def to_one_hot(labels):

    b = np.zeros((labels.size, labels.max()+1))
    b[np.arange(labels.size),labels] = 1
    return b


# Prédictions avec la représentation brute

# start = time.time()
#
# print("Apprentissage données brutes")
#
# k_neighbors = 10
#
# X_train = normalize(X)
# y_train = y

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# knn = KNeighborsClassifier(n_neighbors = k_neighbors)
# knn.fit(X_train, y_train)
#
# pred = knn.predict(X_test)
# pred_proba = knn.predict_proba(X_test)
# pickle.dump(pred_proba, open('../data/brute.csv', 'wb'))

# print("-- Scores en représentation brute pour K =", k_neighbors, " -- \n")
#
# print("Score de précision : ", accuracy_score(y_test, pred))
# y_true_one_hot = to_one_hot(y_test)
# pred_one_hot = to_one_hot(pred)
# print("Score en log loss :", log_loss(y_true_one_hot, pred_one_hot))
# print("temps écoulé :", round(time.time()-start, 2),"secondes \n")

###### Prédictions avec dcDistance

print("Apprentissage données en représentation dcDistance")

start = time.time()
y_train = y
X_train_dc = normalize(X_dc)

# X_train_dc, X_test_dc, y_train, y_test = train_test_split(X_dc, y, test_size=0.25, random_state=42)

# knn = KNeighborsClassifier(n_neighbors = k_neighbors)
# knn.fit(X_train_dc, y_train)

# CLassifieur
clf = MultinomialNB()
clf.fit(X_train_dc, y_train)

# clf = clf = LogisticRegression(random_state=0, solver='lbfgs',
# multi_class='multinomial')
# clf.fit(X_train_dc, y_train)

pred_dc = clf.predict(X_test_dc)
pred_dc_proba = clf.predict_proba(X_test_dc)
pickle.dump(pred_dc_proba, open('../data/dc.csv', 'wb'))

# print("-- Scores en représentation dcDistance pour K =", k_neighbors, "-- \n")
#
# print("Score de précision : ", accuracy_score(y_test, pred_dc))
# y_true_one_hot = to_one_hot(y_test)
# pred_one_hot = to_one_hot(pred_dc)
# print("Score en log loss :", log_loss(y_true_one_hot, pred_one_hot))
#
# print("temps écoulé :", round(time.time()-start, 2),"secondes")

print('done')
