
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, log_loss
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import normalize,scale
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import numpy as np
# import time


# df = pd.read_csv("../data/train.csv").drop("id",axis=1)

# labels = df["target"].values
# print(type(labels))
# print(labels.shape)

# unique_labels = np.unique(labels)
# nb_labels = len(unique_labels)


# df_train = df.drop("target", axis=1)

# # df_train

# X = df_train.values

# print(X.shape)

# y = []
# for label in labels:
#     y.append(int(label[-1:])-1)
# y = np.asarray(y)


# print("Construction des centroïdes")
# centroids = []
# for label in unique_labels:
#     same_labels = df_train.loc[df['target'] == label]
#     same_labels = same_labels.values
#     centroids.append(same_labels.sum(axis = 0))
# centroids = np.asarray(centroids)


# from sklearn.metrics.pairwise import pairwise_distances
# print("Construction de la représentation dcDistance")

# X_dc = pairwise_distances(X,centroids)

# print("Construction terminée \n")


# def to_one_hot(labels):
    
#     b = np.zeros((labels.size, labels.max()+1))
#     b[np.arange(labels.size),labels] = 1
#     return b


# # Prédictions avec la représentation brute

# start = time.time()

# print("Apprentissage données brutes")

# k_neighbors = 50

# X = scale(normalize(X))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # knn = KNeighborsClassifier(n_neighbors = k_neighbors)
# # knn.fit(X_train, y_train)

# # pred = knn.predict(X_test)

# rf = RandomForestClassifier(max_depth = 90)
# rf.fit(X_train, y_train)

# pred = rf.predict(X_test)

# print("-- Scores en représentation brute pour K =", k_neighbors, " -- \n")

# print("Score de précision : ", accuracy_score(y_test, pred))
# y_true_one_hot = to_one_hot(y_test)
# pred_one_hot = to_one_hot(pred)
# print("Score en log loss :", log_loss(y_true_one_hot, pred_one_hot))
# print("temps écoulé :", round(time.time()-start, 2),"secondes \n")

# ###### Prédictions avec dcDistance

# print("Apprentissage données en représentation dcDistance")

# start = time.time()

# X_dc = scale(normalize(X_dc))

# X_train_dc, X_test_dc, y_train, y_test = train_test_split(X_dc, y, test_size=0.25, random_state=42)

# knn = KNeighborsClassifier(n_neighbors = k_neighbors)
# knn.fit(X_train_dc, y_train)

# pred_dc = knn.predict(X_test_dc)

# print("-- Scores en représentation dcDistance pour K =", k_neighbors, "-- \n")

# print("Score de précision : ", accuracy_score(y_test, pred_dc))
# y_true_one_hot = to_one_hot(y_test)
# pred_one_hot = to_one_hot(pred_dc)
# print("Score en log loss :", log_loss(y_true_one_hot, pred_one_hot))

# print("temps écoulé :", round(time.time()-start, 2),"secondes")

# start = time.time()

# rf = RandomForestClassifier(max_depth = 9)
# rf.fit(X_train_dc, y_train)

# pred_dc_rf = rf.predict(X_test_dc)

# print("-- Scores en représentation dcDistance pour Random Forest-- \n")

# print("Score de précision : ", accuracy_score(y_test, pred_dc_rf))
# y_true_one_hot = to_one_hot(y_test)
# pred_one_hot = to_one_hot(pred_dc_rf)
# print("Score en log loss :", log_loss(y_true_one_hot, pred_one_hot))

# print("temps écoulé :", round(time.time()-start, 2),"secondes")

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

## Récupération des données
df_raw = pd.read_csv("../data/train.csv").drop("id",axis=1)
df_test_raw = pd.read_csv("../data/test.csv").drop("id",axis=1)
# Indices récupéré par feature selection cf notebook
# features = ['feat_5', 'feat_50', 'feat_60', 'feat_83']
features_int = [ 0,  1,  2,  3,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36,
       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54,
       55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
       72, 73, 74, 75, 76, 77, 78, 79, 82, 84, 85, 86, 87, 88, 89, 90, 91]
features = []
for i in features_int:
    features.append('feat_'+str(i+1))

features = ['feat_6', 'feat_51', 'feat_61', 'feat_84']
df_train = df_raw[features]
df_test = df_test_raw[features]


# Récupération des labels
labels = df_raw["target"].values
y_train = []
for label in labels:
    y_train.append(int(label[-1:])-1)
y_train = np.asarray(y_train)
# Récupérations des features sous forme de ndarray
X_train = df_train.values
X_test = df_test.values
print('X_train shape : '+str(X_train.shape))
print('y_train shape : '+str(y_train.shape))
print('X_test shape : '+str(X_test.shape))

## Classifieur
# Normalisation
X_train = normalize(X_train)
X_test = normalize(X_test)
# Choice of classifier
# clf = RandomForestClassifier(n_estimators=250, n_jobs=-1)
clf = MultinomialNB()
# clf = KNeighborsClassifier(n_neighbors=10)
# clf = clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
# hidden_layer_sizes=(1000, 100), random_state=1, activation='relu')
# calibration
clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
# Entrainement
print('Entrainement en cours')
# clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
print('Entrainement terminé')
pred_proba = clf.predict_proba(X_test)
pickle.dump(pred_proba, open('../data/brute.csv', 'wb'))
print('Données pickélisées')
