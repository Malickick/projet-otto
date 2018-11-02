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
