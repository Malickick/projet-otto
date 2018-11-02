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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

df_raw = pd.read_csv("../data/train.csv").drop("id",axis=1)
df_test_raw = pd.read_csv("../data/test.csv").drop("id",axis=1)

# Récupération des labels
labels = df_raw["target"].values
y = []
for label in labels:
    y.append(int(label[-1:])-1)
y = np.asarray(y)
# Récupérations des features sous forme de ndarray
X = df_raw.values
# X_test = df_test_raw.values
print('X shape : '+str(X.shape))
print('y shape : '+str(y.shape))
# print('X_test shape : '+str(X_test.shape))

# Feature selection
k_best = 40
selection = SelectKBest(chi2, k=k_best)
selection.fit_transform(X, y)
# X = selection.transform(X)
# X_test = selection.transform(X_test)
print('X new shape : '+str(X.shape))
# print('X test new shape : '+str(X_test.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
print("%.2f" % log_loss(y_test, pred, eps=1e-15, normalize=True))
