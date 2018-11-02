import csv
import numpy as np
import pickle
import pandas as pd

pred = pickle.load(open('../data/brute.csv', 'rb'))

df = pd.DataFrame(pred, columns=["Class_1", "Class_2", "Class_3", "Class_4",
"Class_5", "Class_6", "Class_7", "Class_8", "Class_9",])
df.index += 1
df.to_csv('../data/output.csv',index=True, index_label='id')
