import csv
import numpy as np
import pickle

pred = pickle.load(open('../data/dc.csv', 'rb'))

with open("../data/output.csv", "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["id", "Class_1", "Class_2", "Class_3", "Class_4",
    "Class_5", "Class_6", "Class_7", "Class_8", "Class_9",])
    for row in pred:
        output_row = []
        writer.writerow(row)
