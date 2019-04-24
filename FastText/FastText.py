import fasttext
import sys
import jieba
import json
import re
import pandas as pd
import time
import os.path
import sqlite3
import numpy as np

# Use default dictionary & USER dictionary

jieba.set_dictionary('../data/dict.txt.big')
jieba.load_userdict('../data/dict.txt')

# Train the Classifier

classifier = fasttext.supervised('./data/TrainCorpus.txt', 'model',
                                 label_prefix="__label__")

'''
result = classifier.test('./data/TestCorpus.txt')
print ('Precision@1:', result.precision)
print ('Recall@1:', result.recall)
print ('Number of examples:', result.nexamples)
'''

# Evaluate with Console Input

'''
texts = [' '.join(jieba.cut(sys.argv[1]))]
labels = classifier.predict(texts)
print(labels)
'''

# Or with the probability

'''
labels = classifier.predict_proba(texts)
print(labels)
'''

# Evaluation

CorpusPath = '../data/TestCorpus.txt'
LabelPath = '../data/TestLabel.txt'

x_raw = list(open(CorpusPath, "r", encoding='utf-8').readlines())
x_raw = [s.strip() for s in x_raw]
x_raw = [re.sub(' {2,}', ' ', s) for s in x_raw]

y_raw = list(open(LabelPath, "r", encoding='utf-8').readlines())
y_raw = [s.strip() for s in y_raw]
y_raw = np.asarray([1 if (Label == '1') else 0 for Label in y_raw])

y_predicted = np.asarray([s[0] for s in classifier.predict(x_raw)])
predicted = np.asarray([1 if (Label == 'spam') else 0 for Label in y_predicted])
actual = y_raw

# print(actual)

TP = np.count_nonzero(predicted * actual)
TN = np.count_nonzero((predicted - 1) * (actual - 1))
FP = np.count_nonzero(predicted * (actual - 1))
FN = np.count_nonzero((predicted - 1) * actual)
accuracy = (TP + TN) / (TP + FN + TN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = (2 * precision * recall) / (precision + recall)
print("Total Spam: {}/{}".format(sum(actual), len(actual)))
print("Total Predicted Spam: {}/{}".format(sum(predicted), len(predicted)))
print("TP: {} TN: {}".format(TP, TN))
print("FP: {} FN: {}".format(FP, FN))
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1: {}".format(f1))