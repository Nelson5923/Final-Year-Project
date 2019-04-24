import jieba
import requests
from lxml import etree
from selenium import webdriver
import json
import re
import pandas as pd
import time
import os.path
import sqlite3
import pycantonese as pc
import os

# Use default dictionary & USER dictionary

jieba.set_dictionary('../data/dict.txt.big')
jieba.load_userdict('../data/dict.txt')

# Retrieve the Corpus line by line

Database = sqlite3.connect('../Thread.sqlite3')
ForumList = ["LIHKG","HKG"]

# Include the Stop Word with dictionary in pycantonese

'''
with open('StopWord.txt', 'w', encoding='utf-8') as StopWord:
    stop_words = pc.stop_words()
    for word in stop_words:
        StopWord.write(word + '\n')
'''

# Generate the Stop Word dictionary

stop_words = set()
with open('../data/StopWords.txt', 'r', encoding='utf-8') as StopWords:
    for word in StopWords:
        stop_words.add(word.strip('\n'))

# Create the Training Corpus

wordfreq = {}
TrainSize = 80000

with open('./data/TrainCorpus.txt', 'w', encoding='utf-8') as TrainCorpus:

    with open('./data/TestCorpus.txt', 'w', encoding='utf-8') as TestCorpus:

        for Forum in ForumList:

            # Retrieve the Corpus line by line

            sql = 'SELECT Content, Spam FROM ' + Forum + '_Thread;'
            Result = pd.read_sql_query(sql, Database)

            for index, row in Result.iterrows():

                if TrainSize > 0:

                    # FastText Labelling

                    if(int(row['Spam']) == 1):
                        TrainCorpus.write('__label__spam'  + ' ')
                    elif(int(row['Spam']) != 1):
                        TrainCorpus.write('__label__other' + ' ')

                    # Word Segmentation

                    seg_list = jieba.cut(row['Content'])
                    for word in seg_list:
                        if word not in stop_words:
                            TrainCorpus.write(word + ' ')
                            if word not in wordfreq:
                                wordfreq[word] = 0
                            wordfreq[word] += 1
                    TrainCorpus.write('\n')

                    TrainSize = TrainSize - 1

                else:

                    # FastText Labelling

                    if (int(row['Spam']) == 1):
                        TestCorpus.write('__label__spam' + ' ')
                    elif(int(row['Spam']) != 1):
                        TestCorpus.write('__label__other' + ' ')

                    # Word Segmentation

                    seg_list = jieba.cut(row['Content'])
                    for word in seg_list:
                        if word not in stop_words:
                            TestCorpus.write(word + ' ')
                    TestCorpus.write('\n')

# Count the Word Frequency

with open('./data/Count.txt', 'w', encoding='utf-8') as Count:
    for k, v in sorted(wordfreq.items(), key=lambda x: x[1]):
        Count.write(str(k) + " " + str(v) + '\n')