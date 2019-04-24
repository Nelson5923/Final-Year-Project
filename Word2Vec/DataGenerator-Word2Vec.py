import jieba
import requests
from lxml import etree
from selenium import webdriver
import json
import re
import pandas as pd
import time
from browsermobproxy import Server
from pandas import ExcelWriter
import openpyxl
import os.path
import sqlite3
import pycantonese as pc

# Use default dictionary & USER dictionary

jieba.set_dictionary('../data/dict.txt.big')
jieba.load_userdict('../data/dict.txt')

# Generate the Stop Word dictionary

stop_words = set()
with open('../data/StopWords.txt', 'r', encoding='utf-8') as StopWords:
    for word in StopWords:
        stop_words.add(word.strip('\n'))

# Create the Training Corpus

wordfreq = {}
ForumList = ["LIHKG","HKG"]
Database = sqlite3.connect('../Thread.sqlite3')
LineCount = 0

with open('../data/TrainCorpus.txt', 'w', encoding='utf-8') as TrainCorpus:

    for Forum in ForumList:

        # Retrieve the Corpus line by line

        sql = 'SELECT Content, Spam FROM ' + Forum + '_Thread;'
        Result = pd.read_sql_query(sql, Database)

        for index, row in Result.iterrows():

            # Word Segmentation

            seg_list = jieba.cut(row['Content'])

            for word in seg_list:
                if word not in stop_words:
                    TrainCorpus.write(word + ' ')
                    if word not in wordfreq:
                        wordfreq[word] = 0
                    wordfreq[word] += 1

            LineCount = LineCount + 1
            TrainCorpus.write('\n')

    print("read " + str(LineCount) + " line")


# Count the Word Frequency

with open('../data/Count.txt', 'w', encoding='utf-8') as Count:
    for k, v in sorted(wordfreq.items(), key=lambda x: x[1]):
        Count.write(str(k) + " " + str(v) + '\n')