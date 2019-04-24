import jieba
import requests
from lxml import etree
from selenium import webdriver
import json
import re
import pandas as pd
import time
import openpyxl
import os.path
import sqlite3
import pycantonese as pc

# Use default dictionary & USER dictionary

jieba.set_dictionary('../data/dict.txt.big')
jieba.load_userdict('../data/dict.txt')

# Create the Training Corpus

wordfreq = {}
ForumList = ["LIHKG","HKG"]
Database = sqlite3.connect('../Thread.sqlite3')
LineCount = 0
TrainSize = 100859
SplitSize = int(TrainSize * 1/5) # [2/5 * TrainData, 3/5 * TrainData]
Interval = int(TrainSize * 4/5) # [0, 2/5 * TrainData]

with open('../data/TrainCorpus.txt', 'w', encoding='utf-8') as TrainCorpus:

    with open('../data/TrainLabel.txt', 'w', encoding='utf-8') as TrainLabel:

        with open('../data/TestCorpus.txt', 'w', encoding='utf-8') as TestCorpus:

            with open('../data/TestLabel.txt', 'w', encoding='utf-8') as TestLabel:

                for Forum in ForumList:

                    # Retrieve the Corpus line by line

                    sql = 'SELECT Content, Spam FROM ' + Forum + '_Thread;'
                    Result = pd.read_sql_query(sql, Database)

                    for index, row in Result.iterrows():

                        if Interval > 0:
                            Corpus = TrainCorpus
                            Label = TrainLabel
                            Interval = Interval - 1
                        elif SplitSize > 0:
                            Corpus = TestCorpus
                            Label = TestLabel
                            SplitSize = SplitSize - 1
                        else:
                            Corpus = TrainCorpus
                            Label = TrainLabel

                        # Get Label

                        if(int(row['Spam']) == 1):
                            Label.write("1")
                        else:
                            Label.write("0")

                        # Word Segmentation

                        seg_list = jieba.cut(row['Content'])

                        for word in seg_list:
                            Corpus.write(word + ' ')
                            if word not in wordfreq:
                                wordfreq[word] = 0
                            wordfreq[word] += 1

                        LineCount = LineCount + 1

                        Label.write('\n')
                        Corpus.write('\n')

                print("read " + str(LineCount) + " line")


# Count the Word Frequency

with open('../data/Count.txt', 'w', encoding='utf-8') as Count:
    for k, v in sorted(wordfreq.items(), key=lambda x: x[1]):
        Count.write(str(k) + " " + str(v) + '\n')