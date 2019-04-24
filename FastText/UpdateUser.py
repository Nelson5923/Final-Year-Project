import fasttext
import sys
import jieba
import os.path
import sqlite3
import pandas as pd
jieba.set_dictionary('./data/dict.txt.big')
jieba.load_userdict('./data/dict.txt')
classifier = fasttext.load_model('./data/model.bin', encoding='utf-8')

Database = sqlite3.connect('./Thread.sqlite3')
cursor = Database.cursor()

# Retrieve the Corpus line by line

sql = 'SELECT AuthorID, Content FROM LIHKG_Thread;'

Result = pd.read_sql_query(sql, Database)

for index, row in Result.iterrows():

    texts = [' '.join(jieba.cut(row['Content']))]
    for label in classifier.predict(texts):
        print(label)
        label = [1 if (Label == '__label__spam') else 0 for Label in label]
        print(label)
        for s in label:
            print(s)
            cursor.execute('''UPDATE LIHKG_Thread SET Expected = ? WHERE AuthorID = ? 
            AND Content = ?;''', (s, row['AuthorID'], row['Content']))
            Database.commit()

Database.close()