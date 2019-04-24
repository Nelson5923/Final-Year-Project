import fasttext
import sys
import jieba
import os.path
import sqlite3
import pandas as pd
def LoadModel(isTrainData, model_path):

    # Train the Classifier

    if isTrainData:
        classifier = fasttext_load_model(model_path.join('model.bin'), encoding='utf-8')
    else:
        classifier = fasttext.load_model('./data/model.bin', encoding='utf-8')

    return classifier

def TrainModel(file_path):

    # Use default dictionary & USER dictionary

    jieba.set_dictionary('./data/dict.txt.big')
    jieba.load_userdict('./data/dict.txt')

    # Train the Classifier

    classifier = fasttext.supervised(file_path, 'model',
                                     label_prefix="__label__")

    os.rename("./model.bin", os.path.dirname(file_path) + '/model.bin')

    return classifier

def TextClassifier(t, classifier):

    # Use default dictionary & USER dictionary

    jieba.set_dictionary('./data/dict.txt.big')
    jieba.load_userdict('./data/dict.txt')

    # Evaluate with Console Input

    texts = [' '.join(jieba.cut(t))]
    labels = classifier.predict(texts)

    return labels

def BatchClassifier(content, classifier):

    # Use default dictionary & USER dictionary

    jieba.set_dictionary('./data/dict.txt.big')
    jieba.load_userdict('./data/dict.txt')

    # Evaluate with Console Input
    ResultList = []
    for result in content:
        texts = [' '.join(jieba.cut(result))]
        for label in classifier.predict(texts):
            for s in label:
                result = result + ' [' + s + ']'
        ResultList.append(result)

    return ResultList

def SearchUser(threshold):

    Database = sqlite3.connect('./Thread.sqlite3')
    sql = 'SELECT AuthorID, AuthorName, COUNT(*) FROM LIHKG_Thread WHERE Expected = 1 GROUP BY AuthorID;'
    Result = pd.read_sql_query(sql, Database)
    ResultList = []

    for index, row in Result.iterrows():

        if(row['COUNT(*)'] > threshold):
            ResultList.append(row['AuthorName'])

    return ResultList

def main():

    t = sys.argv[1]
    for label in TextClassifier(t):
        for s in label:
            print(label)

if __name__ == "__main__":
    main()
