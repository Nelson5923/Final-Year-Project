from gensim.models import word2vec
import warnings
import os
import pandas as pd
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
import requests
from lxml import etree
from selenium import webdriver
import json
import re
from pandas import ExcelWriter
import time
import openpyxl
from openpyxl import load_workbook
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

def ExportToExcel(Thread, Name):  # Not in Used
    writer = ExcelWriter(Name)
    Thread.to_excel(writer, 'Sheet')
    writer.save()

def main():
    sentences = word2vec.LineSentence("../data/TrainCorpus.txt")
    model = word2vec.Word2Vec(sentences, size=250)
    model.save("../data/word2vec.model")
    df = most_similar(model, ['中大','生活','香港','台灣','友情','中國', '五毛','生命','世界','幸福','意志','愛'], topn=10)
    ExportToExcel(df, '../data/Result.xlsx')

if __name__ == "__main__":
    main()