import pandas as pd
import sys
import textstat
import numpy as numpy
import math
import gensim
from pprint import pprint
from string import ascii_lowercase
import Use_NN as nn
import re
import rules_implement as ri
import textstat


def read_csv(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
   # colName = ['ID','Comment']
    #column_dtypes = {
    #             'Comment': 'uint8',
    #             'Comment' : 'str'}
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

#Classify Sarcasm Based on the Neural Network That Was Trained for it
#Accuracy Rate is 78% to detect [we weill need to do better but that's for future work]
def detectSarcasm(text):
    #text = re.sub('[^A-Za-z0-9]+', '', text)
   # print(text)
  # return ("3")
    return nn.use_neural_network(text)


def detectwords(text):
    #return ri.R5(text)
    return ri.runThemAll(text)



def doComparison(words,original):


def read_csv_for_model(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    colName = ['ID','Comment','Prediction']
    column_dtypes = {
                 'ID': 'uint8',
                 'Comment' : 'str',
                 'BinaryVal' : 'uint8',
                 'Prediction' : 'uint8'
                 }
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0, dtype=column_dtypes,usecols=colName)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk




#Main Method
if __name__ == '__main__':
    df = read_csv("test_noannotations.csv")
    print(df.head())
   # df['Classify'] = 'default value'
    #print(df.head)
    #print(detectSarcasm("Well I mean they had like 3 or 4 seconds to analyze before the shock wave destroyed their house so that should be plenty of time!"))
    #df['Classify'] = df['Comment'].apply(detectSarcasm)
    df['Prediction'] = df['Comment'].apply(detectwords)
    df = df.drop(['Comment'], axis=1) # drop comments column as we do not need them
    df.to_csv('output_classifer_3.csv',index=False)
    #print(df.head())
    #print(df.head)
