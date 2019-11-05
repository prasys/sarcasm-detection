import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # to surpress future warnings
import pandas as pd
import sys
import textstat
import numpy as numpy
import math
import gensim
from pprint import pprint
from string import ascii_lowercase
#import Use_NN as nn
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split , KFold , LeaveOneOut , LeavePOut , ShuffleSplit , StratifiedKFold , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor , VotingClassifier , RandomTreesEmbedding, ExtraTreesClassifier , RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC , SVC
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import TheilSenRegressor , SGDClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB, MultinomialNB , ComplementNB
from sklearn.linear_model import LogisticRegressionCV , PassiveAggressiveClassifier, HuberRegressor
from sklearn.metrics import f1_score , recall_score , accuracy_score , precision_score , jaccard_score , balanced_accuracy_score, confusion_matrix
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier , RadiusNeighborsClassifier
import nltk
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from imblearn.pipeline import make_pipeline
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import string
import xgboost as xgb



removeUnWanted = re.compile('[\W_]+') #strip off the damn characters
isClassify = False #to run classification on test data
isCreationMode = False
isWord2Vec = False
isEmbeddings = True
isBOW = False
doc2VecFileName ="doc2vec"
useSMOTE = True
STATE = 21
#logistic , nb , svm
DETERMINER = 'xgboost'


# Take any text - and converts it into a vector. Requires the trained set (original vector) and text we pan to infer (shall be known as test)
def vectorize(train,test):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    vectorizer = CountVectorizer(ngram_range=(2,3),min_df=0, lowercase=True, analyzer='char_wb',tokenizer = token.tokenize, stop_words='english') #this is working
    #vectorizer = CountVectorizer(min_df=0, lowercase=True)
   # vectorizer = TfidfTransformer(use_idf=True,smooth_idf=True)
    x = vectorizer.fit(train)
    x = vectorizer.transform(test)
    return x

def loadEmbeddings(filename):
	embeddings = numpy.load(filename,allow_pickle=True)
	print(embeddings.shape)
	return embeddings

# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    colName = ['ID','Comment','Prediction']
    column_dtypes = {
                 'ID': 'uint8',
                 'Comment' : 'str',
                 'Prediction' : 'uint8'
                 }
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0, dtype=column_dtypes,usecols=colName,encoding = "ISO-8859-1")
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

def read_csv2(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    colName = ['ID','Comment']
    column_dtypes = {
                 'ID': 'uint8',
                 'Comment' : 'str'
                 }
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0, dtype=column_dtypes,usecols=colName)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

#Classify Sarcasm Based on the Neural Network That Was Trained for it - TO DO 
def detectSarcasm(text):
    #text = re.sub('[^A-Za-z0-9]+', '', text)
   # print(text)
  # return ("3")
    return nn.use_neural_network(text)

def calcSyllableCount(text):
    return textstat.syllable_count(text, lang='en_US')


def calcLexCount(text):
    return textstat.lexicon_count(text)

def commentCleaner(df):
    df['Comment'] = df['Comment'].str.lower()
   # df['Comment'] = df['Comment'].str.replace("[^abcdefghijklmnopqrstuvwxyz1234567890' ]", "")

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

# Converts to POS Tags that can be used
def tag(sent):
    words=nltk.word_tokenize(sent)
    tagged=nltk.pos_tag(words)
    return tagged

#Checks for Nouns , To Implement the method found in Cindy Chung's Physc Paper (Search for Cindy Chung and James Pennebaker and cite here)
def checkForNouns(text,method='None'):
    counter = 0
    counter2 = 0
    if "aa" in text: #Dummy variable to inform that it is outside , so we dont' track them 
        return counter
    else:
        wrb = tag(text)
        index = 0
        for row  in wrb:
            POSTag = wrb[index][1]
          #  print(POSTag)
            if (POSTag in "IN") or (POSTag in "PRP") or (POSTag in "DT") or (POSTag in "CC") or (POSTag in "VB") or (POSTag in "VB") or (POSTag in "PRP$") or (POSTag is "RB"):
                counter = counter+1
            else:
                counter2 = counter2+1
                
            index = index + 1
        if "function" in method:
            return counter
        elif "ratio" in method:
            return abs(counter2/counter)
        else:
            return counter2

#Given an un-seen dataframe and [TO DO - the column] , it will convert it into Matrix 
def convertToVectorFromDataframe(df):
    matrix = []
    targets = list(df['tokenized_sents'])
    for i in range(len(targets)):
        matrix.append(model.infer_vector(targets[i])) # A lot of tutorials use the model directly , we will do some improvement over it
    targets_out = numpy.asarray(matrix)
    return (matrix)

#A simple method which basically takes in the tokenized_sents and the tag and starts do it. 
def make_tagged_document(df,train):
    #  taggeddocs = []
    for doc, tanda in zip(df['tokenized_sents'], train):
        yield(TaggedDocument(doc,[tanda]))



def calculateScoresVariousAlphaValues(predicted_data,truth_data,threshold_list=[0.00,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,1.00]):
  for i in threshold_list:
    squarer = (lambda x: 1 if x>=i else 0)
    fucd = numpy.vectorize(squarer)
    vfunc = fucd(predicted_data)
    f1score = f1_score(y_true=truth_data,y_pred=vfunc)
    print(str(i)+","+str(perf_measure(truth_data,vfunc)))
    #print(confusion_matrix(vfunc, truth_data))
    #print(str(i)+","+ str(f1score))


# Creates a Doc2Vec Model by giving an input of documents [String]. It's much of an easier way. It then saves to disk , so it can be used later :) 
def createDoc2VecModel(documents,tag):
  docObj = list(make_tagged_document(documents,tag)) # document that we will use to train our model for
  model = Doc2Vec(documents=docObj,vector_size=500,
            # window=2, 
            alpha=.025,
            epochs=100, 
            min_alpha=0.00025,
            sample=0.335,
            ns_exponent=0.59,
            dm_concat=0,
            dm_mean=1,
            # negative=2,
            seed=10000, 
            min_count=2, 
            dm=0, 
            workers=4)
  model.save(doc2VecFileName) #our file name
  return model

# Loads Doc2Vec model based on the filename given
def loadDoc2VecModel(filepath=doc2VecFileName):
  model = Doc2Vec.load(filepath)
  return model

# Implements Class Weight to ensure that fair distribution of the classes
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}


# Selects a Classifier to perform the task
def selectClassifier(weights='balanced',classifymethod='logistic'):
  #classifier = RandomForestRegressor(n_estimators=100)
  #clf = svm.NuSVC(kernel='rbf',decision_function_shape='ovo',probability=True)
  #classifier = LinearSVC(random_state=21, tol=1e-4,C=1000,fit_intercept=False)
  if 'logistic' in classifymethod:
    cy = LogisticRegression(fit_intercept=True, max_iter=8000,solver='newton-cg',random_state=STATE,class_weight=weights)
    return cy
  elif 'nb' in classifymethod:
  	cy = GaussianNB()
  	return cy
  elif 'xgboost' in classifymethod:
    cy = xgb.XGBClassifier()
    return cy
  elif 'svm' in classifymethod:
    cy = SVC(random_state=STATE, tol=1e-3,C=3000,class_weight=weights,max_iter=8000,probability=True)
    return cy
  elif 'kn' in classifymethod:
    cy = MLPClassifier(hidden_layer_sizes=50,learning_rate='adaptive',random_state=STATE,solver='lbfgs')
    return cy
  else:
    return null


def getChars(s):
  count = lambda l1,l2: sum([1 for x in l1 if x in l2])
  return (count(s,set(string.punctuation)))

def mergeMatrix(matrixa,matrixb):
  print(matrixa.shape)
  print(matrixb.shape)
  print(matrixb)
  return(numpy.concatenate((matrixa, matrixb[:,None]), axis=1))

def w2v_preprocessing(df):
  df['Comment'] = df['Comment'].str.lower()
  df['nouns'] = df['Comment'].apply(checkForNouns,'function')
  df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Comment']), axis=1)
  df['uppercase'] = df['Comment'].str.findall(r'[A-Z]').str.len() # get upper case
  df['punct'] = df['Comment'].apply(getChars)


def FoldValidate(original,truth,classifier,iter=3):
  Val = StratifiedKFold(n_splits=iter, random_state=STATE, shuffle=True) # DO OUR FOLD here , maybe add the iteration
  for train_index,test_index in Val.split(original,truth):
    model2 = classifier
    model2.fit(original[train_index], truth[train_index])
    score = classifier.score(original[train_index], truth[train_index])
    print("Linear Regression Accuracy (using Weighted Avg):", score)
    tester = classifier.predict_proba(original[test_index])
    tester = tester[:,1]
    calculateScoresVariousAlphaValues(tester,truth[test_index])

   # print("Valuesdfs for train are ", train_index)
   # print("Values for test index are ",test_index)
   # print("Testing with the values",original[train_index])
   # print("Testing it with the values",truth[train_index])
  	#weights = get_class_weights(truth_data[test_index]) # implement the weights
  	#model2.fit(classifer_data, truth_data, class_weight=weights)
  	#unseendata = convertToVectorFromDataframe(test)
  	#tester = classifier.predict_proba(unseendata)
  	#tester = tester[:,1]
  	#calculateScoresVariousAlphaValues(tester,truth_data)

def showGraph(model):
  xgb.plot_importance(classifier, importance_type='gain',max_num_features=10)
  plt.show()



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP,FP)

#return(TP, FP, TN, FN)


#Main Method
if __name__ == '__main__':
  #train_classifier_3
    df = read_csv("train_classifier.csv") #Read CSV
    w2v_preprocessing(df) # process our junk here by converting it into tokens
    scaler = preprocessing.MinMaxScaler()
    label = df['Prediction'].values # take the values for prediction for our model
    
    #df['scores']
  #  df['Comment'] = df.apply(lambda row: nltk.word_tokenize(row['Comment']), axis=1)
  #  df['Comment'] = df['Comment'].apply(get_good_tokens)
    #tagged_train = list(make_tagged_document(sentences_test,y_train))
  #  df['nounratio'] = df['Comment'].apply(checkForNouns,ratio)
 #   commentCleaner(df) # clean the comments

   # sent = df['Comment'].values # take the comments instead , but in our case we are gonna split them up



    if isEmbeddings is True:
      sent = loadEmbeddings('embeddings.npy')
   #   nouns = df['nouns'].to_numpy()
  #    caps = df['uppercase'].to_numpy()
  #    punt = df['punct'].to_numpy()
      #nouns = preprocessing.normalize(nouns)
  #    nouns = preprocessing.minmax_scale(nouns)
   #   caps = preprocessing.minmax_scale(caps)
    #  punt = preprocessing.minmax_scale(punt)
    #  sent = mergeMatrix(sent,nouns)
    #  sent = mergeMatrix(sent,caps)
    #  sent = mergeMatrix(sent,punt)
    #  print(sent.shape)
      full_comment = sent
    else:
     sent = df
   # sent  = df.drop(['Prediction'],axis=1)

    sentences_train, sentences_test, y_train, y_test = train_test_split(
    sent, label, test_size=0.25, random_state=10000)

    if isBOW is True:
      train_x = vectorize(sentences_train['Comment'],sentences_train['Comment'])
      test_x = vectorize(sentences_train['Comment'],sentences_test['Comment'])
      full_comment = vectorize(sent['Comment'],sent['Comment'])
      full_comment = full_comment.toarray() # convert sparse matrix to dense
      train_x = train_x.todense()
      test_x = test_x.todense()

    if isWord2Vec is True:
      if isCreationMode is True:
        print("Creating Doc2Vec Model")
        model = createDoc2VecModel(sentences_train,y_train)
      else:
        print("Loading Doc2Vec Model")
        model = loadDoc2VecModel()
      train_x = convertToVectorFromDataframe(sentences_train)
      test_x = convertToVectorFromDataframe(sentences_test)
      full_comment = convertToVectorFromDataframe(df)
      full_comment =  numpy.array(full_comment)


    #We need to do the split for being consistent esle the naming runs

    if isEmbeddings is True:
      train_x = sentences_train
      test_x = sentences_test
      #scaler.fit(train_x)
      #train_x = scaler.transform(train_x)
      #test_x = scaler.transform(test_x)



    weights = get_class_weights(label)
    #smt = SMOTE()
    if useSMOTE is True:
      print("USING SMOTE TO BOOST THE IMBALANCED DATA")
      smt = SMOTE() # Boost the samples to improve the classification 
      train_x, y_train = smt.fit_sample(train_x, y_train)

    classifier = selectClassifier(classifymethod=DETERMINER)
   # classifier = AdaBoostClassifier(random_state=STATE,n_estimators=50, base_estimator=old)
    #hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3)
    #train_x = hasher.fit_transform(train_x)
    #classifer = RandomTreesEmbedding
   # print(full_comment.mean(axis=0))
  #  full_comment = preprocessing.scale(full_comment)
  #  train_x = preprocessing.scale(train_x)
   # test_x = preprocessing.scale(test_x)
    #print("SEPERATOR")
    #print(full_comment_test.mean(axis=0))

    #print(full_comment.shape)
    FoldValidate(full_comment,label,classifier)
   # y_crossfold = CrossFoldData(test_x,y_test,classifier)
   # y_crossfold = y_crossfold[:,1] # keep the 1 values only
    #weights = [0.05, 0.10 , 0.15]


    #pipe = make_pipeline(
    #SMOTE(),
    #LogisticRegression(fit_intercept=True, max_iter=1000,solver='newton-cg',random_state=STATE)
    #)
    #gsc = GridSearchCV(
    #estimator=pipe,
    #param_grid={
        #'smote__ratio': [{0: int(num_neg), 1: int(num_neg * w) } for w in weights]
    #    'smote__ratio': weights
    #},
    #scoring='f1',
    #cv=3
    #)
    #grid_result = gsc.fit(train_x, y_train)
    #print("Best parameters : %s" % grid_result.best_params_)




 #   vector_trained = vectorize(sentences_train,sentences_train) 
 #   vector_test = vectorize(sentences_train,sentences_test)
 #   skfold = StratifiedKFold(n_splits=3, random_state=100) # split into 3
 #   results_skfold = cross_val_score(classifier, full_comment, label, cv=skfold)
 #   print("Results Scores Are:" + str(results_skfold))



   # print(scores)
   # voter.fit(train_x, y_train)
 #   classifier.fit(vector_trained, y_train)
 #   clf.fit(vector_trained, y_train)
    # implement the weights



    classifier.fit(train_x, y_train)

 #   clf.fit(train_x, y_train)
 #   clf2.fit(train_x, y_train)
   # classifier2.fit(train_x, y_train)
   # score = classifier.score(vector_test, y_test)
    score = classifier.score(test_x, y_test)
    print("Linear Regression Accuracy (using Weighted Avg):", score)
   # score = classifier.score(full_comment, label)
    #print("Linear Regression Accuracy (using FULL Avg):", score)
    tester = classifier.predict_proba(test_x)
    tester = tester[:,1]
    print("*******NORMAL STARTS HERE*****")
    calculateScoresVariousAlphaValues(tester,y_test)
    #xgb.plot_importance(classifier, importance_type='gain',max_num_features=10)
    #plt.show() # matplotlib plot



  #  score = clf.score(vector_test, y_test)
  #  score = clf.score(test_x, y_test)
    #score = clf.score(vector_test, y_test)
#    score = clf2.score(test_x, y_test)
#    print("NB Accuracy (BOW):", score)
   # f1score = f1_score(test_x, y_test)
   # print("F1 Score:", score)
#    unseendata = convertToVectorFromDataframe(sentences_test)
 #   tester = classifier.predict_proba(unseendata)
 #   tester = tester[:,1]
 #   print(tester.shape)
 #   print(y_test.shape)
#    df3 = pd.DataFrame(tester)
#    print("*******NORMAL STARTS HERE*****")
#    calculateScoresVariousAlphaValues(tester,y_test)
 #   print("*******K-FOLD 3 SCORES STARTS HERE*****")
  #  calculateScoresVariousAlphaValues(y_crossfold,y_test)
  #  df3.to_csv('output_classifer_linear_reg_prob_test.csv',index=False)

  #  sentences_test.to_csv('output_junk',index=False)
  #  df3.to_csv('output_classifer_linear_reg_prob_test.csv',index=False)

  #  score = voter.score(test_x, y_test)
    #score = clf.score(vector_test, y_test)

    #Calculate Prediction for it


   # print("Voter Accuracy:", score)
   # print("MLP Accuracy:", score)
  #  score = classifier2.score(test_x, y_test)
    #score = clf.score(vector_test, y_test)
   # print("MLP Accuracy:", score)

    ## New One Starts here
    if isClassify == True:
        df2 = read_csv2("test_noannotations.csv")
        w2v_preprocessing(df2) # process our junk here by converting it into tokens
        if isBOW is True:
        	unseendata = vectorize(sentences_train['Comment'],sentences_test['Comment'])
        elif isWord2Vec is True:
        	unseendata = convertToVectorFromDataframe(df2)
        elif isEmbeddings is True:
          unseendata = loadEmbeddings('embeddings_prod.npy')
          #nouns = df2['nouns'].to_numpy()
          #nouns = preprocessing.minmax_scale(nouns)
          #unseendata = mergeMatrix(unseendata,nouns)


        tester = classifier.predict_proba(unseendata)
        tester = tester[:,1]
        i = 0.425
        squarer = (lambda x: 1 if x>=i else 0)
        fucd = numpy.vectorize(squarer) #our function to calculate
        tester = fucd(tester)

       # print(tester)
       # print(type(tester))
        df3 = pd.DataFrame(tester)
#        print(df2.head())
        #df2 = df2.drop(['Comment'], axis=1) # drop comments column as we do not need them
#        print(tester)
 #       print(df2.head())
        df2.to_csv('output_classifer_linear_reg_4.csv',index=False)
        df3.to_csv('output_classifer_linear_reg_prob_4.csv',index=False)


   # df['Classify'] = 'default value'
    #print(df.head)
    #print(detectSarcasm("Well I mean they had like 3 or 4 seconds to analyze before the shock wave destroyed their house so that should be plenty of time!"))
    #df['Classify'] = df['Comment'].apply(detectSarcasm)
   # df['Prediction'] = df['Comment'].apply(detectwords)
  #  df = df.drop(['Comment'], axis=1) # drop comments column as we do not need them
   #df2.to_csv('output_classifer_linear_reg.csv',index=False)
    #print(df.head())
    #print(df.head)