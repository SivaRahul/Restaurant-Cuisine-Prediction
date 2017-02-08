
# coding: utf-8

# In[214]:

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import re
import sklearn
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# In[215]:

traindf = pd.read_json("F:\Intern Assignments\indix\What's_cooking/train.json")


# In[216]:

traindf


# In[217]:

(traindf['ingredients'][1])


# In[218]:

' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', "plain flour , ground pepper , salt , tomatoes , ground black pepper , thyme , eggs , green tomatoes , yellow corn meal , milk , vegetable oil"))])


# In[219]:

traindf['ingredients_lem'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists])
                                 .strip() for lists in traindf['ingredients']] 


# In[220]:

traindf['ingredients_lem'][1]


# In[221]:

testdf = pd.read_json("F:\Intern Assignments\indix\What's_cooking/test.json") 


# In[222]:

testdf


# In[223]:

testdf['ingredients'][1]


# In[224]:

testdf['ingredients_lem'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists])
                                 .strip() for lists in testdf['ingredients']] 


# In[225]:

testdf['ingredients_lem'][1]


# In[234]:

corpus_train = traindf['ingredients_lem']
vectorizer_train = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , token_pattern=r'\w+' , sublinear_tf=False)
tfidf_train=vectorizer_train.fit_transform(corpus_train).todense()
corpus_test = testdf['ingredients_lem']
vectorizer_test = TfidfVectorizer(stop_words='english')
tfidf_test=vectorizer_train.transform(corpus_test)


# In[235]:

traindf


# In[236]:

predictors_train = tfidf_train
predictors_test = tfidf_test
targets_train = traindf['cuisine']


# In[237]:

clf_svm= LinearSVC()
classifier_svm=clf_svm.fit(predictors_train,targets_train)
predictions_svm=classifier_svm.predict(predictors_test)


# In[238]:

testdf['cuisine_svm'] = predictions_svm
testdf = testdf.sort('id' , ascending=True)


# In[239]:

testdf


# In[212]:

testdf[['id','cuisine_svm']].to_csv("F:\Intern Assignments\indix\What's_cooking/submission_svm.csv")

