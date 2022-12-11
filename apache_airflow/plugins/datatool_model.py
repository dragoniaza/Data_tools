# -*- coding: utf-8 -*-

import pandas as pd
import re
import joblib
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score,confusion_matrix, plot_roc_curve


class TraffyFondueModel:
  def __init__(self, model):
    self.model = model

  def import_data(self):
    df = pd.read_csv('./data/traffy_fondue_data.csv')
    df = df.dropna()
    df['clean_comment'] = df['comment'].map(lambda x: self.clean_str(x))
    df['clean_comment'] = df['clean_comment'].map(lambda x: self.token_word(x))
    # df['word_token'] = df['clean_comment'].map(lambda x: self.token_word(x.lower()))
    df['type'] = df['type'].map(lambda x: x.split(','))
    df['num_type'] = df['type'].map(lambda x: len(x))
    df = df[df['num_type'] == 1]
    df = df.explode('type')
    df['type'].nunique()
    X = df['clean_comment']
    y = df['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
    cvec = CountVectorizer(tokenizer=self.tokenize)
    cvec.fit(X_train)
    X_train = cvec.transform(X_train)
    X_test = cvec.transform(X_test)
    # print(X_train.shape)
    # print(X_test.shape)
    # save the model to disk
    filename = './model/'+'cvec_model.model'
    # filename = '/Users/80524/Downloads/apache_airflow/model/cvec_model.model'
    joblib.dump(cvec, open(filename, 'wb'))
    return [X_train, X_test, y_train, y_test]

  def train_Model(self, listok):
    logistic = self.classifier_model(self.model,listok)
    return logistic
    
  def classifier_model(self, model, listdata):
    X_train = listdata[0]
    X_test = listdata[1]
    y_train = listdata[2]
    y_test = listdata[3]
    model.fit(X_train,y_train)
    # save the model to disk
    filename = './model/'+'Trainable_model.model'
    # filename = '/Users/80524/Downloads/apache_airflow/model/Trainable_model.model'
    joblib.dump(model, open(filename, 'wb'))
    score_train = model.score(X_train, y_train)
    score_val = cross_val_score(model,X_train,y_train,cv=5).mean()
    score_test = model.score(X_test,y_test)
    predicted = model.predict(X_test)
    print("\n\n",model,'\n')
    print("Train Accuracy Rate           =",round(score_train,4))
    print("Validataion Accuracy Rate    =",round(score_val,4))
    print("Test Accuracy Rate      =",round(score_test,4))
    df = pd.read_csv('./data/traffy_fondue_data.csv')
    # print(type(predicted))
    # df[predicted] = predicted
    df.to_csv('./data/traffy_fondue_data.csv', index= False, encoding="UTF-8")
    return predicted

  def clean_str(self, sentence):
    # Abbreviation word
    abbreviation = {'ซ.': 'ซอย', 'ถ.': 'ถนน','กทม': 'กรุงเทพ', 'พ.ย.': 'พฤศจิกายน', 'พย': 'พฤศจิกายน', 'ก.ม.': 'กิโลเมตร'}
    for abv in abbreviation.keys():
      if abv in sentence:
        new_sentence = sentence.replace(abv, abbreviation[abv])
      else:
        new_sentence = sentence
    # Remove non letter    
    letters_only = re.sub("[^\u0E00-\u0E7Fa-zA-Z']", ' ', new_sentence)

    return letters_only
  
  def token_word(self, text):
    text = str(text)
    text = re.sub('[^ก-๙]','',text)
    stop_word = list(thai_stopwords())
    sentence = word_tokenize(text)
    result = [word for word in sentence if word not in stop_word and " " not in word]
    return " /".join(result)

  def tokenize(self, d):
    result = d.split("/")
    result = list(filter(None, result))
    return result

logistic_model = RandomForestClassifier(random_state = 42)
traffy = TraffyFondueModel(logistic_model)
preprocess = traffy.import_data()
trained_model = traffy.train_Model(preprocess)
# print(trained_model)