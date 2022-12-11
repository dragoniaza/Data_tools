# -*- coding: utf-8 -*-

import pandas as pd
import re
import joblib
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score,confusion_matrix, plot_roc_curve


class TraffyFondueModel:
  def __init__(self, model, item):
    self.model = model
    self.item = item

  def import_data(self):
    df = pd.DataFrame([self.item.dict()])
    df = df.dropna()
    df['clean_comment'] = df['comment'].map(lambda x: self.clean_str(x))
    df['clean_comment'] = df['clean_comment'].map(lambda x: self.token_word(x))
    # print(df.to_markdown())
    # cvec = CountVectorizer(analyzer= 'word')
    # cvec.fit(df)
    # load the model from disk
    filename = './model/'+'cvec_model.model'
    # filename = '/Users/80524/Downloads/apache_airflow/model/cvec_model.model'
    cvec_model = joblib.load(open(filename, 'rb'))
    X_test = []
    X_test = cvec_model.transform(df['clean_comment'])
    return X_test

  def train_Model(self, listok):
    logistic = self.classifier_model(self.model,listok)
    return logistic
    
  def classifier_model(self, model, X_test):
    filename = './model/'+'Trainable_model.model'
    # filename = '/Users/80524/Downloads/apache_airflow/model/Trainable_model.model'
    loaded_model = joblib.load(open(filename, 'rb'))
    predicted = loaded_model.predict(X_test)
    # print("\n\n",model,'\n')
    # df = pd.read_csv('./data/traffy_fondue_data.csv')
    # print(type(predicted))
    # df[predicted] = predicted
    # df.to_csv('./data/traffy_fondue_data.csv', index= False, encoding="UTF-8")
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