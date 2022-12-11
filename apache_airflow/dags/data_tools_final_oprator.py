import json

from airflow.utils.dates import days_ago
from airflow import DAG

from airflow.operators.dummy import DummyOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator

#data scraping module
# -*- coding: utf-8 -*-

import datatool_model
from datatool_model import TraffyFondueModel
import data_extraction
from data_extraction import DataExtract
from sklearn.linear_model import LogisticRegression

def extract_data():
  extract = DataExtract('https://www.traffy.in.th/?page_id=27351')
  extract.exportData()

def Model_processing():
  logistic_model = LogisticRegression(random_state = 42)
  traffy = TraffyFondueModel(logistic_model)
  preprocess = traffy.import_data()
  trained_model = traffy.train_Model(preprocess)
  print(trained_model)

with DAG('traffy_fondue_data_analyst', start_date=days_ago(1)) as dag:
  extract = PythonOperator(task_id='extract_data', python_callable=extract_data)
  model = PythonOperator(task_id='model_processing', python_callable=Model_processing)

  extract >> model