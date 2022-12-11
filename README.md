# Data_tools

1.Build docker container with command

``` docker build . --tag extending_airflow:lastest ```

2.Start docker compose with command

``` docker-compose up -d ```

3.get into api directory

4.install requirements.txt with command

``` pip install -r requirements.txt ```

5.install uvicorn with command

``` pip install "uvicorn[standard]" gunicorn ```

6.edit path for model file like

``` filename = '/Users/80524/Downloads/apache_airflow/model/cvec_model.model' ```

7.run main.py python file 

``` uvicorn main:app --reload --port 5500 --host 0.0.0.0 ```

8.you can access our api from post method on port 5500
