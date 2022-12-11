# Data_tools

1.Build docker container with command

``` docker build . --tag extending_airflow:lastest ```

2.Start docker compose with command

``` docker-compose up -d ```

3.get into api directory

4.install requirements.txt with command

``` pip -r requirements.txt ```

5.install uvicorn with command

``` pip install "uvicorn[standard]" gunicorn ```

6.run main.py python file 

``` uvicorn main:app --reload --port 5500 --host 0.0.0.0 ```

7.you can access our api from post method on port 5500
