# 
FROM python:3.9
# 
WORKDIR /code

# 
COPY ./API/requirements.txt /code/requirements.txt

COPY ./API/TraffyFondue_Model.py /code/TraffyFondue_Model.py

COPY ./model/* /code/model/

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install "uvicorn[standard]" gunicorn

# 
COPY ./API /code/API

# 
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "5500"]
