FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as fastapi

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt --no-cache-dir

COPY . /app
COPY fast_api.py /app/main.py
