FROM python:3.10-slim

WORKDIR /app

RUN apt-get -y update && apt-get install -y \
    python3-dev \
    apt-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools
RUN pip3 install \
    cython==3.0.6

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
