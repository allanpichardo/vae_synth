FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y libsndfile1-dev

RUN python -m pip install --upgrade pip
RUN python -m pip install librosa kapre matplotlib