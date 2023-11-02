FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 vim -y

WORKDIR /root

ENV USER="deep"

RUN mkdir /root/logs
RUN mkdir /root/models

ENV PYTHONPATH="/root/RL-Skid2Mid"

