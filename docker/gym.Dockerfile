FROM jbaptista99/mujoco:0.0

# #COPY . /usr/local/gymnasium/
WORKDIR /home/gymnasium/

# Test with PyTorch CPU build, since CUDA is not available in CI anyway
#RUN pip install gymnasium[all,testing] --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu

RUN     apt-get update && apt-get install -qqy x11-apps 
# 
RUN     export uid=1000 gid=1000
RUN     mkdir -p /home/docker_user
RUN     mkdir -p /etc/sudoers.d
RUN     echo "docker_user:x:${uid}:${gid}:docker_user,,,:/home/docker_user:/bin/bash" >> /etc/passwd
RUN     echo "docker_user:x:${uid}:" >> /etc/group
RUN     echo "docker_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/docker_user
RUN     chmod 0440 /etc/sudoers.d/docker_user
RUN     chown ${uid}:${gid} -R /home/docker_user 

RUN pip install gymnasium
RUN pip install gymnasium-robotics
# For compatibility, cython needs to be less than version 3
RUN pip install stable-baselines3
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN pip install wandb -U --pre

RUN apt install vim -y

COPY test.py /tmp/
RUN python3 /tmp/test.py

# RUN mkdir /usr/local/gymnasium/logs
# RUN mkdir /usr/local/gymnasium/models
# RUN mkdir /usr/local/gymnasium/code
RUN mkdir /home/gymnasium/logs
RUN mkdir /home/gymnasium/models
RUN mkdir /home/gymnasium/code

ENV PYTHONPATH="/home/gymnasium/code"
ENV PHD_ROOT="/home/gymnasium/code"
ENV PHD_MODELS="/home/gymnasium/models"

