# A Dockerfile that sets up a full Gymnasium install with test dependencies
#ARG PYTHON_VERSION
#FROM python:$PYTHON_VERSION

#FROM python:3.11
FROM jbaptista99/base_env:0.0
#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
#FROM gw000/debian-cuda

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev libglew-dev \
    xvfb unzip patchelf ffmpeg cmake swig 
# Did not come with the new image
RUN apt-get install -y wget git 
RUN apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Download mujoco
    && mkdir /root/.mujoco \
#    && mkdir /root/.mujoco/mujoco210 \
    && cd /root/.mujoco \
#COPY ./mujoco210 /root/.mujoco/mujoco210
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so"
# ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so:$LD_PRELOAD"
# Build mujoco-py from source. Pypi installs wheel packages and Cython won't recompile old file versions in the Github Actions CI.
# Thus generating the following error https://github.com/cython/cython/pull/4428
#RUN pip install --upgrade pip
RUN git clone https://github.com/openai/mujoco-py.git\
    && cd mujoco-py \
    && pip install -e . --no-cache

RUN pip install 'cython<3'

