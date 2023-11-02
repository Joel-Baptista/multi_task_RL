# A Dockerfile that sets up a full Gymnasium install with test dependencies
#ARG PYTHON_VERSION
#FROM python:$PYTHON_VERSION

FROM python:3.11
#FROM rch/pytorch:2.0.0-cuda11.7-cudnn8-devel
#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
#FROM gw000/debian-cuda

SHELL ["/bin/bash", "-o", "pipefail", "-c"]


RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    xvfb unzip patchelf ffmpeg cmake swig \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Download mujoco
    && mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

# Build mujoco-py from source. Pypi installs wheel packages and Cython won't recompile old file versions in the Github Actions CI.
# Thus generating the following error https://github.com/cython/cython/pull/4428
#RUN pip install --upgrade pip
RUN git clone https://github.com/openai/mujoco-py.git\
    && cd mujoco-py \
    && pip install -e . --no-cache

#COPY . /usr/local/gymnasium/
WORKDIR /usr/local/gymnasium/

# Test with PyTorch CPU build, since CUDA is not available in CI anyway
#RUN pip install gymnasium[all,testing] --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install gymnasium
RUN pip install gymnasium-robotics
# For compatibility, cython needs to be less than version 3
RUN pip install 'cython<3'
#RUN pip install stable-baselines3
#RUN pip install wandb

RUN mkdir /usr/local/gymnasium/logs
RUN mkdir /usr/local/gymnasium/models
RUN mkdir /usr/local/gymnasium/code

ENV PYTHONPATH="/usr/local/gymnasium/code"

# ENTRYPOINT ["/usr/local/gymnasium/"]
