FROM jbaptista99/mujoco:0.0

# #COPY . /usr/local/gymnasium/
WORKDIR /usr/local/gymnasium/

# Test with PyTorch CPU build, since CUDA is not available in CI anyway
#RUN pip install gymnasium[all,testing] --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install gymnasium
RUN pip install gymnasium-robotics
# For compatibility, cython needs to be less than version 3
RUN pip install stable-baselines3
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

RUN mkdir /usr/local/gymnasium/logs
RUN mkdir /usr/local/gymnasium/models
RUN mkdir /usr/local/gymnasium/code

ENV PYTHONPATH="/usr/local/gymnasium/code"
ENV PHD_ROOT="/usr/local/gymnasium/code"
ENV PHD_MODELS="/usr/local/gymnasium/models"