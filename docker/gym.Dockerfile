FROM jbaptista99/mujoco:1.0

# Test with PyTorch CPU build, since CUDA is not available in CI anyway
#RUN pip install gymnasium[all,testing] --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu

# ARG USERNAME=kuriboh
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID

# Set up user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     && apt-get update \
#     && apt-get install -y sudo wget \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME 
    #
    # Clean up
    # && apt-get autoremove -y \
    # && apt-get clean -y \
    # && rm -rf /var/lib/apt/lists/*

RUN pip install gymnasium
RUN pip install gymnasium-robotics
# For compatibility, cython needs to be less than version 3
RUN pip install stable-baselines3
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN pip install wandb -U --pre

COPY test.py /tmp/
RUN python3 /tmp/test.py

RUN cp /root/.zshrc /home/$USERNAME/
RUN cp /root/.profile /home/$USERNAME/
RUN cp -r /root/.oh-my-zsh /home/$USERNAME/    

RUN sed -i 's/\/root/\/home\/kuriboh/g' /home/$USERNAME/.zshrc

RUN mkdir /home/gymnasium
RUN mkdir /home/gymnasium/results
RUN mkdir /home/gymnasium/results/logs
RUN mkdir /home/gymnasium/results/models
RUN mkdir /home/gymnasium/code

ENV PYTHONPATH=/home/gymnasium/code
ENV PHD_ROOT=/home/gymnasium/code
ENV PHD_RESULTS=/home/gymnasium/results

RUN cp -r /root/.mujoco /home/$USERNAME/

WORKDIR /home/gymnasium

# USER $USERNAME
