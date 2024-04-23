FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Allow for GUI
RUN     apt-get update && apt-get install -qqy x11-apps  
RUN     export uid=$USER_GID gid=$USER_GID
RUN     mkdir -p /home/docker_user
RUN     mkdir -p /etc/sudoers.d
RUN     echo "docker_user:x:${uid}:${gid}:docker_user,,,:/home/docker_user:/bin/bash" >> /etc/passwd
RUN     echo "docker_user:x:${uid}:" >> /etc/group
RUN     echo "docker_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/docker_user
RUN     chmod 0440 /etc/sudoers.d/docker_user
RUN     chown ${uid}:${gid} -R /home/docker_user 

RUN apt install vim gnupg -y


#RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
COPY zsh-in-docker.sh /tmp
RUN /tmp/zsh-in-docker.sh \
    # -t https://github.com/denysdovhan/spaceship-prompt \
    -t "lukerandall" \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'
\
ENTRYPOINT [ "/bin/zsh" ]
CMD ["-l"]
# CMD xeyes