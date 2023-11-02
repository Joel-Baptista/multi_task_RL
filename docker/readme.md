## Build the image

    docker build . -t username/projectname:0.0 -f cuda.Dockerfile

    docker push username/projectname:0.0

    docker pull username/projectname:0.0

## Spawn the container - interactive

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /directory/on/host:/directory/on/guest username/projectname bash

## Attach to a running container

    docker attach <container_id>

## Dettach from running container

    press CRTL-p & CRTL-q