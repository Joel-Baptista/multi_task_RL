#!/bin/bash

cd docker
export EXPERIMENT=$1
export TRAIN_REPLICAS=$2
docker compose up -d
