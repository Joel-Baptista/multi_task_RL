#!/usr/bin/env python3

import argparse
import os
from statistics import mean
import sys
import shutil
import yaml
import pandas as pd
import torch
from pathlib import Path
from colorama import Fore
from sklearn.model_selection import StratifiedKFold
import json
import time
import math
import wandb
import gymnasium as gym
from stable_baselines3 import PPO
import torch as T

from models import *
from utils.common import DotDict, model_class_from_str

def main():
    parser = argparse.ArgumentParser(description='Train asl2text models.')
    parser.add_argument('-en', '--experiment_name', type=str, required=True)
    parser.add_argument('-ow', '--overwrite', action='store_true')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))
    
    overwrite = args['overwrite']
    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}'
    
    # load train config.
    PHD_ROOT = os.getenv("PHD_ROOT")
    sys.path.append(PHD_ROOT)
    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_name}/train.yaml"
    with open(cfg_path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))
    if os.path.exists(experiment_path):
        if overwrite:
            shutil.rmtree(experiment_path)
            print(f'Removing original {experiment_path}')
        else:
            print(f'{experiment_path} already exits. ')
            raise Exception('Experiment name already exists. If you want to overwrite, use flag -ow')

    # create folder to the results.
    os.makedirs(experiment_path)
    print(f"Path create: {experiment_path}")

    if not cfg.debug: logger = wandb.init(project=cfg.project, config=cfg, name=f"{experiment_name}")

    # model = model_class_from_str(cfg.model.type)

    env = gym.make("FetchReach-v2", max_episode_steps=100)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    model = PPO("MultiInputPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=1_000_000)
    model.save("model")

    env.close()

if __name__ == '__main__':
    main() 
