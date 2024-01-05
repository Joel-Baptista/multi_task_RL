#!/usr/bin/env python3

# System
import argparse
import os
import sys
import shutil
from typing import Any
import yaml
from colorama import Fore
import math
import numpy as np
import time

# Logs
import wandb
from wandb.integration.sb3 import WandbCallback

# Reinforcement Learning
import gymnasium as gym
from gymnasium import Env, ObservationWrapper, spaces
from stable_baselines3 import PPO, SAC
import torch as T

# My Own
from models.testings.PPO_exp import CostumAC
from models import *
from utils.common import DotDict, model_class_from_str, class_from_str
from utils.env import add_wrappers

def main():
    parser = argparse.ArgumentParser(description='Train asl2text models.')
    parser.add_argument('-en', '--experiment_name', type=str)
    parser.add_argument('-id', '--identifier', type=str, default='')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")

    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}'    

    # load train config.
    PHD_ROOT = os.getenv("PHD_ROOT")
    sys.path.append(PHD_ROOT)
    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_name}/test.yaml"
    
    experiment_path += args['identifier']
    print(experiment_path)
    with open(cfg_path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))

    if not os.path.exists(experiment_path):
        raise Exception(f"Results from experiment '{experiment_name}' does not exist in path: {experiment_path}")

    # create folder to the results.
    print(f"Path create: {experiment_path}")

    env = gym.make(cfg.env.name, render_mode="human",**cfg.env.args)
    env.metadata['render_fps'] = 70
    
    env = add_wrappers(env, cfg.env.wraps)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
    print(f"device:{device}")
    print(cfg.algorithm.args)

    print(cfg.algorithm.module)
    print(f"Algorithm class: {class_from_str(cfg.algorithm.module, cfg.algorithm.name)}")
    algorithm_class = class_from_str(cfg.algorithm.module, cfg.algorithm.name)

    print(f"Policy class: {class_from_str(cfg.policy.module, cfg.policy.name)}")
    policy_class = class_from_str(cfg.policy.module, cfg.policy.name)

    # model = algorithm_class(policy_class, env, verbose=1,**cfg.algorithm.args)
    if "model_path" in cfg['algorithm']['args'].keys():
            cfg["algorithm"]["args"]["model_path"] = experiment_path

    model = algorithm_class(policy_class, 
                            env, 
                            verbose=1,
                            **cfg.algorithm.args)
    
    try:
        model.load(f"{experiment_path}")
    except: #TODO find more elegant fix
        print(f"{Fore.YELLOW}Stable Baselines models need to specify 'model.zip'{Fore.RESET}")
        model.load(f"{experiment_path}/model.zip")
    
    print(model.policy)

    score = 0
    i = 0
    while True:
        st = time.time()
        action, _state = model.predict(obs, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)

        score += reward
        obs = observation
        
        if terminated or truncated:
            i += 1
            if i >= cfg.num_test: break

            print(f"episode: {i} with score {score}")
            score = 0
            obs, info = env.reset()
    env.close()

   
if __name__ == '__main__':
    main() 


