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
from utils.common import DotDict, model_class_from_str, class_from_str, setup_test
from utils.env import add_wrappers
import cv2 as cv

def main():
    parser = argparse.ArgumentParser(description='Train asl2text models.')
    parser.add_argument('-en', '--experiment_name', type=str)
    parser.add_argument('-id', '--identifier', type=str, default='')
    parser.add_argument('-b', '--best', action="store_true", default=False)
    parser.add_argument('-r', '--record', action="store_true", default=False)

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    print(args["best"])
    
    log_path, experiment_path, cfg = setup_test(args)

    render_mode = "human"
    if args["record"]: render_mode = "rgb_array"

    env = gym.make(cfg.env.name, render_mode=render_mode, **cfg.env.args)
    # env.metadata['render_fps'] = 240
    
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
    
    if args["best"]: experiment_path += "/best_model"
    files = os.listdir(experiment_path)

    is_single_model = False
    for file in files:
        if 'model.zip' in file: 
            is_single_model = True
            if args["best"]: 

                # evaluations = np.load(f"{experiment_path}/evaluations.npz")
                # print(evaluations)
                # print(f"timesteps: {evaluations['timesteps']}")
                # print(f"results: {evaluations['results']}")
                # print(f"ep_lengths: {evaluations['ep_lengths']}")
                # print(f"successes: {evaluations['successes']}")

                experiment_path += "/best_model.zip"
                break
            else:
                experiment_path += "/model.zip"
                break

            

    print(experiment_path)
    model = model.load(experiment_path)

    # try:
    #     model.load(f"{experiment_path}")
    # except: #TODO find more elegant fix
    #     print(f"{Fore.YELLOW}Stable Baselines models need to specify 'model.zip'{Fore.RESET}")
    #     model.load(f"{experiment_path}/model.zip")
    
    print(model.policy)
    scores = []
    finished = 0

    score = 0
    i = 0
    fps = 30
    
    if args["record"]:
        video = cv.VideoWriter(f"{log_path}/test_video.mp4",
                                    cv.VideoWriter_fourcc(*"mp4v"),
                                    30,
                                    (480, 480))

    while True:
        st = time.time()
        if args["record"]:
            img = env.render()
            video.write(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        # print(obs)
        action, _state = model.predict(obs, deterministic=True)
        # print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        obs = observation
        
        if terminated or truncated:
            i += 1
            scores.append(score)
            finished += 1

            if i >= cfg.num_test: break

            # print(obs[28:31])
            print(f"episode: {i} with score {score}")
            score = 0
            obs, info = env.reset()

        while time.time() - st  < (1 / fps) and not args["record"]:
            continue

    env.close()
    if args["record"]: video.release()

    print(f"Overall mean episode reward: {np.mean(scores)}")
    print(f"Finished episodes percentage: {finished / cfg.num_test * 100} %")
   
if __name__ == '__main__':
    main() 


