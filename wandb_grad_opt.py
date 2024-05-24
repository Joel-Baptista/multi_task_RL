#!/usr/bin/env python3

# System
import argparse
import os
import sys
import shutil
import copy
import numpy as np

import yaml
from colorama import Fore
from tqdm import tqdm

# Logs
import wandb
from wandb.integration.sb3 import WandbCallback

# Reinforcement Learning
import gymnasium as gym
import torch as T
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

# My Own
from utils.common import DotDict, model_class_from_str, class_from_str, setup_experiment
from utils.env import add_wrappers
from callbacks.video_recorder import VideoRecorder
from callbacks.early_stopping import EarlyStopping


args = {
    "debug": True,
    "experiment_name": "reloc_sac",
    "overwrite": False,
    "identifier": ""
}

nested_args = {"policy_kwargs":  ["net_arch"]}

def main():
    
    experiment_name, experiment_path, log_path, cfg = setup_experiment(args)
 
    run = wandb.init(
        project=f"{cfg.project}", 
        sync_tensorboard=True,
        name=f"{cfg.algorithm.name}_{experiment_name}"
        )

    env = gym.make(cfg.env.name,**cfg.env.args)
    record_env = gym.make(cfg.env.name, render_mode="rgb_array",**cfg.env.args)
    
    env = add_wrappers(env, cfg.env.wraps)
    record_env = add_wrappers(record_env, cfg.env.wraps)

    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    print(env.observation_space.shape)
    print(env.reward_range)
    
    print(cfg.algorithm.module)
    print(f"Algorithm class: {class_from_str(cfg.algorithm.module, cfg.algorithm.name)}")
    algorithm_class = class_from_str(cfg.algorithm.module, cfg.algorithm.name)

    print(f"Policy class: {class_from_str(cfg.policy.module, cfg.policy.name)}")
    policy_class = class_from_str(cfg.policy.module, cfg.policy.name)

    cfg.algorithm.args.model_path = experiment_path
    print(cfg.algorithm.args.model_path)

    config_directory = dict(run.config)
    print(run.config)
    for nested_key_args in nested_args.keys():
        config_directory[nested_key_args] = {}
        for key_arg in nested_args[nested_key_args]:
            config_directory[nested_key_args][key_arg] = run.config[key_arg]
            print(key_arg)
            del config_directory[key_arg]

    config_directory["device"] = cfg.algorithm.args.device
    print(config_directory)
    early_stopping = EarlyStopping(
            verbose=2
            )
    
    callbacks =  [early_stopping]
    
    model = algorithm_class(policy_class, 
                            env, 
                            verbose=0, 
                            tensorboard_log=f"{experiment_path}/{run.id}",
                            **config_directory)
    # Setup Callbacks
    print(f"Device: {model.device}")
    print(model.policy)
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
    )

    # scores = []
    # score = 0
    # i = 0
    # n_test = 100
    # env.reset()

    # while True:
        
    #     # print(obs)
    #     action, _state = model.predict(obs, deterministic=True)
    #     # print(action)

    #     observation, reward, terminated, truncated, info = env.step(action)

    #     score += reward
    #     obs = observation
        
    #     if terminated or truncated:
    #         i += 1
    #         scores.append(score)

    #         if i >= n_test: break

    #         print(f"episode: {i} with score {score}")
    #         score = 0
    #         obs, info = env.reset()

    # wandb.log({"test/mean_reward": np.mean(scores)})
    wandb.finish()
    env.close()

if __name__ == '__main__':
    main() 
