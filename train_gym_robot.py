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

# Logs
import wandb
from wandb.integration.sb3 import WandbCallback

# Reinforcement Learning
import gymnasium as gym
from gymnasium import Env, ObservationWrapper, spaces
from stable_baselines3 import PPO
import torch as T

# My Own
from models.testings.PPO_exp import CostumAC
from models import *
from utils.common import DotDict, model_class_from_str

class ObsWrap(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env
        dim = 0
        for box in self.flatten_space(env.observation_space):
            dim += box.shape[0]

        self.observation_space = spaces.Box(
            low = -math.inf,
            high = math.inf,
            shape = (dim,),
            dtype= np.float64
        )

    def observation(self, observation: Any) -> Any:
        return np.array(self.flatten_obs(observation))

    def flatten_space(self, obs):
        result=[]
        for key in obs:
            if isinstance(obs[key], spaces.Dict) or isinstance(obs[key], dict):
                result.extend(self.flatten_space(obs[key]))
            else:
                result.append(obs[key])
        return result

    def flatten_obs(self, obs):
        result=[]
        for key in obs:
            if isinstance(obs[key], dict):
                result.extend(self.flatten_obs(obs[key]))
            else:
                result.extend(list(obs[key]))
        return result

def main():
    parser = argparse.ArgumentParser(description='Train asl2text models.')
    parser.add_argument('-en', '--experiment_name', type=str)
    parser.add_argument('-ow', '--overwrite', action='store_true')
    parser.add_argument('-id', '--identifier', type=str, default='')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")

    overwrite = args['overwrite']
    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}{args["identifier"]}'    

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
    if not cfg.debug: 
        run = wandb.init(
            project=cfg.project, 
            sync_tensorboard=True,
            config=cfg,
            name=f"{cfg.algorithm.name}_{experiment_name}"
            )

    env = gym.make("FrankaKitchen-v1", tasks_to_complete=["microwave"])
    env = ObsWrap(env)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
    print(f"device:{device}")
    print(cfg.algorithm.args)
    model = PPO(CostumAC, env, verbose=1, tensorboard_log=f"{experiment_path}/{run.id}",**cfg.algorithm.args)
    print(model.policy)

    if not cfg.debug:

        for i in range(0, cfg.checkpoints): 
            model.learn(
                total_timesteps=int(cfg.total_timesteps / cfg.checkpoints),
                callback=WandbCallback(
                    verbose=2
                    )
                )
            model.save(f"{experiment_path}/model{i}")
    else:
        model.learn(total_timesteps=cfg.total_timesteps)

    wandb.finish()
    env.close()

if __name__ == '__main__':
    main() 
