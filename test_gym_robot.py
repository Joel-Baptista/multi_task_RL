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
from utils.common import DotDict, model_class_from_str, class_from_str

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
   parser.add_argument('-id', '--identifier', type=str, default='')

   arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
   args = vars(parser.parse_args(args=arglist))

   if args["experiment_name"] is None:
      args["experiment_name"] = "baseline"
      print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")

   experiment_name = args['experiment_name']
   experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}{args["identifier"]}'    

   # load train config.
   PHD_ROOT = os.getenv("PHD_ROOT")
   sys.path.append(PHD_ROOT)
   cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_name}/test.yaml"

   with open(cfg_path) as f:
      cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))

   if not os.path.exists(experiment_path):
      raise Exception(f"Results from experiment '{experiment_name}' does not exist in path: {experiment_path}")

   # create folder to the results.
   print(f"Path create: {experiment_path}")

   env = gym.make("FrankaKitchen-v1", render_mode="human",tasks_to_complete=["microwave"])
   env = ObsWrap(env)
   env.reset()
   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

   device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
   print(f"device:{device}")
   print(cfg.algorithm.args)
   model = PPO.load(f"{experiment_path}/model0.zip", env)

   print(model.policy)

   score = 0
   i = 0
   for _ in range(2000):
      action, _state = model.predict(obs, deterministic=True)

      observation, reward, terminated, truncated, info = env.step(action)

      score += reward
      obs = observation

      if terminated or truncated:
         i += 1
         print(f"episode: {i} with score {score}")
         score = 0
         obs, info = env.reset()
   env.close()

   
if __name__ == '__main__':
    main() 


