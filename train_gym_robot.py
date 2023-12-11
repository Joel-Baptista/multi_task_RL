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
from typing import SupportsFloat, TypeVar
from tqdm import tqdm

# Logs
import wandb
from wandb.integration.sb3 import WandbCallback

# Reinforcement Learning
import gymnasium as gym
from gymnasium import error
from gymnasium import Env, ObservationWrapper, spaces, RewardWrapper
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv, FrankaRobot
# from stable_baselines3 import PPO
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.ppo.policies import MlpPolicy
import torch as T

# My Own
from models.testings.PPO_exp import CostumAC
from models import *
from utils.common import DotDict, model_class_from_str, class_from_str

# Mujoco
try:
    import mujoco
    from mujoco import MjData, MjModel, mjtObj
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco")

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class RewWrap(RewardWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        observation, reward, terminated, truncated, info = self.env.step(action)

        # delta_objective = abs(observation['achieved_goal']['microwave'][0] - observation['desired_goal']['microwave'][0])

        reward = 10 * reward
        return observation, self.reward(reward), terminated, truncated, info


    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        # print(self.env.unwrapped.model)
        # print(self.env.unwrapped.data)

        # joint_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:panda0_joint5")
        # joint_type = self.env.unwrapped.model.jnt_type[joint_id]
        # joint_addr = self.env.unwrapped.model.jnt_qposadr[joint_id]
        
        # if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        #     ndim = 7
        # elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        #     ndim = 4
        # else:
        #     assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        #     ndim = 1

        # start_idx = joint_addr
        # end_idx = joint_addr + ndim

        # print(self.env.unwrapped.data.qpos[start_idx:end_idx].copy())

        return reward
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
    parser.add_argument('-d', '--debug', action='store_true')

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
    
    if not args['debug']:
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
    
    device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
    env = gym.make(cfg.env.name, **cfg.env.args)
    # env = RewWrap(env)
    env = ObsWrap(env)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(env.observation_space.shape)
    print(env.reward_range)
    
    print(f"device:{device}")
    print(cfg.algorithm.args)

    print(cfg.algorithm.module)
    print(f"Algorithm class: {class_from_str(cfg.algorithm.module, cfg.algorithm.name)}")
    algorithm_class = class_from_str(cfg.algorithm.module, cfg.algorithm.name)

    print(f"Policy class: {class_from_str(cfg.policy.module, cfg.policy.name)}")
    policy_class = class_from_str(cfg.policy.module, cfg.policy.name)

    if not args['debug']:
        run = wandb.init(
            project=cfg.project, 
            sync_tensorboard=True,
            config=cfg,
            name=f"{cfg.algorithm.name}_{experiment_name}"
            )

        model = algorithm_class(policy_class, env, verbose=1, tensorboard_log=f"{experiment_path}/{run.id}",**cfg.algorithm.args)
    
        print(model.policy)
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=WandbCallback(
                verbose=2,
                model_save_path=experiment_path,
                model_save_freq= int(cfg.total_timesteps / cfg.checkpoints),
                log = "all"
                )
            )
    else:
        model = algorithm_class(policy_class, env, verbose=1, **cfg.algorithm.args)
        print(model.policy)
        for i in tqdm(range(0, cfg.checkpoints)): 
            model.learn(total_timesteps=int(cfg.total_timesteps / cfg.checkpoints))
            
    wandb.finish()
    env.close()

if __name__ == '__main__':
    main() 
