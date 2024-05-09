#!/usr/bin/env python3

# System
import argparse
import sys
from tqdm import tqdm

# Logs
import wandb

# Reinforcement Learning
import gymnasium as gym

# My Own
from utils.common import class_from_str, setup_experiment
from utils.env import add_wrappers, make_env
from utils.sb import setup_callbacks


def main():
    parser = argparse.ArgumentParser(description='Train asl2text models.')
    parser.add_argument('-en', '--experiment_name', type=str)
    parser.add_argument('-ow', '--overwrite', action='store_true')
    parser.add_argument('-id', '--identifier', type=str, default='') # if "auto", then auto assigns
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-a', '--add', action='store_true')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))
    
    if args["debug"]: args["identifier"] = "debug"
    elif args["add"]: args["identifier"] = "auto"

    experiment_name, experiment_path, cfg = setup_experiment(args)

    print(cfg.algorithm.args)

    # env, record_env = make_env(**cfg.env)
    
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
    if not args['debug']:
        # run = wandb.init(
        #     project=cfg.project, 
        #     sync_tensorboard=True,
        #     config=cfg,
        #     name=f"{cfg.algorithm.name}_{experiment_name}{args['identifier']}"
        #     )

        if "model_path" in cfg['algorithm']['args'].keys():
            cfg["algorithm"]["args"]["model_path"] = experiment_path

        model = algorithm_class(policy_class, 
                                env, 
                                verbose=0, 
                                # tensorboard_log=f"{experiment_path}/{run.id}",
                                **cfg.algorithm.args)
        # Setup Callbacks

        callbacks = setup_callbacks(cfg, experiment_path, record_env, model)
        
        print(model.policy)
        model.learn(
            total_timesteps=cfg.total_timesteps,
            log_interval=1,
            callback=callbacks
            )
    else:
        model = algorithm_class(policy_class, 
                                env, 
                                verbose=1, 
                                **cfg.algorithm.args)
        print(model.policy)
        for i in tqdm(range(0, cfg.checkpoints)): 
            model.learn(total_timesteps=int(cfg.total_timesteps / cfg.checkpoints))
            
    wandb.finish()
    env.close()

if __name__ == '__main__':
    main() 
