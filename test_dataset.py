import minari
import torch
import torch.nn as nn

# from utils.dataloader import collate_fn
from gymnasium import spaces
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import numpy as np
import time
from tqdm import tqdm
import math
import wandb
import os
import shutil
import time

from utils.common import DotDict, model_class_from_str, class_from_str
from wraps.observation.observation_wrap import OBSERVATION_WRAP

TASKS = ['microwave']
# DATASETS = ["kitchen-complete-v1", "kitchen-mixed-v1", "kitchen-partial-v1"]
DATASETS = ["kitchen-complete-v1"]
DEGUB = False
args = {
    "epochs": 10_000,
    "hidden_dims": [4096, 4096, 4096, 4096, 2048, 1024, 512],
    "hidden_activation": "ReLU",
    "dropout": 0.2,
    "weight_decay" :1e-5,
    "std": "auto", # "auto" for adaptable; number for fixed
}

_LOG_2PI = math.log(2 * math.pi)
# _LOG_2PI = 0

def collate_episode_fn(batch):
    batch_tensor = {
        "id": np.array([x.id for x in batch]),
        "seed": np.array([x.seed for x in batch]),
        "total_timesteps": np.array([x.total_timesteps for x in batch]),
        "observations": [x.observations["observation"] for x in batch],
        "actions": [x.actions for x in batch],
        "rewards": [x.rewards for x in batch],
        "terminations": [x.terminations for x in batch],
        "truncations": [x.truncations for x in batch]
    }

    return batch_tensor

def collate_fn(batch):
    
    batch_tensor = {
        "id": np.array([x.id for x in batch]),
        "seed": np.array([x.seed for x in batch]),
        "total_timesteps": np.array([x.total_timesteps for x in batch]),
        "observations": [x.observations["observation"] for x in batch],
        "actions": [x.actions for x in batch],
        "rewards": [x.rewards for x in batch],
        "terminations": [x.terminations for x in batch],
        "truncations": [x.truncations for x in batch]
    }
    batch_transitions = []
    
    states_actions = np.zeros((sum(batch_tensor["total_timesteps"]), 
                              batch[0].observations["observation"].shape[1] + batch[0].actions.shape[1]))
    states_next = np.zeros((sum(batch_tensor["total_timesteps"]), 
                              batch[0].observations["observation"].shape[1]))

    count = 0
    for i in range(0, len(batch_tensor["observations"])):
        observations = batch_tensor["observations"][i]
        actions = batch_tensor["actions"][i]
        
        for j in range(0, observations.shape[0] - 1):
            
            state_action = np.concatenate((observations[j], actions[j]))
        
            states_actions[count] = state_action
            states_next[count] = observations[j+1]

            count += 1

    shuffler = np.random.permutation(len(states_actions))

    states_actions_shuffled = states_actions[shuffler]
    states_next_shuffled = states_next[shuffler]
    
    return states_actions_shuffled,  states_next_shuffled

full_dataset = None
for dataset_name in DATASETS:
    try:
        dataset = minari.load_dataset(dataset_name)
    except:
        dataset = minari.load_dataset(dataset_name, download=True)

    if full_dataset is None:
        full_dataset = dataset
    else:
        full_dataset = ConcatDataset([full_dataset, dataset])


# print(len(full_dataset))
# print(full_dataset)
# train_idx = int(0.8 * len(full_dataset))
# val_idx = len(full_dataset) - train_idx

# print(train_idx)
# print(val_idx)
# train_dataset, val_dataset = random_split(full_dataset, [train_idx, val_idx])

train_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_episode_fn)
# val_loader = DataLoader(val_dataset, batch_size=1, num_workers=3)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

env = dataset.recover_environment()
env = OBSERVATION_WRAP(env)

st = time.time()

min_val_loss = np.inf

for batch in train_loader:
    print("----------------------")
    for state in batch["observations"][0]:
        print(env.robot_env.get_state())

    # states_actions = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
    # states_next = torch.as_tensor(batch[1], dtype=torch.float32, device=device)






print(time.time() - st)
