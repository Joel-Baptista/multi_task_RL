import minari
import torch
# from utils.dataloader import collate_fn
from gymnasium import spaces
from torch.utils.data import DataLoader
import numpy as np

TASKS = ['microwave']

def flatten_space(obs):
    result=[]
    for key in obs:
        if isinstance(obs[key], spaces.Dict) or isinstance(obs[key], dict):
            result.extend(flatten_space(obs[key]))
        else:
            result.append(obs[key])
    return result

def flatten_obs(obs):
    result=[]
    for key in obs:
        if isinstance(obs[key], dict):
            result.extend(flatten_obs(obs[key]))
        else:
            result.extend(list(obs[key]))
    
    return result
    # return np.array(result)


def filter_obs(obs: dict) -> np.array:
    observations = []
    for i in range(0, len(obs['observation'])):
        observation = []
        # print(obs['observation'][i])
        observation.extend(obs['observation'][i])
        for task in TASKS:
            # print(obs['desired_goal'][task][i])
            observation.extend(obs['desired_goal'][task][i])
            # print(obs['achieved_goal'][task][i])
            observation.extend(obs['achieved_goal'][task][i])

        observations.append(observation)
    # print(observations)
    return np.array(observations)

def collate_fn(batch):
    # print(batch)
    
    # for x in batch:
    #     observations = []
    #     for i in range(0, len(x.observations['observation'])):
    #         observation = []
    #         print(x.observations['observation'][i])
    #         observation.extend(x.observations['observation'][i])
    #         for task in TASKS:
    #             print(x.observations['desired_goal'][task][i])
    #             observation.extend(x.observations['desired_goal'][task][i])
    #             print(x.observations['achieved_goal'][task][i])
    #             observation.extend(x.observations['achieved_goal'][task][i])

    #         observations.append(np.array(observation))

    # print(observations)

    return {
        "id": torch.Tensor(np.array([x.id for x in batch])),
        "seed": torch.Tensor(np.array([x.seed for x in batch])),
        "total_timesteps": torch.Tensor(np.array([x.total_timesteps for x in batch])),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(filter_obs(x.observations)) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }

try:
    dataset = minari.load_dataset('kitchen-mixed-v1')
except:
    dataset = minari.load_dataset('kitchen-mixed-v1', download=True)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

print(dataloader)

for batch in dataloader:
    print(batch["id"])