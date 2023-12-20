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

from utils.common import DotDict, model_class_from_str, class_from_str
from wraps.observation.observation_wrap import OBSERVATION_WRAP

TASKS = ['microwave']
DATASETS = ["kitchen-complete-v1", "kitchen-mixed-v1", "kitchen-partial-v1"]
DEGUB = False
experiment_path = f'{os.getenv("PHD_MODELS")}/world_model'  
args = {
    "epochs": 1000,
    "hidden_dims": [4048, 2012, 512, 206],
    "hidden_activation": "ReLU",
    "dropout": 0.0,
}

if os.path.exists(experiment_path):
    shutil.rmtree(experiment_path)
    print(f'Removing original {experiment_path}')
os.makedirs(experiment_path)

_LOG_2PI = math.log(2 * math.pi)
# _LOG_2PI = 0


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



# def collate_fn(batch):
#     # print(batch)
    
#     # for x in batch:
#     #     observations = []
#     #     for i in range(0, len(x.observations['observation'])):
#     #         observation = []
#     #         print(x.observations['observation'][i])
#     #         observation.extend(x.observations['observation'][i])
#     #         for task in TASKS:
#     #             print(x.observations['desired_goal'][task][i])
#     #             observation.extend(x.observations['desired_goal'][task][i])
#     #             print(x.observations['achieved_goal'][task][i])
#     #             observation.extend(x.observations['achieved_goal'][task][i])

#     #         observations.append(np.array(observation))

#     # print(observations)

#     return {
#         "id": torch.Tensor(np.array([x.id for x in batch])),
#         "seed": torch.Tensor(np.array([x.seed for x in batch])),
#         "total_timesteps": torch.Tensor(np.array([x.total_timesteps for x in batch])),
#         "observations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.observations["observation"]) for x in batch],
#             batch_first=True
#         ),
#         "actions": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.actions) for x in batch],
#             batch_first=True
#         ),
#         "rewards": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.rewards) for x in batch],
#             batch_first=True
#         ),
#         "terminations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.terminations) for x in batch],
#             batch_first=True
#         ),
#         "truncations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.truncations) for x in batch],
#             batch_first=True
#         )
#     }

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

if not DEGUB:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Kitchen_World_Model",
        name="world_model",
        # Track hyperparameters and run metadata
        config=args,
    )

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


print(len(full_dataset))
train_idx = int(0.7 * len(full_dataset))
val_idx = len(full_dataset) - train_idx

print(train_idx)
print(val_idx)
train_dataset, val_dataset = random_split(full_dataset, [train_idx, val_idx])

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=3,shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=3, collate_fn=collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

env = dataset.recover_environment()
env = OBSERVATION_WRAP(env)

world_model = class_from_str(f"models.world_model.mlp", "mlp".upper())(
        inp_dim = env.action_space.shape[0] + env.observation_space.shape[0], 
        outp_dim = env.observation_space.shape[0], 
        hidden_dims = args["hidden_dims"],
        hidden_activation = class_from_str("torch.nn", args["hidden_activation"]),
        dropout = args["dropout"]
        )
world_model.to(device)
print(world_model)
del env

st = time.time()

min_val_loss = np.inf

for epoch in range(0, args["epochs"]):
    print(f"Epoch {epoch + 1}")
    world_model_losses = []
    world_model_val_losses = []

    world_model.train()
    for batch in tqdm(train_loader):
        states_actions = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
        states_next = torch.as_tensor(batch[1], dtype=torch.float32, device=device)

        mu_state_next, log_std_state_next = world_model(states_actions)

        model_loss = world_model.calc_loss(mu_state_next, log_std_state_next, states_next)

        world_model.optim.zero_grad()
        model_loss.backward()
        world_model.optim.step()

        world_model_losses.append(model_loss.item())

    log_std_list = []
    mu_list = []
    world_model.eval()
    for batch in tqdm(val_loader):
        states_actions = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
        states_next = torch.as_tensor(batch[1], dtype=torch.float32, device=device)

        with torch.no_grad():
            mu_state_next, log_std_state_next = world_model(states_actions)

            model_loss = world_model.calc_loss(mu_state_next, log_std_state_next, states_next)

            world_model_val_losses.append(model_loss.item())
            log_std_list.append(log_std_state_next.mean().item())
            mu_list.append(mu_state_next.mean().item())

    mean_loss = np.array(world_model_losses).mean()
    mean_val_loss = np.array(world_model_val_losses).mean()

    print(f"Avg loss: {mean_loss}")
    print(f"Avg val loss: {mean_val_loss}")
    wandb.log({"train/avg_loss": mean_loss,
              "train/avg_val_loss": mean_val_loss,
              "train/log_std_mean":  np.array(log_std_list).mean(),
              "train/mu_mean":  np.array(mu_list).mean(),
              })
    
    if mean_val_loss < min_val_loss:
        min_val_loss = mean_val_loss
        torch.save(world_model.state_dict(), f"{experiment_path}/world_model.pt")
        print("-------------------------Saved Model-------------------------------")



print(time.time() - st)
