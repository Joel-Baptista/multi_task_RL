import minari
import torch
# from utils.dataloader import collate_fn
from gymnasium import spaces
from torch.utils.data import DataLoader
import numpy as np
import random
import time

DATASET = "kitchen-complete-v1"

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

    # for i in range(0, len(batch_tensor["observations"])):
    #     # print(batch_tensor["observations"][i])
    #     observations = batch_tensor["observations"][i]
    #     actions = batch_tensor["actions"][i]
    #     transition = {"state": None, "action": None, "next_state": None}
    #     for j in range(0, observations.shape[0] - 1):
    #         transition = {"state": np.array(observations[j]), 
    #                       "action": np.array(actions[j]), 
    #                       "next_state": np.array(observations[j+1])}
    #         batch_transitions.append(transition)

    # random.shuffle(batch_transitions)

    
    return states_actions_shuffled,  states_next_shuffled

# def collate_fn(batch):
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

try:
    dataset = minari.load_dataset(DATASET)
except:
    dataset = minari.load_dataset(DATASET, download=True)

print(len(dataset))
train_idx = int(0.8 * len(dataset))
val_idx = len(dataset) - train_idx

print(train_idx)
print(val_idx)

dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

print(dataloader)
st = time.time()

for batch in dataloader:
    states_actions = torch.as_tensor(batch[0])
    states_next = torch.as_tensor(batch[1])



print(time.time() - st)