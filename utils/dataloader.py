import torch
import numpy as np

def collate_fn(batch):

    batch_tensor = {
        "id": np.array([x.id for x in batch]),
        "seed": np.array([x.seed for x in batch]),
        "total_timesteps": np.array([x.total_timesteps for x in batch]),
        # "observations": [x.observations["observation"] for x in batch],
        "observations": [x.observations for x in batch],
        "actions": [x.actions for x in batch],
        "rewards": [x.rewards for x in batch],
        "terminations": [x.terminations for x in batch],
        "truncations": [x.truncations for x in batch]
    }
    batch_transitions = []
    
    # states_actions = np.zeros((sum(batch_tensor["total_timesteps"]), 
    #                           batch[0].observations["observation"].shape[1] + batch[0].actions.shape[1]))
    # states_next = np.zeros((sum(batch_tensor["total_timesteps"]), 
    #                           batch[0].observations["observation"].shape[1]))

    states_actions = np.zeros((sum(batch_tensor["total_timesteps"]), 
                              batch[0].observations.shape[1] + batch[0].actions.shape[1]))
    states_next = np.zeros((sum(batch_tensor["total_timesteps"]), 
                              batch[0].observations.shape[1]))

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


# def collate_fn(batch):
#     print(batch)
#     return {
#         "id": torch.Tensor([x.id for x in batch]),
#         "seed": torch.Tensor([x.seed for x in batch]),
#         "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
#         "observations": torch.nn.utils.rnn.pad_sequence(
#             [x.observations for x in batch],
#             batch_first=True
#         ),
#         "actions": torch.nn.utils.rnn.pad_sequence(
#             [x.actions for x in batch],
#             batch_first=True
#         ),
#         "rewards": torch.nn.utils.rnn.pad_sequence(
#             [x.rewards for x in batch],
#             batch_first=True
#         ),
#         "terminations": torch.nn.utils.rnn.pad_sequence(
#             [x.terminations for x in batch],
#             batch_first=True
#         ),
#         "truncations": torch.nn.utils.rnn.pad_sequence(
#             [x.truncations for x in batch],
#             batch_first=True
#         )
#     }
# def collate_fn(batch):
#     print(batch)
#     return {
#         "id": torch.Tensor([x.id for x in batch]),
#         "seed": torch.Tensor([x.seed for x in batch]),
#         "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
#         "observations": torch.nn.utils.rnn.pad_sequence(
#             [torch.as_tensor(x.observations) for x in batch],
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