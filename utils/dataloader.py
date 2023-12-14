import torch

def collate_fn(batch):
    print(batch)
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [x.observations for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [x.actions for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [x.rewards for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [x.terminations for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [x.truncations for x in batch],
            batch_first=True
        )
    }
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