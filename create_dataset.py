import minari
import gymnasium as gym
from tqdm import tqdm
import numpy as np
from wraps.observation.observation_wrap import OBSERVATION_WRAP
from wraps.reward.correct_success_wrap import CORRECT_SUCCESS_WRAP

total_steps = 1_000_000

dataset_id = "kitchen-dataset-v0"

# minari.delete_dataset(dataset_id)
local_datasets = minari.list_local_datasets()
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)

env=gym.make('AdroitHandRelocate-v1')
# env = CORRECT_SUCCESS_WRAP(env)
env = minari.DataCollectorV0(env=env)
env.reset()

for _ in tqdm(range(total_steps)):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        env.reset()

dataset = minari.create_dataset_from_collector_env(
    collector_env=env,
    algorithm_name="Random",
    dataset_id="adroit-dataset-v1", 
    code_permalink="https://github.com/Joel-Baptista/multi_task_RL/create_dataset.py",
    author="Joel Baptista",
    author_email="joelbaptista@ua.pt"
    )
    
    