import minari
import gymnasium as gym
from tqdm import tqdm

total_steps = 1_000_000
dataset_id = "fetch-dataset-v0"

local_datasets = minari.list_local_datasets()
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)

env=gym.make("FetchPickAndPlace-v2", reward_type="sparse", max_episode_steps=50)
env = minari.DataCollectorV0(env=env, record_infos=True)

env.reset()

for _ in tqdm(range(total_steps)):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.reset()

dataset = minari.create_dataset_from_collector_env(
    collector_env=env,
    algorithm_name="Random",
    dataset_id="fetch-dataset-v0", 
    code_permalink="https://github.com/Joel-Baptista/multi_task_RL/create_dataset.py",
    author="Joel Baptista",
    author_email="joelbaptista@ua.pt"
    )
    
    