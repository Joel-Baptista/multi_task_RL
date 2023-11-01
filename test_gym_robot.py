import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FetchPickAndPlace-v2", render_mode="human", max_episode_steps=100)
obs, _ = env.reset()

model = PPO("MultiInputPolicy", env, verbose=1)
model.load("model")

for _ in range(1000):
   action, _state = model.predict(obs, deterministic=True)
   observation, reward, terminated, truncated, info = env.step(action)

   obs = observation
   if terminated or truncated:
      obs, info = env.reset()
env.close()