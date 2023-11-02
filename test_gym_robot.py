import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FetchReach-v2", render_mode="human", max_episode_steps=100)
obs, _ = env.reset()

model = PPO.load("model.zip", env)

score = 0
i = 0
for _ in range(2000):
   action, _state = model.predict(obs, deterministic=True)

   observation, reward, terminated, truncated, info = env.step(action)

   score += reward
   obs = observation

   if terminated or truncated:
      i += 1
      print(f"episode: {i} with score {score}")
      score = 0
      obs, info = env.reset()
env.close()