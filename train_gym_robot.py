import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("FetchReach-v2", render_mode="rgb_array", max_episode_steps=100)
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("model")

env.close()
# obs, _ = env.reset()
# The following always has to hold:
# assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
# assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
# assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# However goals can also be substituted:
# substitute_goal = obs["achieved_goal"].copy()
# print(substitute_goal)
# substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
# substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
# substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)

# for _ in range(1000):
#    #action = policy(observation)  # User-defined policy function
#    # action = env.action_space.sample()  # User-defined policy function 
#    action, _state = model.predict(obs, deterministic=True)
#    observation, reward, terminated, truncated, info = env.step(action)

#    obs = observation
#    if terminated or truncated:
#       obs, info = env.reset()
# env.close()
