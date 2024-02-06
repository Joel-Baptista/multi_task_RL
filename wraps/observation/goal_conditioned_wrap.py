from gymnasium import Env, ObservationWrapper, spaces
from typing import TypeVar, Any
import math
import numpy as np

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class GOAL_CONDITIONED_WRAP(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env
        dim = 0
        # print(env.observation_space)
        # print(self.flatten_space(env.observation_space))
        for box in self.flatten_space(env.observation_space):
            dim += box.shape[0]
        print(env.observation_space)
        self.observation_space = spaces.Box(
            low = -math.inf,
            high = math.inf,
            shape = (dim,),
            dtype= np.float64
        )

    def observation(self, observation: Any) -> Any:
        # print(f"Pre flatten: {observation}")
        # print(f"Post flatten: {self.flatten_obs(observation)[28:32]}")
        obs = self.flatten_obs(observation)
        # obs[-1] = 0.47
        # obs[-1] = 0.42469975
        return np.array(obs)

    def flatten_space(self, obs):
        result=[]
        for key in obs:
            if isinstance(obs[key], spaces.Dict) or isinstance(obs[key], dict):
                result.extend(self.flatten_space(obs[key]))
            else:
                result.append(obs[key])
        return result

    def flatten_obs(self, obs):
        result=[]
        for key in obs:
            if isinstance(obs[key], dict):
                result.extend(self.flatten_obs(obs[key]))
            else:
                result.extend(list(obs[key]))
        return result


    def filter_obs(self, obs: dict, tasks) -> np.array:
        observations = []
        for i in range(0, len(obs['observation'])):
            observation = []
            # print(obs['observation'][i])
            observation.extend(obs['observation'][i])
            for task in tasks:
                # print(obs['desired_goal'][task][i])
                observation.extend(obs['desired_goal'][task][i])
                # print(obs['achieved_goal'][task][i])
                observation.extend(obs['achieved_goal'][task][i])

            observations.append(observation)
        # print(observations)
        return np.array(observations)
