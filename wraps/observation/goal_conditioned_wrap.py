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
        # print(f"Post flatten: {self.flatten_obs(observation)}")
        return np.array(self.flatten_obs(observation))

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
