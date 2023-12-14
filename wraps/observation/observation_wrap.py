from gymnasium import Env, ObservationWrapper, spaces
from typing import TypeVar, Any
import math
import numpy as np


ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class OBSERVATION_WRAP(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env
        dim = 0
        
        self.observation_space = self.observation_space["observation"]

    def observation(self, observation: Any) -> Any:
        return np.array(observation["observation"])
