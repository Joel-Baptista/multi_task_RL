from gymnasium import Env, RewardWrapper
from typing import SupportsFloat
from typing import TypeVar
from typing import Any
import numpy as np

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class   CORRECT_SUCCESS_WRAP(RewardWrapper):
    def __init__(self, env: Env, reward_scale: float = 1.0):
        super().__init__(env)
        self.env = env
        self.reward_scale = reward_scale

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        observation, reward, terminated, truncated, info = self.env.step(action)

        hand_ball_diff = observation[30:33] 
        
        # Correct env's error. It adds instead of subtracting, so we subtract two times to correct it.
        reward += - 0.1 * 2 * np.linalg.norm(hand_ball_diff)   
        
        new_info = {}
        success_key = None
        for key in info.keys():
            if "success" in key:
                success_key = key
                break
        
        if success_key is not None:
            new_info["is_success"] = info[success_key]
        
        return observation, reward, terminated, truncated, new_info
