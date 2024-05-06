from gymnasium_robotics.envs.adroit_hand import AdroitHandRelocateEnv
import gymnasium as gym
from gymnasium.envs.registration import register

class AdroitRelocateEnvV2(AdroitHandRelocateEnv):
    def __init__(self, reward_type: str = "dense", **kwargs):
        super().__init__(reward_type, **kwargs)

        

if __name__=="__main__":
    register(id="AdroitRelocateEnvV2", entry_point="envs.adroit_reloc:AdroitRelocateEnvV2")