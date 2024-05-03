from gymnasium_robotics.envs.adroit_hand import AdroitHandRelocateEnv


class AdroitRelocateEnvV2(AdroitHandRelocateEnv):
    def __init__(self, reward_type: str = "dense", **kwargs):
        super().__init__(reward_type, **kwargs)

        