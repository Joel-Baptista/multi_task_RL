from gymnasium import Env, ObservationWrapper, spaces
from typing import TypeVar, Any, SupportsFloat
import math
import numpy as np


ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
PI = math.pi

class OBS_NORM_WRAP(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env
        dim = 0

        self.limits = np.array(
        [
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-PI, PI], # Changed from [-0.4, 0.25]
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-PI, PI],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ]
        )
 
        # self.limits = np.array(
        #     [
        #         [-0.3, 0.5],
        #         [-0.3, 0.5],
        #         [-0.3, 0.5],
        #         [-0.4, 0.25], # Changed from [-0.4, 0.25]
        #         [-0.3, 0.3], # Changed from [-0.3, 0.3]
        #         [-1, 2],
        #         [-0.524, 0.175],
        #         [-0.79, 0.61],
        #         [-0.44, 0.44],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [-0.44, 0.44],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [-0.44, 0.44],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [0, 0.7],
        #         [-0.44, 0.44],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [0, 1.6],
        #         [-1.047, 1.047],
        #         [0, 1.3],
        #         [-0.26, 0.26],
        #         [-0.52, 0.52],
        #         [-1.571, 0],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #     ]
        # )
        # [[-0.30698418  0.5       ]
        #     [-0.3         0.5       ]
        #     [-0.30268945  0.55496873]
        #     [-0.52721688  1.00690281]
        #     [-0.87063929  0.95427287]
        #     [-1.03421871  2.        ]
        #     [-0.75687836  0.42205087]
        #     [-1.00011118  0.76430317]
        #     [-0.78071292  0.90500734]
        #     [-0.11827366  1.94849483]
        #     [-0.21207896  1.6       ]
        #     [-0.33866352  1.68828432]
        #     [-0.84689303  0.8037845 ]
        #     [-0.00703212  1.93349892]
        #     [-0.14528901  1.6       ]
        #     [-0.30374388  1.63093191]
        #     [-0.85748579  0.92305251]
        #     [-0.16231897  1.9852598 ]
        #     [-0.42705748  1.61517722]
        #     [-0.45824924  1.66387598]
        #     [-0.33109569  1.39703239]
        #     [-0.88567437  0.78999421]
        #     [-0.16595245  1.89649813]
        #     [-0.30622255  1.69634304]
        #     [-0.20184769  1.6587479 ]
        #     [-1.13453583  1.11731293]
        #     [-0.12149049  1.83539788]
        #     [-0.40217681  0.44746593]
        #     [-0.82153388  1.09116204]
        #     [-1.83662568  0.28477805]
        #     [-0.86029937  0.52862352]
        #     [-1.12377788  0.5       ]
        #     [-0.5         0.5       ]
        #     [-0.59505772  0.51091807]
        #     [-0.88674417  0.5       ]
        #     [-0.5         0.5       ]
        #     [-0.5         0.65667861]
        #     [-0.5         1.19143332]
        #     [-0.5         0.5       ]]

        # self.observation_space = self.observation_space["observation"]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        

        observation, reward, terminated, truncated, info = self.env.step(action)

        off_limits = False
        monitored_obs = observation[30:39]

        for i in range(0, len(monitored_obs)):
            if abs(observation[i]) > 1: 
                off_limits = True
                break 

        if off_limits:
            terminated = True
            truncated = True 
            reward = -10


        return self.observation(observation), reward, terminated, truncated, info


    def observation(self, observation: Any) -> Any:

        for i in range(0, len(observation)):

            # if not (self.limits[i][0] <= observation[i] <= self.limits[i][1]):
            #     print(f"Value {observation[i]} out of bounds {self.limits[i]}")

            #     if observation[i] > self.limits[i][1]:
            #         self.limits[i][1] = observation[i]
            #     else:
            #         self.limits[i][0] = observation[i]

                # print(self.limits)

            observation[i] = 2 * ((observation[i] - self.limits[i][0]) / (self.limits[i][1] - self.limits[i][0])) - 1

            if observation[i] < -1: observation[i] = -1
            if observation[i] > 1: observation[i] = 1
            
        # print(observation)
        return observation


