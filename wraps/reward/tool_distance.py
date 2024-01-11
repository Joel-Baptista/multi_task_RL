from gymnasium import Env, RewardWrapper
from typing import SupportsFloat
from typing import TypeVar
from typing import Any

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class TOOL_DISTANCE(RewardWrapper):
    def __init__(self, env: Env, reward_scale: float = 1.0, tool_weight: float = 1.0):
        super().__init__(env)
        self.env = env
        self.reward_scale = reward_scale
        self.tool_weight = tool_weight

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        observation, reward, terminated, truncated, info = self.env.step(action)

        # delta_objective = abs(observation['achieved_goal']['microwave'][0] - observation['desired_goal']['microwave'][0])

        return observation, self.reward(reward, observation), terminated, truncated, info


    def reward(self, reward: SupportsFloat, observation: ObsType) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        print(f"Tool Dist: {sum(abs(observation['observation'][6:9]))}")
        print(f"Normal Reward: {reward}")
        print("----------------------------------------------------------")
        return self.reward_scale * reward - self.tool_weight * sum(abs(observation["observation"][6:9]))