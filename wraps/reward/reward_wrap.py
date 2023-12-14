from gymnasium import Env, RewardWrapper
from typing import SupportsFloat
from typing import TypeVar
from typing import Any

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class REWARD_WRAP(RewardWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        print(observation)

        # delta_objective = abs(observation['achieved_goal']['microwave'][0] - observation['desired_goal']['microwave'][0])

        return observation, self.reward(reward), terminated, truncated, info


    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        # print(self.env.unwrapped.model)
        # print(self.env.unwrapped.data)

        # joint_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:panda0_joint5")
        # joint_type = self.env.unwrapped.model.jnt_type[joint_id]
        # joint_addr = self.env.unwrapped.model.jnt_qposadr[joint_id]
        
        # if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        #     ndim = 7
        # elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        #     ndim = 4
        # else:
        #     assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        #     ndim = 1

        # start_idx = joint_addr
        # end_idx = joint_addr + ndim

        # print(self.env.unwrapped.data.qpos[start_idx:end_idx].copy())

        return 10 * reward