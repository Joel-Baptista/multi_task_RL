from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

import cv2 as cv


class VideoRecorder(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, 
                 record_env: Union[gym.Env, VecEnv, None],
                 log_path: str,
                 record_freq: int = 50, 
                 verbose=0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  
        # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  
        self.recording_env = record_env  
        self.record_freq = record_freq
        self.n_videos = 0
        self.log_path = log_path 
        # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        # self.recording_env.render_mode = "rgb_array"


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if (self.num_timesteps % self.record_freq) == 0 and self.record_freq > 0:

            obs, _ = self.recording_env.reset()
            total_reward = 0
            timesteps = 0
            self.n_videos += 1

            video = cv.VideoWriter(f"{self.log_path}/video_{self.n_videos}.mp4",
                                    cv.VideoWriter_fourcc(*"mp4v"),
                                    30,
                                    (480, 480))

            video_arrays = []
            for _ in range(0, 300):
                img = self.recording_env.render()
                
                video.write(cv.cvtColor(img, cv.COLOR_BGR2RGB))

                action, _ = self.model.predict(obs, deterministic=True)
                
                obs_new, reward, terminated, truncated, info = self.recording_env.step(action)

                total_reward += reward
                timesteps += 1
                obs = obs_new
                if terminated or truncated:
                    break
        
            video.release()
            print(f"Reward: {total_reward}, Ep. Lenght: {timesteps}")

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass