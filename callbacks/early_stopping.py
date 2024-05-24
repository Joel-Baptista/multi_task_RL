from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv


class EarlyStopping(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: (int)
    """
    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.parent = self
    ...
    
    def _on_step(self):
        return not bool(self.model.max_grad > 5_000)
