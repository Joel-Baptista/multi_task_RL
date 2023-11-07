from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
import torch as T
from typing import Any, Dict, List, Optional, Type, Union, Tuple
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch import nn


class CostumAC(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, 
                 action_space: spaces.Space, 
                 lr_schedule: Schedule, 
                 net_arch: List[int] | Dict[str, List[int]] | None = None,
                 activation_fn: type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True, 
                 use_sde: bool = False, 
                 log_std_init: float = 0, 
                 full_std: bool = True, 
                 use_expln: bool = False, 
                 squash_output: bool = False, 
                 features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor, 
                 features_extractor_kwargs: Dict[str, Any] | None = None, 
                 share_features_extractor: bool = True, 
                 normalize_images: bool = True, 
                 optimizer_class: type[T.optim.Optimizer] = th.optim.Adam, 
                 optimizer_kwargs: Dict[str, Any] | None = None):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init, full_std, use_expln, squash_output, features_extractor_class, features_extractor_kwargs, share_features_extractor, normalize_images, optimizer_class, optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

