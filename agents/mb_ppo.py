import numpy as np
import torch as th
from typing import Any, Dict

from gymnasium import spaces

from stable_baselines3.common.buffers import  RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch._C import device

# JB's
from utils.common import class_from_str
import math
from influence_estimation.kl_torch import kl_div
import time
import os

_LOG_2PI = math.log(2 * math.pi)
MODELS = os.getenv("PHD_MODELS")

class MB_PPO(PPO):
    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: float | Schedule | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: Dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: device | str = "auto",
        _init_setup_model: bool = True,
        world_model: dict = None,
        K: int = 16,
        world_steps_to_train: int = -1,
        world_num_updates: int = 10,
        world_batch_size: int = 512,

    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        # JB's
        self.K = K
        self.world_num_updates = world_num_updates
        self.world_steps_to_train = world_steps_to_train
        self.world_batch_size = world_batch_size

        #TODO Find clever way to load world model when loading the algorithm
        if isinstance(world_model, dict):
            self.world_model = class_from_str(f"{world_model['module']}.{world_model['name']}", world_model['name'].upper())(
                inp_dim = env.action_space.shape[0] + env.observation_space.shape[0], 
                outp_dim = env.observation_space.shape[0], 
                **world_model['args']
                )
            # print(f"For Entropy: {float(-np.prod(self.env.action_space.shape).astype(np.float32))}")
            if world_model["pretrained"]:
                self.world_model.load_state_dict(
                    th.load(f"{MODELS}/{world_model['path']}/world_model.pt",
                                                        map_location=self.device))

            self.world_model.to(self.device)
            print(self.world_model)

    def train(self) -> None:
        # Train model before PPO
        world_model_losses = []
        self.world_model.train()

        observations = self.rollout_buffer.observations.squeeze(1) 
        actions = self.rollout_buffer.actions.squeeze(1) 

        state_actions = np.zeros((observations.shape[0] - 1, 
                                  observations.shape[1] + actions.shape[1]))

        state_next = np.zeros((observations.shape[0] - 1,
                              observations.shape[1]))

        for i in range(0, observations.shape[0] - 1):
            action = actions[i]
            observation = observations[i]

            state_actions = np.concatenate((observation, action))
            state_next[i] = self.rollout_buffer.observations[i+1]

        for _ in range(0, self.world_num_updates):

            for batch_idx in range(0, state_actions.shape[0], self.world_batch_size): 
                
                end_idx = batch_idx + self.world_batch_size
                if end_idx > state_actions.shape[0]: end_idx = state_actions.shape[0]

                model_inp = obs_as_tensor(state_actions, device=self.device)[batch_idx: end_idx]

                mu_next, log_std_next = self.world_model(model_inp)

                model_loss = self.world_model.calc_loss(mu_next, log_std_next, obs_as_tensor(state_next, device=self.device))

                self.world_model.optim.zero_grad()
                model_loss.backward()
                self.world_model.optim.step()

                world_model_losses.append(model_loss.item())

        self.logger.record("world_model/world_model_loss", np.mean(world_model_losses))

        # Normal PPO
        super().train()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
    
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # JB's
        
        cai, info = self.calc_causal_influence(obs_as_tensor(rollout_buffer.observations, self.device))
        cai = cai.unsqueeze(1)
        #TODO CLip CAI in 100ish
        rollout_buffer.rewards = rollout_buffer.rewards + cai.detach().cpu().numpy()
        
        # log cai metrics
        self.logger.record("cai/cai", np.mean(
            cai.squeeze(1).detach().cpu().numpy()
            ))
        self.logger.record("cai/cai_min", np.mean(
            cai.squeeze(1).min().detach().cpu().numpy()
            ))
        self.logger.record("cai/cai_max", np.mean(
            cai.squeeze(1).max().detach().cpu().numpy()
            ))
        for log in info:
            self.logger.record(f"cai/{log}", np.mean(info[log]))

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    @th.no_grad()
    def calc_causal_influence(self, states: th.Tensor):
        self.world_model.eval()
        states = states.squeeze(1)
        # Get transition distribution with action        
        # mu_mean, _, log_probs = self.policy(states)
        pi_features = self.policy.extract_features(states)
        latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        dist = self.policy._get_action_dist_from_latent(latent_pi)

        outer_actions = th.zeros((self.K, states.shape[0], *self.action_space.shape)).to(self.device)
        outer_states = th.zeros((self.K, states.shape[0], *self.observation_space.shape)).to(self.device)

        for j in range(0, self.K):
            actions = dist.get_actions(deterministic=False).to(self.device)
            outer_actions[j] = actions
            outer_states[j] = states
        
        outer_states = outer_states.view(self.K * states.shape[0], *self.observation_space.shape)
        outer_actions = outer_actions.view(self.K * states.shape[0], *self.action_space.shape)

        model_inp = th.concat((outer_states, outer_actions), dim=1).type(th.float32)
        outer_mu_next, outer_logvar_next = self.world_model(model_inp)

        # Get transition distribution with action averaged out
        inner_actions = th.zeros((self.K, states.shape[0], *self.action_space.shape)).to(self.device)
        for i in range(0, self.K):
            actions = dist.get_actions(deterministic=False)
            inner_actions[i] = actions

        inner_actions = inner_actions.view(self.K * states.shape[0], *self.action_space.shape)

        model_inp = th.concat((states.repeat((self.K, 1)), inner_actions), dim=1).type(th.float32)

        inner_mu_next, inner_logvar_next = self.world_model(model_inp)

        inner_mu_next = inner_mu_next.view(self.K, states.shape[0], *self.observation_space.shape).mean(dim=0).repeat((self.K, 1))
        inner_logvar_next = inner_logvar_next.view(self.K, states.shape[0], *self.observation_space.shape).mean(dim=0).repeat((self.K, 1))

        kls = kl_div(outer_mu_next, outer_logvar_next.exp(), inner_mu_next, inner_logvar_next.exp())
        kls = th.clip(kls, min=0)
        kls = kls.view(self.K, states.shape[0])

        cai = th.mean(kls, dim=0)      

        info = { 
            "outer_mu_next": outer_mu_next.mean().mean().detach().cpu().numpy(),
            "inner_mu_next": inner_mu_next.mean().mean().detach().cpu().numpy(),
            "outer_log_std_next": outer_logvar_next.mean().mean().detach().cpu().numpy(),
            "inner_log_std_next": inner_logvar_next.mean().mean().detach().cpu().numpy(),
        }   

        self.world_model.train()
        return cai, info