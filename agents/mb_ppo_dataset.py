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
import minari
from torch.utils.data import DataLoader
from utils.dataloader import collate_fn

_LOG_2PI = math.log(2 * math.pi)
MODELS = os.getenv("PHD_RESULTS")

class MB_PPO_DATASET(PPO):
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
        lambda_cai: float = 1.0,

    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        # JB's
        self.K = K
        self.world_num_updates = world_num_updates
        self.world_steps_to_train = world_steps_to_train
        self.world_batch_size = world_batch_size
        self.lambda_cai = lambda_cai
        
        if "partial_obs" in world_model.keys():
            self.partial_obs = world_model["partial_obs"]
        else:
            self.partial_obs = None

        #TODO Find clever way to load world model when loading the algorithm
        if isinstance(world_model, dict):
            self.world_model = class_from_str(f"{world_model['module']}.{world_model['name']}", world_model['name'].upper())(
                inp_dim = env.action_space.shape[0] + env.observation_space.shape[0], 
                outp_dim = env.observation_space.shape[0], 
                **world_model['args']
                )
            # print(f"For Entropy: {float(-np.prod(self.env.action_space.shape).astype(np.float32))}")
            print(f"{MODELS}/{world_model['path']}/world_model.pt")
            if world_model["pretrained"]:
                self.world_model.load_state_dict(
                    th.load(f"{MODELS}/{world_model['path']}/world_model.pt",
                                                        map_location=self.device))

            self.world_model.to(self.device)
            print(self.world_model)

        
        dataset_name = world_model.get("dataset_name", None)
            
        if dataset_name is not None:
            try:
                dataset = minari.load_dataset(dataset_name)
            except:
                dataset = minari.load_dataset(dataset_name, download=True)

        self.dataset = dataset

        self.dataset_loader = DataLoader(dataset, batch_size=4, num_workers=3,shuffle=True, collate_fn=collate_fn)
        

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

            state_actions[i] = np.concatenate((observation, action))
            state_next[i] = self.rollout_buffer.observations[i+1]

        for _ in range(0, self.world_num_updates):
            indices = np.random.choice(len(state_actions), int(self.world_batch_size / 2), replace=False)
            
            batch = self.dataset_loader.__iter__().__next__() #TODO for now the world batch size caps at 800. Fix to be adaptable
            
            indices_dataset = np.random.choice(len(batch[0]), self.world_batch_size - len(indices), replace=False)
            
            sampled_state_actions = state_actions[indices]
            sampled_state_next = state_next[indices]

            sampled_state_actions = np.concatenate((sampled_state_actions, batch[0][indices_dataset]))
            sampled_state_next = np.concatenate((sampled_state_next, batch[1][indices_dataset]))

            model_inp = obs_as_tensor(sampled_state_actions, device=self.device).type(th.float32)

            mu_next, log_std_next = self.world_model(model_inp)

            model_loss = self.world_model.calc_loss(mu_next, log_std_next, obs_as_tensor(sampled_state_next, device=self.device))

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
        
        cais_list = np.zeros((n_rollout_steps,))
        rewards_list = np.zeros((n_rollout_steps,))
        cais_and_rewards_list = np.zeros((n_rollout_steps,))

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

            obs = obs_as_tensor(self._last_obs, self.device)
            cais, _ = self.calc_causal_influence(obs)
            cais = cais.unsqueeze(1).cpu().numpy()

            rewards_list[n_steps] = rewards

            rewards += self.lambda_cai * cais.squeeze(0)

            cais_and_rewards_list[n_steps] = rewards

            cais_list[n_steps] = cais
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

        # log cai metrics
        self.logger.record("cai/cai", np.mean(
            cais_list
            ))
        self.logger.record("cai/rewards", np.mean(
            rewards_list
            ))
        self.logger.record("cai/cai_and_reward", np.mean(
            cais_and_rewards_list
            ))
        self.logger.record("cai/cai_min", np.min(
            cais_list
            ))
        self.logger.record("cai/cai_max", np.max(
            cais_list
            ))

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    @th.no_grad()
    def calc_causal_influence(self, states: th.Tensor):

        self.world_model.eval()
        batch_size = states.shape[0]

        # Get transition distribution with action        

        actions = th.rand(batch_size, self.K, *self.action_space.shape).to(self.device) * 2 - 1
        
        states = (states.unsqueeze(1).repeat(1, self.K, 1).view(-1, states.shape[-1]))
        actions = actions.view(self.K * batch_size, *self.action_space.shape)

        model_inp = th.concat((states, actions), dim=1).type(th.float32)
        outer_mu_next, outer_logvar_next = self.world_model(model_inp)

        outer_mu_next = outer_mu_next.view(batch_size, self.K, -1) 
        outer_logvar_next = outer_logvar_next.view(batch_size, self.K, -1) 
        # Get transition distribution with action averaged out
        
        inner_actions = th.rand(batch_size, self.K, *self.action_space.shape).to(self.device) * 2 - 1
        inner_actions = inner_actions.view(self.K * batch_size, *self.action_space.shape)

        model_inp = th.concat((states, inner_actions), dim=1).type(th.float32)

        inner_mu_next, inner_logvar_next = self.world_model(model_inp)

        inner_mu_next = inner_mu_next.view(batch_size, self.K, -1).mean(dim=1)
        inner_logvar_next = inner_logvar_next.view(batch_size, self.K, -1).mean(dim=1) 

        if self.partial_obs is not None:
            outer_mu_next = outer_mu_next[:, :,self.partial_obs[0]:self.partial_obs[1]]
            outer_logvar_next = outer_logvar_next[:, :,self.partial_obs[0]:self.partial_obs[1]]
            inner_mu_next = inner_mu_next[:, self.partial_obs[0]:self.partial_obs[1]]
            inner_logvar_next = inner_logvar_next[:, self.partial_obs[0]:self.partial_obs[1]]

        kls = kl_div(outer_mu_next, outer_logvar_next.exp(), inner_mu_next[:, None], inner_logvar_next.exp()[:, None])
        kls = th.clip(kls, min=0)

        cai = th.mean(kls, dim=1)    

        info = {}
        self.world_model.train()
        return cai, info