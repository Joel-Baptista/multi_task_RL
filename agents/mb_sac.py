from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise

# JB's
from utils.common import class_from_str
import math
from influence_estimation.kl_torch import kl_div
import time
import os

SelfSAC = TypeVar("SelfSAC", bound="SAC")
_LOG_2PI = math.log(2 * math.pi)
MODELS = os.getenv("PHD_MODELS")


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        K: int = 16,
        world_steps_to_train: int= 1_000, #-1 to not train
        world_num_updates: int = 10,
        world_batch_size: int = 512,
        cai_clip: int = None,
        lambda_cai: int = 1,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        world_model: dict = None,
        model_path = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.K = K
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.world_steps_to_train = world_steps_to_train 
        self.world_num_updates = world_num_updates
        self.world_batch_size = world_batch_size
        self.model_path = model_path
        self.lambda_cai = lambda_cai
        self.cai_clip = cai_clip
        

        if "partial_obs" in world_model.keys():
            self.partial_obs = world_model["partial_obs"]
        else:
            self.partial_obs = None

        if _init_setup_model:
            self._setup_model()

        # JB's
        # aux = class_from_str(f"{world_model['module']}.{world_model['name']}", world_model['name'].upper())
        # print(aux)
        print(env.action_space.shape[0])
        print(env.observation_space.shape[0])

        if self.partial_obs is None:
            obs_shape = env.observation_space.shape[0]
        else:
            obs_shape = self.partial_obs[1] - self.partial_obs[0]

        world_model['args']['device'] = self.device
        self.world_model = class_from_str(f"{world_model['module']}.{world_model['name']}", world_model['name'].upper())(
            inp_dim = env.action_space.shape[0] + obs_shape, 
            outp_dim = obs_shape, 
            **world_model['args']
            )
        # print(f"For Entropy: {float(-np.prod(self.env.action_space.shape).astype(np.float32))}")
        if world_model["pretrained"]:
            self.world_model.load_state_dict(
                th.load(f"{MODELS}/{world_model['path']}/world_model.pt",
                                                      map_location=self.device))

        print(self.device)
        self.world_model.to(self.device)
        print(self.world_model)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        
        # st = time.time()

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, world_model_losses = [], [], []
        critic_eval = []

        sample_time, world_update_time, cai_time, target_time, ac_update_time, train_time =  [], [], [], [], [], []

        # print(f"Setup time: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            st = time.time()
            st_train = time.time()
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            sample_time.append(time.time() - st)

            if (self._n_updates + gradient_step) % self.world_steps_to_train == 0 \
                and self.world_steps_to_train > 0:
                # print("Training world model!!!!")
                self.world_model.train()
                for _ in range(0, self.world_num_updates):
                    st = time.time()
                    world_data = self.replay_buffer.sample(self.world_batch_size, env=self._vec_normalize_env)


                    if self.partial_obs is None:
                        obs = world_data.observations
                        obs_next = world_data.next_observations
                    else:
                        obs = world_data.observations[:,self.partial_obs[0]:self.partial_obs[1]]
                        obs_next = world_data.next_observations[:,self.partial_obs[0]:self.partial_obs[1]]

                    model_inp = th.concat((obs, world_data.actions), dim=1).type(th.float32)

                    mu_next, log_std_next = self.world_model(model_inp)

                    model_loss = self.world_model.calc_loss(mu_next, log_std_next, obs_next.type(th.float32))

                    self.world_model.optim.zero_grad()
                    model_loss.backward()
                    self.world_model.optim.step()

                    world_model_losses.append(model_loss.item())
                    world_update_time.append(time.time() - st)
                
                # print(f"Train World Model time: {time.time() - st}")
                # times.append(time.time() - st)
                # st = time.time()
                
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # print(f"Sample actions time: {time.time() - st}")
            # times.append(time.time() - st)
            # st = time.time()

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # print(f"Backprop Entropy time: {time.time() - st}")
            # times.append(time.time() - st)
            # st = time.time()

            st = time.time()
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                critic_eval.append(next_q_values.mean().item())
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term

                # print(f"Calc Q target time: {time.time() - st}")
                # times.append(time.time() - st)
                # st = time.time()

                if self.partial_obs is None:
                    obs = replay_data.observations
                else:
                    obs = replay_data.observations[:,self.partial_obs[0]:self.partial_obs[1]]

                st_cai = time.time()
                cai, info = self.calc_causal_influence(obs)
                cai = cai.unsqueeze(1)
                cai_time.append(time.time() - st)
                # cai2 = self.calc_causal_influence_2(replay_data.observations).unsqueeze(1)
                # cai =  cai * 10 ** (-6)
                # print(f"cai: {cai.mean()}")
                # print(f"cai2: {cai2.mean()}")
                if self.cai_clip is not None:
                    cai = th.clip(cai, 0, self.cai_clip)

                cai_time.append(time.time() - st_cai)
                rewards = replay_data.rewards + self.lambda_cai * cai
                
                # print(f"Calc CAI time: {time.time() - st}")
                # cai_time = time.time() - st
                # times.append(time.time() - st)
                # st = time.time()

                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            target_time.append(time.time() - st)
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            st = time.time()
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # print(f"Optimize Critic time: {time.time() - st}")
            # times.append(time.time() - st)
            # st = time.time()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            ac_update_time.append(time.time() - st)
            # print(f"Optimize Actor time: {time.time() - st}")
            # times.append(time.time() - st)
            # st = time.time()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                # print(f"Polyak Update time: {time.time() - st}")
                # times.append(time.time() - st)
                # st = time.time()

            train_time.append(time.time() - st_train)

        self._n_updates += gradient_steps

        st_log = time.time()
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/critic_eval", np.mean(critic_eval))
        self.logger.record("cai/cai", np.mean(
            cai.squeeze(1).detach().cpu().numpy()
            ))
        self.logger.record("cai/cai_and_reward", np.mean(
            rewards.squeeze(1).detach().cpu().numpy()
            ))
        self.logger.record("cai/cai_min", np.mean(
            cai.squeeze(1).min().detach().cpu().numpy()
            ))
        self.logger.record("cai/cai_max", np.mean(
            cai.squeeze(1).max().detach().cpu().numpy()
            ))
        for log in info:
            self.logger.record(f"cai/{log}", np.mean(info[log]))
            
        if len(world_model_losses) > 0:
            self.logger.record("world_model/world_model_loss", np.mean(world_model_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        self.logger.record("time/sample_time", 1 / np.mean(sample_time))
        self.logger.record("time/world_update_time", 1 / np.mean(world_update_time))
        self.logger.record("time/cai_time", 1 / np.mean(cai_time))
        self.logger.record("time/ac_update_time", 1 / np.mean(ac_update_time))
        self.logger.record("time/train_time", 1 / np.mean(train_time))
        
        self.logger.record("time/log_time", 1 / np.mean(time.time() - st_log))

        # print(f"Log time: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()
        
        # print(f"Total time {np.sum(times)}")
        # print(f"CAI time percent {cai_time / np.sum(times) * 100}")
        # print("----------------------------------------------------------------------------")
        # print("----------------------------------------------------------------------------")
        # print("----------------------------------------------------------------------------")


    @th.no_grad()
    def calc_causal_influence(self, states: th.Tensor):
        # print("---------------------------------------------------------------------")
        # times = []
        # st = time.time()

        self.world_model.eval()

        # Get transition distribution with action        
        # mu_mean, log_std, _ = self.actor.get_action_dist_params(states)

        # print(f"Get action dist: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        # print(f"mu_mean: {mu_mean.shape}")

        actions = th.rand(self.batch_size, self.K, *self.action_space.shape).to(self.device) * 2 - 1
        # outer_actions = th.zeros((self.K, self.batch_size, *self.action_space.shape)).to(self.device)
        # outer_states = th.zeros((self.K, self.batch_size, *self.observation_space.shape)).to(self.device)

        # for j in range(0, self.K):
        #     actions = self.actor.action_dist.actions_from_params(mu_mean, log_std, deterministic=False, **{})
        #     outer_actions[j] = actions
        #     outer_states[j] = states

        # print(f"Sample outer actions and states: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()
        
        states = (states.unsqueeze(1).repeat(1, self.K, 1).view(-1, states.shape[-1]))
        actions = actions.view(self.K * self.batch_size, *self.action_space.shape)

        # print(f"Reshape outer states and actions: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        model_inp = th.concat((states, actions), dim=1).type(th.float32)
        outer_mu_next, outer_logvar_next = self.world_model(model_inp)

        outer_mu_next = outer_mu_next.view(self.batch_size, self.K, -1) 
        outer_logvar_next = outer_logvar_next.view(self.batch_size, self.K, -1) 

        # print(f"Get outer next states: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        # Get transition distribution with action averaged out
        
        inner_actions = th.rand(self.batch_size, self.K, *self.action_space.shape).to(self.device) * 2 - 1
        inner_actions = inner_actions.view(self.K * self.batch_size, *self.action_space.shape)

        # inner_actions = th.zeros((self.K, self.batch_size, *self.action_space.shape)).to(self.device)
        # for i in range(0, self.K):
        #     actions = self.actor.action_dist.actions_from_params(mu_mean, log_std, deterministic=False, **{})
        #     inner_actions[i] = actions

        # outer_states = outer_states.view(K * self.batch_size, *self.observation_space.shape)
        # inner_actions = inner_actions.view(self.K * self.batch_size, *self.action_space.shape)

        # print(f"Sample inner actions: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        model_inp = th.concat((states, inner_actions), dim=1).type(th.float32)

        inner_mu_next, inner_logvar_next = self.world_model(model_inp)

        inner_mu_next = inner_mu_next.view(self.batch_size, self.K, -1).mean(dim=1)
        inner_logvar_next = inner_logvar_next.view(self.batch_size, self.K, -1).mean(dim=1) 

        # inner_mu_next = inner_mu_next.view(self.K, self.batch_size, *self.observation_space.shape).mean(dim=0)
        # inner_logvar_next = inner_logvar_next.view(self.K, self.batch_size, *self.observation_space.shape).mean(dim=0)

        # print(f"Sample inner next states: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        # print(inner_log_std_next[:, 0])
        # print(outer_log_std_next[:, 0])

        # print(f"inner_mu_next: {inner_mu_next.shape}")
        # print(f"outer_mu_next: {outer_mu_next.shape}")
        # print(f"inner_std_next: {inner_log_std_next.shape}")
        # print(f"outer_next: {outer_log_std_next.shape}")
        kls = kl_div(outer_mu_next, outer_logvar_next.exp(), inner_mu_next[:, None], inner_logvar_next.exp()[:, None])
        kls = th.clip(kls, min=0)
        # kls = kls.view(self.K, self.batch_size)
        # print(f"kls: {kls.shape}")

        cai = th.mean(kls, dim=1)

        # print(f"Calc CAI: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        # print(cai.shape)
        # all_kls[:, j] = kls

        # mu_next = mu_next.view(self.batch_size, K, -1).mean(dim=1)
        # log_std_next = log_std_next.view(self.batch_size, K, -1).mean(dim=1)

        # all_log_std_next[i] = log_std_next.mean()
        # all_log_std_next_ac[i] = log_std_next_ac.mean()
        # all_mu_next_ac[i] = mu_next_ac.mean()
        # all_mu_next[i] = mu_next.mean()
        # print(cai)

        # self.logger.record("train/outer_mu_next", outer_mu_next.mean().mean().detach().cpu().numpy())
        # self.logger.record("train/inner_mu_next", inner_mu_next.mean().mean().detach().cpu().numpy())
        # self.logger.record("train/outer_log_std_next", outer_logvar_next.mean().mean().detach().cpu().numpy())
        # self.logger.record("train/inner_log_std_next", inner_logvar_next.mean().mean().detach().cpu().numpy())
        # self.logger.record("train/cai_min", cai.min().detach().cpu().numpy())        
        # self.logger.record("train/cai_max", cai.max().detach().cpu().numpy())        

        info = {}
        # info = { 
        #     "outer_mu_next": outer_mu_next.mean().mean().detach().cpu().numpy(),
        #     "inner_mu_next": inner_mu_next.mean().mean().detach().cpu().numpy(),
        #     "outer_log_std_next": outer_logvar_next.mean().mean().detach().cpu().numpy(),
        #     "inner_log_std_next": inner_logvar_next.mean().mean().detach().cpu().numpy(),
        # }

        # print(f"Log info: {time.time() - st}")
        # times.append(time.time() - st)
        # st = time.time()

        # print(f"Total time {np.sum(times)}")
        # # print(f"CAI time percent {cai_time / np.sum(times) * 100}")

        # print("-----------------------------------------------------")
        self.world_model.train()
        return cai, info

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    def save(self, local_path: str = "") -> None:
        path = self.model_path

        if "best_model" in local_path:
            print(path)
            path = f"{self.model_path}/best_model"

        if self.model_path is None:
            print(f"Model path is {self.model_path}. Saving was not performed")
        else:
            print("-------------SAVING MODELS-------------------------")
            th.save(self.world_model.state_dict(), f"{path}/world_model.pt")
            th.save(self.actor.state_dict(), f"{path}/actor.pt")
            th.save(self.critic.state_dict(), f"{path}/critic.pt")
            th.save(self.critic_target.state_dict(), f"{path}/critic_target.pt")

    def load(self, local_path: str = ""):

        path = self.model_path

        if "best_model" in local_path:
            path = f"{self.model_path}/best_model"

        print("-------------LOADING MODELS-------------------------")
        self.world_model.load_state_dict(th.load(f"{path}/world_model.pt", map_location=self.device))
        self.actor.load_state_dict(th.load(f"{path}/actor.pt", map_location=self.device))
        self.critic.load_state_dict(th.load(f"{path}/critic.pt", map_location=self.device))
        self.critic_target.load_state_dict(th.load(f"{path}/critic_target.pt", map_location=self.device))

        return self


    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
