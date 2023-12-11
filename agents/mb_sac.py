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
from utils.common import DotDict, model_class_from_str, class_from_str
import math
from influence_estimation.kl_torch import kl_div

SelfSAC = TypeVar("SelfSAC", bound="SAC")
_LOG_2PI = math.log(2 * math.pi)



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
        world_model: str = None,
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
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

        # JB's
        self.world_model = class_from_str(f"models.world_model.{world_model}", world_model.upper())(
            inp_dim = env.action_space.shape[0] + env.observation_space.shape[0], 
            outp_dim = env.observation_space.shape[0], 
            hidden_dims = [4156, 2048, 512, 206]
            )
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

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            model_inp = th.concat((replay_data.observations, replay_data.actions), dim=1).type(th.float32)

            mu_next, sigma_next = self.world_model(model_inp)

            model_loss = 0.5 * (((replay_data.next_observations.type(th.float32) - mu_next) ** 2) / (sigma_next) + th.log(sigma_next) + _LOG_2PI)
            model_loss = model_loss.mean()

            self.world_model.optim.zero_grad()
            model_loss.backward()
            self.world_model.optim.step()

            world_model_losses.append(model_loss.item())


            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

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

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term

                cai = self.calc_causal_influence(replay_data.observations).unsqueeze(1)

                rewards = replay_data.rewards + cai
                
                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

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

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/world_model_loss", np.mean(world_model_losses))
        self.logger.record("train/cai", np.mean(
            cai.squeeze(1).detach().cpu().numpy()
            ))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    @th.no_grad()
    def calc_causal_influence(self, states: th.Tensor):
        cai = 0
        K = 2

        all_kls = th.zeros((self.batch_size, K)).to(self.device)

        # Get transition distribution with action        
        mu_mean, log_std, _ = self.actor.get_action_dist_params(states)

        for j in range(0, K):
            actions = self.actor.action_dist.actions_from_params(mu_mean, log_std, deterministic=False, **{})
            
            model_inp = th.concat((states, actions), dim=1).type(th.float32)
            mu_next_ac, sigma_next_ac = self.world_model(model_inp)

            # Get transition distribution with action averaged out
            all_actions = th.zeros((K, self.batch_size, *self.action_space.shape)).to(self.device)
            for i in range(0, K):
                actions = self.actor.action_dist.actions_from_params(mu_mean, log_std, deterministic=False, **{})
                
                all_actions[i] = actions.unsqueeze(0)

            model_inp = th.concat((states.repeat((K, 1)), all_actions.view(-1, *self.action_space.shape)), dim=1).type(th.float32)

            mu_next, sigma_next = self.world_model(model_inp)

            mu_next = mu_next.view(self.batch_size, K, -1).mean(dim=1)
            sigma_next = sigma_next.view(self.batch_size, K, -1).mean(dim=1)

            kls = kl_div(mu_next_ac, sigma_next_ac, mu_next, sigma_next)
            kls = th.clip(kls, min=0)
            
            all_kls[:, j] = kls

        cai = th.mean(all_kls, dim=1)

        return cai

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
    
    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     train_freq: TrainFreq,
    #     replay_buffer: ReplayBuffer,
    #     action_noise: Optional[ActionNoise] = None,
    #     learning_starts: int = 0,
    #     log_interval: Optional[int] = None,
    # ) -> RolloutReturn:
    #     """
    #     Collect experiences and store them into a ``ReplayBuffer``.

    #     :param env: The training environment
    #     :param callback: Callback that will be called at each step
    #         (and at the beginning and end of the rollout)
    #     :param train_freq: How much experience to collect
    #         by doing rollouts of current policy.
    #         Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
    #         or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
    #         with ``<n>`` being an integer greater than 0.
    #     :param action_noise: Action noise that will be used for exploration
    #         Required for deterministic policy (e.g. TD3). This can also be used
    #         in addition to the stochastic policy for SAC.
    #     :param learning_starts: Number of steps before learning for the warm-up phase.
    #     :param replay_buffer:
    #     :param log_interval: Log data every ``log_interval`` episodes
    #     :return:
    #     """
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     num_collected_steps, num_collected_episodes = 0, 0

    #     assert isinstance(env, VecEnv), "You must pass a VecEnv"
    #     assert train_freq.frequency > 0, "Should at least collect one step or episode."

    #     if env.num_envs > 1:
    #         assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

    #     # Vectorize action noise if needed
    #     if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
    #         action_noise = VectorizedActionNoise(action_noise, env.num_envs)

    #     if self.use_sde:
    #         self.actor.reset_noise(env.num_envs)

    #     callback.on_rollout_start()
    #     continue_training = True
    #     while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
    #         if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.actor.reset_noise(env.num_envs)

    #         # Select action randomly or according to policy
    #         actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

    #         # Rescale and perform action
    #         new_obs, rewards, dones, infos = env.step(actions)

    #         self.num_timesteps += env.num_envs
    #         num_collected_steps += 1

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         # Only stop training if return value is False, not when it is None.
    #         if callback.on_step() is False:
    #             return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

    #         # Retrieve reward and episode length if using Monitor wrapper
    #         self._update_info_buffer(infos, dones)

    #         # Store data in replay buffer (normalized action and unnormalized observation)
    #         self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

    #         self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

    #         # For DQN, check if the target network should be updated
    #         # and update the exploration schedule
    #         # For SAC/TD3, the update is dones as the same time as the gradient update
    #         # see https://github.com/hill-a/stable-baselines/issues/900
    #         self._on_step()

    #         for idx, done in enumerate(dones):
    #             if done:
    #                 # Update stats
    #                 num_collected_episodes += 1
    #                 self._episode_num += 1

    #                 if action_noise is not None:
    #                     kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
    #                     action_noise.reset(**kwargs)

    #                 # Log training infos
    #                 if log_interval is not None and self._episode_num % log_interval == 0:
    #                     self._dump_logs()
    #     callback.on_rollout_end()

    #     return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


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
