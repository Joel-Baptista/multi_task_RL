# System
import argparse
import os
import sys

import wandb

import numpy as np
import torch as th
import gymnasium as gym

from stable_baselines3.common.buffers import ReplayBuffer

from utils.common import setup_experiment, class_from_str
from utils.env import add_wrappers

# SelfWORLD = TypeVar("SelfWORLD", bound="WORLD")
MODELS = os.getenv("PHD_MODELS")

# class WORLD(OffPolicyAlgorithm):
#     def __init__(
#         self,
#         policy: str | type[BasePolicy],
#         env: GymEnv | str,
#         learning_rate: float | Schedule,
#         buffer_size: int = 1000000,
#         learning_starts: int = 100,
#         batch_size: int = 256,
#         tau: float = 0.005,
#         gamma: float = 0.99,
#         train_freq: int | Tuple[int, str] = ...,
#         gradient_steps: int = 1,
#         action_noise: ActionNoise | None = None,
#         replay_buffer_class: type[ReplayBuffer] | None = None,
#         replay_buffer_kwargs: Dict[str, Any] | None = None,
#         optimize_memory_usage: bool = False,
#         policy_kwargs: Dict[str, Any] | None = None,
#         stats_window_size: int = 100,
#         tensorboard_log: str | None = None,
#         verbose: int = 0,
#         device: Union[th.device, str] = "auto",
#         support_multi_env: bool = False,
#         monitor_wrapper: bool = True,
#         seed: int | None = None,
#         use_sde: bool = False,
#         sde_sample_freq: int = -1,
#         use_sde_at_warmup: bool = False,
#         sde_support: bool = True,
#         supported_action_spaces: (spaces.Box,) = None,
#         world_model_kwargs = None,
#         _init_setup_model: bool = True,
#     ):
#         super().__init__(
#             policy,
#             env,
#             learning_rate,
#             buffer_size,
#             learning_starts,
#             batch_size,
#             tau,
#             gamma,
#             train_freq,
#             gradient_steps,
#             action_noise,
#             replay_buffer_class,
#             replay_buffer_kwargs,
#             optimize_memory_usage,
#             policy_kwargs,
#             stats_window_size,
#             tensorboard_log,
#             verbose,
#             device,
#             support_multi_env,
#             monitor_wrapper,
#             seed,
#             use_sde,
#             sde_sample_freq,
#             use_sde_at_warmup,
#             sde_support,
#             supported_action_spaces,
#         )

#         if _init_setup_model:
#             self._setup_model()

#         print(world_model_kwargs)
#         self.world_model = class_from_str(f"{world_model_kwargs['module']}.{world_model_kwargs['name']}", world_model_kwargs['name'].upper())(
#             inp_dim = env.action_space.shape[0] + env.observation_space.shape[0], 
#             outp_dim = env.observation_space.shape[0], 
#             **world_model_kwargs['args']
#             )

#         self.world_model.to(self.device)
#         print(self.world_model)


#     def collect_rollouts(
#         self,
#         env: VecEnv,
#         callback: BaseCallback,
#         train_freq: TrainFreq,
#         replay_buffer: ReplayBuffer,
#         action_noise: Optional[ActionNoise] = None,
#         learning_starts: int = 0,
#         log_interval: Optional[int] = None,
#     ) -> RolloutReturn:
#         """
#         Collect experiences and store them into a ``ReplayBuffer``.

#         :param env: The training environment
#         :param callback: Callback that will be called at each step
#             (and at the beginning and end of the rollout)
#         :param train_freq: How much experience to collect
#             by doing rollouts of current policy.
#             Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
#             or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
#             with ``<n>`` being an integer greater than 0.
#         :param action_noise: Action noise that will be used for exploration
#             Required for deterministic policy (e.g. TD3). This can also be used
#             in addition to the stochastic policy for SAC.
#         :param learning_starts: Number of steps before learning for the warm-up phase.
#         :param replay_buffer:
#         :param log_interval: Log data every ``log_interval`` episodes
#         :return:
#         """
#         # Switch to eval mode (this affects batch norm / dropout)
#         # self.policy.set_training_mode(False)

#         num_collected_steps, num_collected_episodes = 0, 0

#         assert isinstance(env, VecEnv), "You must pass a VecEnv"
#         assert train_freq.frequency > 0, "Should at least collect one step or episode."

#         if env.num_envs > 1:
#             assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

#         # Vectorize action noise if needed
#         if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
#             action_noise = VectorizedActionNoise(action_noise, env.num_envs)

#         callback.on_rollout_start()
#         continue_training = True
#         while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

#             # Select action randomly or according to policy
#             actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
#             buffer_actions = actions

#             # Rescale and perform action
#             new_obs, rewards, dones, infos = env.step(actions)

#             self.num_timesteps += env.num_envs
#             num_collected_steps += 1

#             # Give access to local variables
#             callback.update_locals(locals())
#             # Only stop training if return value is False, not when it is None.
#             if callback.on_step() is False:
#                 return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

#             # Retrieve reward and episode length if using Monitor wrapper
#             self._update_info_buffer(infos, dones)

#             # Store data in replay buffer (normalized action and unnormalized observation)
#             self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

#             self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

#             # For DQN, check if the target network should be updated
#             # and update the exploration schedule
#             # For SAC/TD3, the update is dones as the same time as the gradient update
#             # see https://github.com/hill-a/stable-baselines/issues/900
#             self._on_step()

#             for idx, done in enumerate(dones):
#                 if done:
#                     # Update stats
#                     num_collected_episodes += 1
#                     self._episode_num += 1

#                     if action_noise is not None:
#                         kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
#                         action_noise.reset(**kwargs)

#                     # Log training infos
#                     if log_interval is not None and self._episode_num % log_interval == 0:
#                         self._dump_logs()
#         callback.on_rollout_end()

#         return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    # def learn(
    #     self: SelfWORLD,
    #     total_timesteps: int,
    #     callback: MaybeCallback = None,
    #     log_interval: int = 4,
    #     tb_log_name: str = "run",
    #     reset_num_timesteps: bool = True,
    #     progress_bar: bool = False,
    # ) -> SelfWORLD:
    #     total_timesteps, callback = self._setup_learn(
    #         total_timesteps,
    #         callback,
    #         reset_num_timesteps,
    #         tb_log_name,
    #         progress_bar,
    #     )

    #     callback.on_training_start(locals(), globals())

    #     while self.num_timesteps < total_timesteps:
    #         rollout = self.collect_rollouts(
    #             self.env,
    #             train_freq=self.train_freq,
    #             action_noise=self.action_noise,
    #             callback=callback,
    #             learning_starts=self.learning_starts,
    #             replay_buffer=self.replay_buffer,
    #             log_interval=log_interval,
    #         )

    #         if rollout.continue_training is False:
    #             break

    #         if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
    #             # If no `gradient_steps` is specified,
    #             # do as many gradients steps as steps performed during the rollout
    #             gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
    #             # Special case when the user passes `gradient_steps=0`
    #             if gradient_steps > 0:
    #                 self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

    #     callback.on_training_end()

    #     return self

    # def train(self, gradient_steps: int, batch_size: int) -> None:
    #     """
    #     Sample the replay buffer and do the updates
    #     (gradient descent and update target networks)
    #     """
    #     self.world_model.train()
        
    #     world_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)

    #     model_inp = th.concat((world_data.observations, world_data.actions), dim=1).type(th.float32)

    #     mu_next, log_std_next = self.world_model(model_inp)

    #     model_loss = self.world_model.calc_loss(mu_next, log_std_next, world_data.next_observations.type(th.float32))

    #     self.world_model.optim.zero_grad()
    #     model_loss.backward()
    #     self.world_model.optim.step()

    #     # world_model_losses.append(model_loss.item())
    #     self.logger.record("world_model/world_model_loss", np.mean(model_loss.item()))
                
    #     # print(f"Train World Model time: {time.time() - st}")

# def save(world_model: nn.Modules, local_path: str = "") -> None:

#     if "best_model" in local_path:
#         path = local_path

#     if model_path is None:
#         print(f"Model path is {self.model_path}. Saving was not performed")
#     else:
#         print("-------------SAVING MODELS-------------------------")
#         th.save(world_model.state_dict(), f"{path}/world_model.pt")



def main():
    parser = argparse.ArgumentParser(description='Train asl2text models.')
    parser.add_argument('-en', '--experiment_name', type=str)
    parser.add_argument('-ow', '--overwrite', action='store_true')
    parser.add_argument('-id', '--identifier', type=str, default='')
    parser.add_argument('-d', '--debug', action='store_true')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))
    
    experiment_name, experiment_path, log_path, cfg = setup_experiment(args)

    print(cfg.algorithm.args)
    device = th.device(cfg.algorithm.args.device if th.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    env = gym.make(cfg.env.name,**cfg.env.args)
    record_env = gym.make(cfg.env.name, render_mode="rgb_array",**cfg.env.args)
    
    env = add_wrappers(env, cfg.env.wraps)
    record_env = add_wrappers(record_env, cfg.env.wraps)

    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    print(env.observation_space.shape)
    print(env.reward_range)

    if "model_path" in cfg['algorithm']['args'].keys():
        cfg["algorithm"]["args"]["model_path"] = experiment_path
    
    algorithm_args = dict(cfg.algorithm.args)
    world_model_kwargs = algorithm_args["world_model_kwargs"]

    print(world_model_kwargs)
    world_model = class_from_str(f"{world_model_kwargs['module']}.{world_model_kwargs['name']}", world_model_kwargs['name'].upper())(
        inp_dim = env.action_space.shape[0] + env.observation_space.shape[0], 
        outp_dim = env.observation_space.shape[0], 
        **world_model_kwargs['args']
        )

    world_model.to(device)
    print(world_model)
    replay_buffer = ReplayBuffer(algorithm_args["buffer_size"], env.observation_space, env.action_space, device=device)

    if not args['debug']:
        run = wandb.init(
            project=cfg.project, 
            sync_tensorboard=True,
            config=cfg,
            dir=log_path,
            name=f"{cfg.algorithm.name}_{experiment_name}{args['identifier']}"
            )

    min_loss = 1e6

    for i in range(0, cfg.total_timesteps):

        action = env.action_space.sample()
        next_obs, reward, done, truncated, infos = env.step(action)    

        if done or truncated: done = True

        replay_buffer.add(
                obs,  # type: ignore[arg-type]
                next_obs,  # type: ignore[arg-type]
                action,
                reward,
                done,
                [infos],
            )
                
        if done:
            next_obs, infos = env.reset()

        obs = next_obs

        if i >= algorithm_args["learning_starts"]:

            replay_data = replay_buffer.sample(algorithm_args["batch_size"], env=None)

            model_inp = th.concat((replay_data.observations, replay_data.actions), dim=1).type(th.float32)

            mu_next, log_std_next = world_model(model_inp)

            model_loss = world_model.calc_loss(mu_next, log_std_next, replay_data.next_observations.type(th.float32))

            world_model.optim.zero_grad()
            model_loss.backward()
            # if self.grad_norm_clipping is not None:
            #     world_grad_norm = th.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_norm_clipping)
            #     world_grad_norms.append(world_grad_norm.mean().item())
            world_model.optim.step()

            wandb.log({"world_model/world_model_loss": model_loss.item()})

            if min_loss > model_loss.item():
                min_loss = model_loss.item()
                print("Saving model at iteration: ", i) 
                th.save(world_model.state_dict(), f"{experiment_path}/world_model.pt")
            

if __name__ == "__main__":
    main()
