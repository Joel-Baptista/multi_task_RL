from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callbacks.video_recorder import VideoRecorder
from callbacks.early_stopping import EarlyStopping
from wandb.integration.sb3 import WandbCallback

from utils.common import class_from_str

def setup_callbacks(cfg, experiment_path, log_path, record_env, model):

    eval_callback = EvalCallback(model.env, best_model_save_path=f'{experiment_path}/best_model',
                        log_path=f'{log_path}', eval_freq=cfg.eval_freq, n_eval_episodes=100,
                        deterministic=True, render=False)
    wand_callback = WandbCallback(
            verbose=2,
            model_save_path=experiment_path,
            model_save_freq= int(cfg.total_timesteps / cfg.checkpoints),
            log = "all"
            )
    
#     early_stopping = EarlyStopping(
#             verbose=2
#             )
    
    video_callback = VideoRecorder(record_env, log_path=log_path, record_freq=cfg.record_freq)
    
    callbacks_list = [eval_callback, wand_callback, video_callback]
    # callbacks_list = [eval_callback, video_callback]

    if not (cfg.algorithm.callbacks is None):
        for callback_info in cfg.algorithm.callbacks:
            print(callback_info)
            callback_class = class_from_str(callback_info["module"], callback_info["class_type"])

            if callback_info["args"] is None: callback_info["args"] = {}

            new_callback = callback_class(**callback_info["args"])
            callbacks_list.append(new_callback)



    
    callbacks = CallbackList(callbacks_list)



    return callbacks