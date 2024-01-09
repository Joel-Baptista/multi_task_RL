# System
import importlib
import os
import sys
import shutil
import copy
import yaml
from colorama import Fore

class DotDict(dict):
    """
    Dot notation access to dictionary attributes, recursively.
    """
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__

    def __delattr__(self, attr):
        del self[attr]

    def __missing__(self, key):
        self[key] = DotDict()
        return self[key]


  
def model_class_from_str(model_name, model_type=None):
    """
    Retrieve the model class based on the provided model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        class: The model class corresponding to the provided model name.

    Example:
        >>> model = model_class_from_str('encoder1')
        >>> model_instance = model()
        >>> model_instance.train()
    """
    if model_type is None:
        module_name = importlib.import_module(f'models.{model_name}')
    else:
        module_name = importlib.import_module(f'models.{model_type}.{model_name}')
    model_class = getattr(module_name, model_name)
    assert callable(model_class)
    return model_class


def class_from_str(module, class_type):
    """
    Retrieve the model class based on the provided model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        class: The model class corresponding to the provided model name.

    Example:
        >>> model = model_class_from_str('encoder1')
        >>> model_instance = model()
        >>> model_instance.train()
    """

    module_name = importlib.import_module(module)
    
    model_class = getattr(module_name, class_type)
    assert callable(model_class)
    return model_class

def setup_experiment(args: dict) -> dict:
    
    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")
    
    overwrite = args['overwrite']
    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}{args["identifier"]}'    

    # load train config.
    PHD_ROOT = os.getenv("PHD_ROOT")
    sys.path.append(PHD_ROOT)
    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_name}/train.yaml"
    with open(cfg_path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))
    
    if not args['debug']:
        if os.path.exists(experiment_path):
            if overwrite:
                shutil.rmtree(experiment_path)
                print(f'Removing original {experiment_path}')
            else:
                print(f'{experiment_path} already exits. ')
                raise Exception('Experiment name already exists. If you want to overwrite, use flag -ow')

        # create folder to the results.
        os.makedirs(experiment_path)
        os.makedirs(f"{experiment_path}/best_model")
        print(f"Path create: {experiment_path}") 
        shutil.copy(cfg_path, f"{experiment_path}/train.yaml")
    
    return experiment_name, experiment_path, cfg