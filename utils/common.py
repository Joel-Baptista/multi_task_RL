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

def setup_experiment(args: dict, file: str = "train.yaml") -> dict:

    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")
    
    overwrite = args['overwrite']
    experiment_name = args['experiment_name']

    if args["identifier"] == "": args["identifier"] = "1"
    elif args["identifier"] == "auto":

        files = os.listdir(f'{os.getenv("PHD_MODELS")}/{experiment_name}')
        folder_experiments = [int(s) for s in files if s.isdigit()]

        if len(folder_experiments) == 0:
            args["identifier"] = 1
        else:
            folder_experiments.sort()
            args["identifier"] = folder_experiments[-1] + 1 
            
    experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}/{args["identifier"]}'    

    # load train config.
    PHD_ROOT = os.getenv("PHD_ROOT")
    sys.path.append(PHD_ROOT)

    experiment_base = None
    for folder in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/"):
        print(folder)
        for experiment in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/{folder}"):
            if experiment == experiment_name: experiment_base = folder
        if folder is not None: break 
    
    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_base}/{experiment_name}/{file}"
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

def setup_test(args: dict) -> dict:

    if args["experiment_name"] is None:
        args["experiment_name"] = "baseline"
        print(f"{Fore.YELLOW}Missing input 'experiment_name'. Assumed to be 'baseline'{Fore.RESET}")

    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}'    

    # load train config.
    PHD_ROOT = os.getenv("PHD_ROOT")
    sys.path.append(PHD_ROOT)

    experiment_base = None
    for folder in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/"):
        print(folder)
        for experiment in os.listdir(f"{PHD_ROOT}/multi_task_RL/experiments/{folder}"):
            if experiment == experiment_name: experiment_base = folder
        if folder is not None: break 


    cfg_path = f"{PHD_ROOT}/multi_task_RL/experiments/{experiment_base}/{experiment_name}/test.yaml"
    log_path = f'{os.getenv("PHD_MODELS")}/{experiment_name}/{args["identifier"]}'
    
    # experiment_path += args['identifier']
    experiment_path += f"/{args['identifier']}"

    print(experiment_path)
    with open(cfg_path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))

    if not os.path.exists(experiment_path):
        raise Exception(f"Results from experiment '{experiment_name}' does not exist in path: {experiment_path}")
    
    return log_path, experiment_path, cfg