from utils.common import class_from_str
from gymnasium import Env
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gymnasium as gym


def add_wrappers(env: Env, wrappers: Dict) -> Env:
    
    if wrappers is None: wrappers = {}  

    for wrap in wrappers:
        if not "args" in wrap["wrap"].keys(): wrap["wrap"]["args"] = {}
        
        wrap_class = class_from_str(f"{wrap['wrap']['module']}.{wrap['wrap']['name']}", wrap['wrap']['name'].upper())
        env = wrap_class(env, **wrap['wrap']['args'])
        print(f"Wrapping Env with {wrap_class}")
        
    return env


def make_env(**kwargs):
    #TODO complete th way to use costum environments
    if not (len(kwargs["path"])==0):
        
        print("HELÇLOO")
        print(f"{kwargs['path']}:{kwargs['name']}")
        env = gym.make(f"{kwargs['path']}:{kwargs['path']}/{kwargs['name']}", **kwargs['args'])
        record_env = gym.make(f"{kwargs['path']}:{kwargs['path']}/{kwargs['name']}", render_mode="rgb_array",**kwargs['args'])
    else:
        env = gym.make(kwargs['name'], **kwargs['args'])
        record_env = gym.make(kwargs['name'], render_mode="rgb_array",**kwargs['args'])

    return env, record_env
