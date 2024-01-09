from utils.common import class_from_str
from gymnasium import Env
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union


def add_wrappers(env: Env, wrappers: Dict) -> Env:
    
    if wrappers is None: wrappers = {}  

    for wrap in wrappers:
        if not "args" in wrap["wrap"].keys(): wrap["wrap"]["args"] = {}
        
        wrap_class = class_from_str(f"{wrap['wrap']['module']}.{wrap['wrap']['name']}", wrap['wrap']['name'].upper())
        env = wrap_class(env, **wrap['wrap']['args'])
        print(f"Wrapping Env with {wrap_class}")
        
    return env
