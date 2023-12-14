from utils.common import class_from_str
from gymnasium import Env
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

def add_wrappers(env: Env, wrappers: Dict) -> Env:
    
    for wrap in wrappers:
        wrap_class = class_from_str(f"{wrap['wrap']['module']}.{wrap['wrap']['name']}", wrap['wrap']['name'].upper())
        env = wrap_class(env)
        print(f"Wrapping Env with {wrap_class}")
        
    return env