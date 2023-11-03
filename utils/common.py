import importlib


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
