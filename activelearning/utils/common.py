def set_seeds(seed):
    import random

    import numpy as np
    import torch

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


import inspect


def match_kwargs(kwargs_dict, check_function):
    sig = inspect.signature(check_function)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    filtered_dict = {
        filter_key: kwargs_dict[filter_key]
        for filter_key in filter_keys
        if filter_key in kwargs_dict
    }
    return filtered_dict
