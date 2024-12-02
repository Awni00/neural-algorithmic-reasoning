import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict
import torch

# region: AttributeDict

class AttributeDict(Dict):
    """
    A drop-in replacement for a Python dictionary, with the additional functionality to access and modify keys
    through attribute lookup for convenience.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # recursively convert nested dictionaries to AttributeDicts
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttributeDict(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delattr__(self, item: str) -> None:
        if item not in self:
            raise KeyError(item)
        del self[item]

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        return "\n".join(rows)

    def todict(self) -> dict:
        """
        Convert the `AttributeDict` to a dictionary.

        Returns
        -------
        dict
            The dictionary.
        """
        return {k: (v.todict() if isinstance(v, AttributeDict) else v) for k, v in self.items()}

# endregion

def print_gpu_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

# region: general helpers and utilities

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# endregion

# region tensor and pytorch utils

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# endregion