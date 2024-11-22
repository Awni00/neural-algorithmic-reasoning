import torch
from torch import nn
import torch.nn.functional as F

# helpers and utilities

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def not_equals(val):
    def inner(x):
        return x != val
    return inner

def equals(val):
    def inner(x):
        return x == val
    return inner

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

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