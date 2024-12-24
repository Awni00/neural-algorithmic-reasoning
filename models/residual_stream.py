import torch
import torch.nn as nn
from functools import partial

VALID_NORM_METHODS = ['pre-norm', 'post-norm', 'pre+post-norm', 'hypersphere-interpolation', 'none']
class ResidualStreamBlock(nn.Module):
    def __init__(self, dim, norm_config=None):
        """This Module applies a residual connection to the input of a model_func, with optional normalization before and/or after the model_func.

        E.g., implements y = x + model_func(norm(x)), in case of pre-norm.

        Parameters
        ----------
        dim : int
            Dimension of the input tenso and output tensors (e.g, d_model)
        norm_config : dict, optional
            norm_type: specifies type of normalization to use (see create_norm for options). Default is 'layernorm'.
            norm_method: specifies whether to apply normalization before or after attention ('none, 'pre-norm', 'post-norm', or 'pre+post-norm'). Default is 'pre-norm'.
        """
        super().__init__()

        self.dim = dim
        self.norm_config = norm_config or {}
        self.norm_method = self.norm_config.get('norm_method', 'pre-norm') # 'pre-norm' or 'post-norm' or 'none'
        assert self.norm_method in VALID_NORM_METHODS, f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}'

        self.norm_type = self.norm_config.get('norm_type', 'layernorm')

        if self.norm_method in ['pre-norm', 'post-norm']:
            self.norm = create_norm(self.dim, self.norm_type) if self.norm_method != 'none' else None

        elif self.norm_method == 'pre+post-norm':
            self.pre_norm = create_norm(self.dim, self.norm_type) if self.norm_method != 'none' else None
            self.post_norm = create_norm(self.dim, self.norm_type) if self.norm_method != 'none' else None

        elif self.norm_method == 'hypersphere-interpolation':
            # this is based on nGPT: Normalized Transformer with Representation Learning on the Hypersphere (arxiv.org/abs/2410.01131)
            # NOTE: this is only implementing the linear interpolation on hypersphere part,
            # nGPT also has other pieces (e.g., weights of linear maps in attention are also normalized to hypersphere)
            # we can consider implementing something like this in the future as well

            lerp_scale = self.norm_config.get('lerp_scale', self.dim ** 0.5)
            lerp_init = self.norm_config.get('lerp_init', 1.0) # NOTE: can be set 1 / n_layers
            self.forward_lerp_weight_scale = lerp_init / lerp_scale
            self.lerp_weight = nn.Parameter(torch.ones(self.dim) * lerp_scale, requires_grad=True)

            # note: norm_type is not used here, we always normalize to unit-norm hypersphere

        elif self.norm_method == 'none':
            pass
        else:
            raise ValueError(f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}')

        self.dim = dim


    def forward(self, x, model_func, **model_kwargs):

        if self.norm_method == 'none':
            y = model_func(x, **model_kwargs)
            x = x + y

        elif self.norm_method == 'pre-norm':
            y = model_func(self.norm(x), **model_kwargs)
            x = x + y

        elif self.norm_method == 'post-norm':
            y = model_func(x, **model_kwargs)
            x = self.norm(x + y)

        elif self.norm_method == 'pre+post-norm':
            y = model_func(self.pre_norm(x), **model_kwargs)
            x = self.post_norm(x + y)

        elif self.norm_method == 'hypersphere-interpolation':
            y = model_func(x, **model_kwargs)
            y = torch.nn.functional.normalize(y, p=2, dim=-1) # normalize to hypersphere (unit-norm)

            x = torch.lerp(x, y, self.lerp_weight * self.forward_lerp_weight_scale) # interpolate between x and y = func(x)
            x = torch.nn.functional.normalize(x, p=2, dim=-1) # normalize to hypersphere (unit-norm)

        else:
            raise ValueError(f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}')

        return x

# TODO: implement hyperspherical interpolation residual connection? Based on: nGPT: Normalized Transformer with Representation Learning on the Hypersphere (arxiv.org/abs/2410.01131)

class ConcatCombine(nn.Module):
    def __init__(self, dim, dim2=None):
        super().__init__()
        self.dim = dim
        self.dim2 = dim2 if dim2 is not None else dim
        self.total_dim = self.dim + self.dim2
        self.combine = nn.Linear(self.total_dim, self.dim, bias = False)

    def forward(self, x, skip):
        concatted_skip = torch.cat((skip, x), dim = -1)
        return self.combine(concatted_skip)

def create_norm(d_model, norm_type):
    if norm_type=='layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type=='rmsnorm':
        return nn.RMSNorm(d_model)
    elif norm_type == 'l2':
        return partial(torch.nn.functional.normalize, dim=-1, p=2)
    elif norm_type=='none':
        return  nn.Identity()
    else:
        raise ValueError(f'norm_type {norm_type} not valid')
