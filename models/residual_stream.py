import torch
import torch.nn as nn
from functools import partial

VALID_NORM_METHODS = ['pre-norm', 'post-norm', 'pre+post-norm', 'hypersphere-interpolation', 'hypersphere-spherical-interpolation', 'adaptive-hypersphere-interpolation', 'none']
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
            # use only valid kwargs in norm_config
            self.res_stream = HypersphereLERP(dim, lerp_weight_constraint=self.norm_config.get('lerp_weight_constraint', 'none'))
            # lerp_scale = self.norm_config.get('lerp_scale', self.dim ** 0.5)
            # lerp_init = self.norm_config.get('lerp_init', 1.0) # NOTE: can be set 1 / n_layers
            # self.forward_lerp_weight_scale = lerp_init / lerp_scale
            # self.lerp_weight = nn.Parameter(torch.ones(self.dim) * lerp_scale, requires_grad=True)

            # note: norm_type is not used here, we always normalize to unit-norm hypersphere
        elif self.norm_method == 'hypersphere-spherical-interpolation':
            self.res_stream = HypersphereSLERP(dim, single_weight=self.norm_config.get('single_weight', True))

        elif self.norm_method == 'adaptive-hypersphere-interpolation':
            self.res_stream = AdaptiveHypersphereSLERP(dim, single_weight=self.norm_config.get('single_weight', True))

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

        elif self.norm_method in ['hypersphere-interpolation', 'hypersphere-spherical-interpolation', 'adaptive-hypersphere-interpolation']:
            y = model_func(x, **model_kwargs)
            # y = torch.nn.functional.normalize(y, p=2, dim=-1) # normalize to hypersphere (unit-norm)

            # # x = torch.lerp(x, y, self.lerp_weight * self.forward_lerp_weight_scale) # interpolate between x and y = func(x)
            # x = x + (self.lerp_weight * self.forward_lerp_weight_scale) * (y - x) # interpolate between x and y = func(x)
            # x = torch.nn.functional.normalize(x, p=2, dim=-1) # normalize to hypersphere (unit-norm)
            x = self.res_stream(x, y)
        else:
            raise ValueError(f'norm_method {self.norm_method} not valid; must be in {VALID_NORM_METHODS}')

        return x

class HypersphereLERP(nn.Module):
    """
    Implements linear interpolation on the hypersphere, based on the nGPT paper:
    "Normalized Transformer with Representation Learning on the Hypersphere" (arxiv.org/abs/2410.01131).

    The basic idea is to maintain embeddings on the unit-norm hypersphere and update them by interpolating
    along geodesics on the hypersphere. This class approximates spherical interpolation (SLERP) as
    linear interpolation (LERP), as proposed in the nGPT paper.

    SLERP(a, b; alpha) = sin((1-alpha) * theta) / sin(theta) * a + sin(alpha * theta) / sin(theta) * b,
    where theta is the angle between a and b = arccos(<a, b>), and alpha is the interpolation weight in [0, 1].

    Here, following the nGPT paper, we approximate this by:
    LERP(a, b; alpha) = a + alpha * (b - a) = (1 - alpha) * a + alpha * b.
    In the limit as theta -> 0, SLERP(a, b; alpha) -> LERP(a, b; alpha).

    Parameters
    ----------
    dim : int
        Dimension of the input and output tensors.
    lerp_scale : float, optional
        Scale factor for the interpolation weight. Default is sqrt(dim).
    lerp_init : float, optional
        Initial value for the interpolation weight. Default is 1.0.
    lerp_weight_constraint : str, optional
        Constraint to apply to the interpolation weight. Options are 'none', 'sigmoid', 'abs', 'clamp'. Default is 'none'.

    Methods
    -------
    forward(x, y)
        Performs the linear interpolation on the hypersphere between tensors x and y.

    """

    def __init__(self, dim, lerp_scale=None, lerp_init=1.0, lerp_weight_constraint='none'):
        super().__init__()

        self.dim = dim
        self.lerp_init = lerp_init
        self.lerp_scale = lerp_scale if lerp_scale is not None else self.dim ** 0.5
        self.lerp_weight = nn.Parameter(torch.ones(self.dim) * self.lerp_scale, requires_grad=True)
        self.forward_lerp_weight_scale = self.lerp_init / self.lerp_scale

        # if normalize_lerp_weight, then normalize lerp_weight to [0,1] using sigmoid
        # NOTE: in nGPT paper, they don't normalize interpolation weight alpha
        # (which is a bit confusing to me, since operation is not interpretable and may be strongly biased to)
        self.lerp_weight_constraint = lerp_weight_constraint # whether to normalize lerp_weight to [0,1]
        assert lerp_weight_constraint in ['none', 'sigmoid', 'abs', 'clamp']
        self.lerp_weight_constraint_fn = {
            'none': lambda x: x,
            'sigmoid': lambda x: x.sigmoid(),
            'abs': lambda x: torch.abs(x),
            'clamp': lambda x: x.clamp(0, 1),
        }.get(lerp_weight_constraint)

    def forward(self, x, y):
        # normalize/project to hypersphere
        # typically (e.g., in ResNet architecture with this resstream method), x will already be normalized to unit norm
        x, y = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(y, p=2, dim=-1)

        interpolation_weight = self.lerp_weight_constraint_fn(self.lerp_weight * self.forward_lerp_weight_scale)
        x = x + interpolation_weight * (y - x)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x

class HypersphereSLERP(nn.Module):
    """
    This module implements spherical interpolation (slerp) between two vectors on the unit-norm hypersphere.

    Intended to be used as a way to update embeddings in the "residual stream" of a model.

    SLERP(x, y; alpha) = sin((1-alpha) * theta) / sin(theta) * x + sin(alpha * theta) / sin(theta) * y,
    where theta = angle between x and y = arccos(<x, y>), and alpha is the interpolation weight in [0, 1].

    Unlike HypersphereLERP, this does not use a linear approximation, and strictly enforces alpha to be in [0,1].

    Parameters
    ----------
    dim : int
        Dimension of the input and output tensors.
    single_weight : bool, optional
        If True, use a single scalar weight for all dimensions; otherwise, use a separate weight for each dimension. Default is True.

    Methods
    -------
    forward(x, y)
        Performs the spherical interpolation on the hypersphere between tensors x and y.
    """

    def __init__(self, dim, single_weight=True):
        super().__init__()

        self.dim = dim
        self.single_weight = single_weight

        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight = nn.Parameter(torch.ones(1) if single_weight else torch.ones(self.dim), requires_grad=True)
        # what is geometric interpretation of single_weight = False?

    def forward(self, x, y):
        # x, y: [batch_size, ..., dim]

        # normalize to unit norm
        x, y = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(y, p=2, dim=-1)
        cos_theta = (x * y).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., 1]


        # sigmoid to ensure map interpolation weight to [0,1]
        alpha = self.slerp_weight.sigmoid() # shape: [1] or [dim]

        x = torch.sin((1 - alpha) * theta) / sin_theta * x + torch.sin(alpha * theta) / sin_theta * y
        # shape: [batch_size, ..., dim], where each vector is interpolated between x and y
        # norm(x, dim=-1) = 1 (i.e., preserves unit-norm after interpolation)

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x

class AdaptiveHypersphereSLERP(nn.Module):
    """
    An adaptive variant of HypersphereSLERP, where the interpolation weight is a learned function of the update direction y.

    Intended to be used as a way to update embeddings in the "residual stream" of a model.

    SLERP(x, y; alpha) = sin((1-alpha) * theta) / sin(theta) * x + sin(alpha * theta) / sin(theta) * y,
    where theta = the angle between x and y = arccos(<x, y>), and alpha is the interpolation weight in [0, 1].

    Here, alpha is computed as alpha = sigmoid(y * W_alpha), where W_alpha is a learnable weight matrix.

    Parameters
    ----------
    dim : int
        Dimension of the input and output tensors.
    single_weight : bool, optional
        If True, use a single scalar weight for all dimensions; otherwise, use a separate weight for each dimension. Default is True.

    Methods
    -------
    forward(x, y)
        Performs the (adaptive) spherical interpolation on the hypersphere between tensors x and y.
    """

    def __init__(self, dim, single_weight=True):
        super().__init__()

        self.dim = dim
        self.single_weight = single_weight

        # linear map from y to interpolation weight alpha
        # if single_weight, then use a single scalar weight for all dimensions;
        # otherwise, use a separate weight for each dimension
        self.slerp_weight_map = nn.Linear(dim, 1) if single_weight else nn.Linear(dim, dim)

    def forward(self, x, y):
        # x, y: [batch_size, ..., dim]

        # normalize to unit norm
        x, y = torch.nn.functional.normalize(x, p=2, dim=-1), torch.nn.functional.normalize(y, p=2, dim=-1)
        cos_theta = (x * y).sum(dim=-1, keepdim=True) # shape: [batch_size, ..., 1]
        theta = torch.acos(cos_theta) # shape: [batch_size, ..., 1]
        sin_theta = torch.sin(theta) # shape: [batch_size, ..., 1]


        # sigmoid to ensure map interpolation weight to [0,1]
        alpha = self.slerp_weight_map(y).sigmoid() # shape: [1] or [dim]

        x = torch.sin((1 - alpha) * theta) / sin_theta * x + torch.sin(alpha * theta) / sin_theta * y
        # shape: [batch_size, ..., dim], where each vector is interpolated between x and y
        # norm(x, dim=-1) = 1 (i.e., preserves unit-norm after interpolation)

        # if not single weight, this is not strictly spherical interpolation, and may not preserve unit-norm
        if not self.single_weight:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x

    # TODO: IDEA: alpha interpolation weight can be learnable function of x and/or y. This implements a natural type of gating mechanism.

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
