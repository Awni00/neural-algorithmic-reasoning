import torch
from typing import Any, Optional, Tuple
from functools import partial
from .adaptive_temperature_softmax import AdaptiveTemperatureSoftmax


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# NOTE: be careful. pytorch API is inconsistent about whether True means attend or not attend. 
# this works with the Attention module implemented above, but will only be compatible with some but not all pytorch implementations
# e.g., works with nn.functional.scaled_dot_product_attention but not nn.MultiHeadAttention
def compute_diag_mask(size, device=None):
    """computes an attention mask with False on the diagonal and True elsewhere"""

    diag_mask = torch.eye(size, device=device).logical_not()
    # diag_mask = diag_mask.masked_fill(diag_mask == 1, float('-inf'))
    return diag_mask

def compute_causal_mask(size, device=None):
    """computes an attention mask with True at (i,j) if i <= j"""
    causal_mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return causal_mask


# region alternative attention activations (e.g., softmax, topk-softmax, one-hot straight-through)

def topk_softmax(logits: torch.Tensor, k: int, straight_through: bool = False) -> torch.Tensor:
    """
    Apply top-k softmax to the logits.

    Parameters
    ----------
    logits : torch.Tensor
        [batch_size, n_heads, seq_len, seq_len] tensor of logits.
    k : int
        The number of top elements to consider.
    straight_through : bool, optional
        Whether to use the straight-through estimator (default is False).

    Returns
    -------
    torch.Tensor
        topk-softmax attention scores.
    """

    orig_logits = logits

    mask_value = -torch.finfo(logits.dtype).max
    top_values, _ = logits.topk(k, dim = -1)
    sparse_topk_mask = (logits >= top_values[..., -1:]) & (logits > mask_value)
    logits = logits.masked_fill(~sparse_topk_mask, mask_value)
    topk_attn = logits.softmax(dim = -1)

    if straight_through:
        # straight-through estimator: value returned is topk_attn, but gradient is soft_attn

        soft_attn = orig_logits.softmax(dim = -1)
        return topk_attn.detach() + soft_attn - soft_attn.detach()
    else:
        return topk_attn

def get_attention_function(activation: str, kwargs: dict) -> Any:
    """
    Get the attention function based on the activation.

    Parameters
    ----------
    activation : str
        The activation function.
    kwargs : dict
        The keyword arguments for the activation function (if applicable).

    Returns
    -------
    Any
        The attention function.
    """

    if activation == "softmax":
        return partial(torch.nn.functional.softmax, dim=-1)
    if activation == 'adaptive-temperature-softmax':
        return AdaptiveTemperatureSoftmax(**kwargs)
    elif activation == "topk-softmax":
        return partial(topk_softmax, **kwargs)
    elif activation == "hard":
        return partial(topk_softmax, k=1, **kwargs)
    elif activation == "sigmoid":
        return torch.nn.functional.sigmoid
    elif activation == "relu":
        return torch.nn.functional.relu
    elif activation == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Activation function {activation} not valid.")
# endregion

# TODO: implement different similarity functions
# TODO: e.g., Selective Attention, which masks *future* tokens: https://arxiv.org/pdf/2410.02703
# TODO: allow for more finegrained control over backend to use in torch.nn.functional.scaled_dot_product_attention: https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html

# TODO: consider implmenting a "Quiet Attention" as described in https://www.evanmiller.org/attention-is-off-by-one.html