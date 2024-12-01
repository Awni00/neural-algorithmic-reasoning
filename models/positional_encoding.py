import sys
import os
import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, ModuleDict

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils import exists, default, l2norm, pad_at_dim, Sequential

# region
# code in this region is based on https://github.com/lucidrains/x-transformers/blob/144d9ba84955139347e798ab025457b2d7adc314/x_transformers/x_transformers.py (November 8, 2024)


class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.0
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert (
            seq_len <= self.max_seq_len
        ), f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (
            dim % 2 == 0
        ), "dimension of the model must be divisible by 2 for sinusoidal positional encoding"
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


# This is adapted from Mesh Tensorflow: 
# https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
# perhaps via Huggingface Transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L344

class T5RelativePositionBias(Module):
    def __init__(self,  heads, scale=1, causal=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal, num_buckets, max_distance):
        """
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.
        If causal=True, then positive relative positions are
        invalid.

        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            causal: a boolean - whether the attention is causal
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, qlen, klen):
        device = self.device
        q_pos = torch.arange(klen - qlen, klen, dtype=torch.long, device=device)
        k_pos = torch.arange(klen, dtype=torch.long, device=device)

        rel_pos = einx.subtract("j, i -> i j", k_pos, q_pos)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """

        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return bias * self.scale


class CoPE(Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """

    def __init__(
        self,
        dim,
        heads,
        max_pos,
        soft_onehot=False,
        talking_heads=False,
        soft_onehot_temp=5e-2,
    ):
        super().__init__()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = (
            nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else None
        )
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer("positions", torch.arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(
                causal_mask, -torch.finfo(attn_logits.dtype).max
            )

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.max_pos - 1)

        logits_int = einsum("b h n d, p d -> b h n p", query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract("i, j -> i j", pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim=-1)
            cope_pos_emb = einsum(
                "b h i j p, b h i p -> b h i j", soft_onehot_pos, logits_int
            )
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb


class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert (
            depth >= 1
        ), "depth for dynamic position bias MLP must be greater or equal to 1"
        self.log_distance = log_distance

        self.mlp = ModuleList([])

        self.mlp.append(
            Sequential(nn.Linear(1, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU())
        )

        for _ in range(depth - 1):
            self.mlp.append(
                Sequential(
                    nn.Linear(dim, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU()
                )
            )

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, qlen, klen):
        assert qlen == klen
        n, device = klen, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = einx.subtract("i, j -> i j", seq_arange, context_arange)
        indices += n - 1

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device).float()
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(
                pos.abs() + 1
            )  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


class AlibiPositionalBias(Module):
    def __init__(
        self, heads, total_heads=None, slopes: list[int] | None = None, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, "h -> h 1 1")

        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward_custom_pos(self, pos_i: Tensor, pos_j: Tensor | None = None):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract("... j, ... i -> ... i j", pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, "b i j -> b 1 i j")

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        return bias

    def forward(self, qlen, klen):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= klen and self.bias.shape[-2] >= qlen:
            return self.bias[..., -qlen:, -klen:]

        seq_arange = torch.arange(klen - qlen, klen, device=device)
        context_arange = torch.arange(klen, device=device)
        bias = -einx.subtract("j, i -> 1 i j", context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        self.register_buffer("bias", bias, persistent=False)
        return self.bias


class DataDependentAlibi(Module):
    """https://openreview.net/forum?id=q2Lnyegkr8"""

    def __init__(
        self,
        dim,
        heads,
        causal=True,
        bias_init=5.0,
        post_log_scale=1.0,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear, Rearrange("b n h -> b h n"), nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim=-1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim=1)

        forget_gates = einx.subtract(
            "b h i, b h j -> b h i j", forget_gates, forget_gates
        )

        if bidirectional:
            forget_gates_reversed = einx.subtract(
                "b h j, b h i -> b h i j", forget_gates_reversed, forget_gates_reversed
            )
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates


class PerRowDataDependentAlibi(Module):
    """same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension"""

    def __init__(self, dim, heads, causal=True, dim_head=8, post_log_scale=1.0):
        super().__init__()
        assert causal, "bidirectional not supported yet"

        self.scale = dim_head**-0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias=False)

        self.to_forget_gates = nn.Sequential(
            linear, Rearrange("b n (qk h d) -> qk b h n d", qk=2, d=dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum("... i d, ... j d -> ... i j", q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype=torch.bool, device=x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.0)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims=(-1,))
        forget_gates = forget_gates.cumsum(dim=-1)
        forget_gates = forget_gates.flip(dims=(-1,))

        return forget_gates


# endregion


# this implementation is from the torchtune package, and is based on the official llama source code
# https://github.com/pytorch/torchtune/blob/bce70917c3d0d1f7693c9ae8b59cd72ee55b659d/torchtune/modules/position_embeddings.py (11/11/2024)
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
