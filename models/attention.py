"""
An implementation of attention including several additional features and customizations over the standard pytorch implementation.
"""

import torch
from torch import nn
from einops import rearrange
import math
from .positional_encoding import RotaryPositionalEmbeddings, AlibiPositionalBias, T5RelativePositionBias, ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding
from .attention_utils import repeat_kv, compute_causal_mask, get_attention_function

class Attention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            pos_enc_model: nn.Module = None,
            key_dim: int = None,
            n_kv_heads: int = None,
            dropout: float = 0.0,
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
            symmetric_attn: bool = False,
            attn_score_fn: str = 'softmax',
            attn_score_fn_params: dict = None,
            ):
        """
        An implementation of Attention with some added customization.

        Allows multi-query attention/grouped query attention, rotary positional embeddings,
        and custom relation activation functions.

        Parameters
        ----------
        d_model : int
            model dimension
        n_heads : int
            number of heads (query heads if n_kv_heads is set)
        pos_enc_model : nn.Module, optional
            positional encoding model, e.g., RoPE, T5RelativePositionalBias, etc. (default is None)
        dropout : float
            dropout rate
        n_kv_heads : int, optional
            number of key/value heads. used to implement multi-query attention or grouped query attention.
            n_kv_heads=1 corresponds to MQA, n_kv_heads > 1 corresponsd to grouped query attention.
            n_kv_heads=n_heads is standard MHA. uses MHA when None. By default None
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        symmetric_attn : bool, optional
            whether to weight-tie the query and key projections, making a symmetric attention criterion. By default False
        attn_score_fn : str, optional
            activation function for attention scores. One of 'softmax', 'hard', 'topk-softmax', 'sigmoid', or 'linear' (default is 'softmax').
        attn_score_fn_params : dict, optional
            additional parameters for the attention score function, e.g., whether to use straight-through estimator for sparse softmax variants, etc. (default is None)
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads # number of heads (for query)
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.symmetric_attn = symmetric_attn

        self.pos_enc_model = pos_enc_model
        self.pos_enc_model_type = get_pos_enc_model_type(pos_enc_model)

        self.key_dim = key_dim if key_dim is not None else self.d_model // self.n_heads # key dimension
        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.head_dim = self.d_model // self.n_heads # dim of projections
        assert self.n_heads % self.n_kv_heads == 0 # make sure n_kv_heads fits into n_heads (i.e., can be grouped)
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.key_dim) # for scaled dot product attention

        self.wq = nn.Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)
        if symmetric_attn:
            self.wk = self.wq
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # activation function for attention scores (e.g., softmax, hard, topk-softmax, sigmoid, linear)
        self.attn_score_fn = attn_score_fn
        self.attn_score_fn_params = attn_score_fn_params or {}
        self.attn_score_fn_ = get_attention_function(self.attn_score_fn, self.attn_score_fn_params)

        # check whether configuration (namely positional encoding model and attention score function) supports flash attention
        self.support_flash = self.is_flash_supported()

    def is_flash_supported(self):
        pos_enc_support = get_pos_enc_support(self.pos_enc_model)
        attn_func_support = self.attn_score_fn == 'softmax'
        return pos_enc_support['flash'] and attn_func_support

    def create_attn_score_mod(self, bias=None):
        if bias is not None:
            def score_mod(score, b, h, q_idx, kv_idx):
                score_bias = bias[h, q_idx, kv_idx]
                return score + score_bias
        else:
            score_mod = None

        return score_mod

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_func: callable = None,
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and mask_func
        need_weights: bool = False
    ):
        """
        compute attention with given query, key, value.

        if freqs_cos and freqs_sin are given, apply rotary positional embeddings.
        if attn_mask is given, apply attention mask.
        if is_causal is True, apply causal mask (attn_mask must be None).

        Parameters
        ----------
        query : torch.Tensor
            query sequence of shape [bsz, len_in, d_model]
        key : torch.Tensor
            key sequence of shape [bsz, len_ctx, d_model]
        value : torch.Tensor
            value sequence of shape [bsz, len_ctx, d_model]
        mask_func : callable, optional
            mask_mod function. This is a callable that defines the masking pattern for the attention mechanism. It takes four arguments: b (batch size), h (number of heads), q_idx (query index), and kv_idx (key/value index). It should return a boolean tensor indicating which attention connections are allowed (True) or masked out (False).
        is_causal : bool, optional
            whether to apply a causal mask. If True, mask_func must be None. Only applies for self-attention.
            By default False
        need_weights : bool, optional
            whether to return the attention scores. If True, return value will be tuple (output, attn_scores).
            If True, will compute attention manually rather than using flash attention. By default False

        Returns
        -------
        torch.Tensor
            result of attention
        """

        bsz, qseqlen, _ = query.shape
        bsz, kseqlen, _ = key.shape
        bsz, vseqlen, _ = value.shape

        assert kseqlen == vseqlen, "key and value sequences must have the same length"

        assert not (mask_func is not None and is_causal), "only one of attn_mask and is_causal should be set"
        # compute causal mask if is_causal and no maks given
        if is_causal and mask_func is None:
            assert qseqlen == kseqlen, "query and key sequences must have the same length for causal mask"
            attn_mask = compute_causal_mask(qseqlen, device=query.device)
        elif not is_causal and mask_func is not None:
            # TODO: avoid flex_attention dependency now that I've removed it
            attn_mask = torch.nn.attention.flex_attention.create_mask(
                mask_func, B=None, H=None, Q_LEN=qseqlen, KV_LEN=kseqlen, device=query.device)
        else:
            attn_mask = None

        # apply query/key/value projections and reshape to split into different heads
        xq, xk, xv = self.wq(query), self.wk(key), self.wv(value)
        xq = xq.view(bsz, qseqlen, self.n_heads, self.key_dim) # shape (bs, seqlen, n_heads, key_dim)
        xk = xk.view(bsz, kseqlen, self.n_kv_heads, self.key_dim) # shape (bs, seqlen, n_kv_heads, key_dim)
        xv = xv.view(bsz, vseqlen, self.n_kv_heads, self.head_dim) # shape (bs, seqlen, n_kv_heads, head_dim)

        # apply RoPE to queries and keys (if positional encoding model is RoPE)
        if self.pos_enc_model_type == 'rotary':
            # recall that RotaryPositionalEmbeddings expects an input of shape (bs, seqlen, n_heads, key_dim)
            xq = self.pos_enc_model(xq)
            xk = self.pos_enc_model(xk)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)  # (bs, seqlen, n_heads, key_dim)
            xv = repeat_kv(xv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, key_dim)
        xk = xk.transpose(1, 2)  # (bs, n_heads, seqlen, key_dim)
        xv = xv.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)

        # determine whether to use flash attention or manual attention
        # use flash attention if no need for weights and positional encoding method supports it
        use_flash_attn = (not need_weights) and self.support_flash and (not mask_func)

        # can use F.scaled_dot_product_attention's implementation of flash attention
        if use_flash_attn:

            # fixed bias-based positional encoding method (e.g., AlibiPositionalBias)
            if self.pos_enc_model_type in ['score_bias']:
                scores_bias = self.pos_enc_model(qseqlen, kseqlen)
                if attn_mask is not None:
                    mask_bias = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
                    scores_bias = scores_bias + mask_bias

                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=scores_bias, dropout_p=self.dropout if self.training else 0.0, scale=self.attn_scale)

            # pos enc already applied to xq and/or xk (e.g., RoPE or NoPE)
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.attn_scale)
            scores = None

        # manual implementation (which explicitly computes attention scores)
        else:
            # compute dot product attention scores (pre-softmax)
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale

            # apply bias-based positional encoding (if applicable)
            if self.pos_enc_model_type in ['score_bias']:
                scores_bias = self.pos_enc_model(qseqlen, kseqlen)
                scores = scores + scores_bias

            if attn_mask is not None and self.attn_score_fn in ['softmax', 'topk-softmax', 'hard']:
                attn_mask_ = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
                scores = scores + attn_mask_

            # apply softmax (or other) activation to inner products
            scores = self.attn_score_fn_(scores)

            if attn_mask is not None and self.attn_score_fn not in ['softmax', 'topk-softmax', 'hard']:
                scores = scores.masked_fill(attn_mask.logical_not(), 0)

            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, qseqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores

# these position encoding models have an interface of the form (qseqlen: int, kseqlen: int) -> Tensor[n_heads, qseqlen, kseqlen]
def get_pos_enc_model_type(pos_enc_model):
    # this groups positional encoding models into categories based on their interface
    if any(isinstance(pos_enc_model, model) for model in [AlibiPositionalBias, T5RelativePositionBias]):
        return 'score_bias'
    elif isinstance(pos_enc_model, RotaryPositionalEmbeddings):
        return 'rotary'
    elif pos_enc_model is None:
        return None
    elif any(isinstance(pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
        return None
    else:
        raise ValueError(f"unknown positional encoding model: {pos_enc_model}")

def get_pos_enc_support(pos_enc_model):
    flash_support = [RotaryPositionalEmbeddings, AlibiPositionalBias]
    # NOTE: T5RelativePositionBias does not support flash attention because flash attention requires a fixed bias (cannot backprop)

    support_dict = dict(
        flash=any(isinstance(pos_enc_model, model) for model in flash_support), # positional encoding methods that support flash attention
        manual=True # all support manual
        )
    if pos_enc_model is None:
        support_dict['flash'] = True
    return support_dict