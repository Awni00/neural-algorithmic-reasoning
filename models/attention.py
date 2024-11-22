"""
An implementation of attention including several additional features and customizations over the standard pytorch implementation.
"""

import torch
from torch import nn
from einops import rearrange
import math
from models.positional_encoding import RotaryPositionalEmbeddings, AlibiPositionalBias, T5RelativePositionBias
from models.attention_utils import repeat_kv, apply_rotary_emb, compute_causal_mask
from torch.nn.attention.flex_attention import flex_attention


# these position encoding models have an interface of the form (qseqlen: int, kseqlen: int) -> Tensor[n_heads, qseqlen, kseqlen]
def get_pos_enc_model_type(pos_enc_model):
    # this groups positional encoding models into categories based on their interface
    if any(isinstance(pos_enc_model, model) for model in [AlibiPositionalBias, T5RelativePositionBias]):
        return 'score_bias'
    elif isinstance(pos_enc_model, RotaryPositionalEmbeddings):
        return 'rotary'
    elif pos_enc_model is None:
        return None
    else:
        raise ValueError(f"unknown positional encoding model: {pos_enc_model}")

def get_pos_enc_support(pos_enc_model):
    flash_support = [RotaryPositionalEmbeddings]
    flex_support = [AlibiPositionalBias]
    # NOTE: positional encoding methods that support flash_attention also support flex_attention, but flash_attention is faster
    # so we only use flex_attention when flash_attention is not available

    support_dict = dict(
        flash=any(isinstance(pos_enc_model, model) for model in flash_support), # positional encoding methods that support flash_attention
        flex=any(isinstance(pos_enc_model, model) for model in flex_support), # positional encoding methods that support flex_attention
        manual=True # all support manual
        )
    if pos_enc_model is None:
        support_dict['flash'] = True
        support_dict['flex'] = True
    return support_dict

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
        self.pos_enc_support = get_pos_enc_support(pos_enc_model)

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


        self.block_mask = None


    def create_attn_score_mod(self, bias=None):
        if bias is not None:
            def score_mod(score, b, h, q_idx, kv_idx):
                score_bias = bias[h, q_idx, kv_idx]
                return score + score_bias
        else:
            score_mod = None

        return score_mod

    def build_attn_block_mask(self, mask_mod, qseqlen, kseqlen, device):
        print("WARNING: after building block_mask, is_causal and need_weights will be ignored in forward and self.block_mask will be used instead (if flex attention is being used)")
        self.block_mask = torch.nn.attention.flex_attention.create_block_mask(mask_mod, B=None, H=None, Q_LEN=qseqlen, KV_LEN=kseqlen, device=device)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_func: callable = None,
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and attn_mask
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
        attn_mask : torch.Tensor, optional
            boolean attention mask of shape [len_in, len_ctx]. True at [i,j] indicates i is allowed to attend to j.
            By default None
        is_causal : bool, optional
            whether to apply a causal mask. If True, attn_mask must be None. Only applies for self-attention.
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
            attn_mask = torch.nn.attention.flex_attention.create_mask(mask_func, B=None, H=None, Q_LEN=qseqlen, KV_LEN=kseqlen, device=query.device)
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

        # determine whether to use flash attention or flex attention or manual attention
        # use flash attention if no need for weights and no positional encoding or rotary positional encoding
        use_flash_attn = (not need_weights) and self.pos_enc_support['flash']
        # use flex attention if no need for weights and positional encoding is bias-type and ((block_mask is computed) or no need for block_mask)
        use_flex_attn = (not need_weights) and self.pos_enc_support['flex'] and (self.block_mask is not None or (mask_func is None and not is_causal))

        # can use F.scaled_dot_product_attention's implementation of flash attention
        if use_flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.attn_scale)
            scores = None

        # use flex attention (e.g., supports score modification like relative positional bias)
        elif use_flex_attn:
            # compute bias to be added to attention score
            if self.pos_enc_model_type in ['score_bias']:
                scores_bias = self.pos_enc_model(qseqlen, kseqlen)
            else:
                scores_bias = None
            # create `score_mod` for flex_attention
            score_mod = self.create_attn_score_mod(bias=scores_bias)

            # NOTE: flex_attention does not support dropout on attention scores (unlike flash attention or manual implementation below)
            output = flex_attention(
                xq, xk, xv, score_mod=score_mod, block_mask=self.block_mask, scale=self.attn_scale)
            scores = None

        # manual implementation (which exposes attention scores)
        else:
            # compute dot product attention scores (pre-softmax)
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale

            # apply bias-based positional encoding (if applicable)
            if self.pos_enc_model_type in ['score_bias']:
                scores_bias = self.pos_enc_model(qseqlen, kseqlen)
                scores = scores + scores_bias

            if attn_mask is not None:
                attn_mask_ = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
                scores = scores + attn_mask_

            # apply softmax activation to inner products
            scores = torch.nn.functional.softmax(scores, dim=-1)

            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, qseqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores
