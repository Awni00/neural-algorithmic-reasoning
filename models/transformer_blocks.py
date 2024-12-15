import torch
import torch.nn as nn
from .attention import Attention

class EncoderBlock(nn.Module):

    def __init__(self,
            d_model: int,
            n_heads: int,
            pos_enc_model = None,
            dff: int = None,
            activation: str = 'relu',
            norm_config: dict = None,
            dropout_rate: float = 0.0,
            bias: bool = True,
            causal: bool = False,
            attn_kwargs: dict = None,
            ):
        """
        A Transformer Encoder Block.

        Consists of Self-attention, Feed-forward block and LayerNorms/Residuals.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads : int
            number of self-attention heads.
        pos_enc_model : nn.Module, optional
            positional encoding model to use. Default is None.
        dff : int
            intermediate dimension of feed-forward block.
        activation : str
            name of activation function to use in feed-forward block.
        norm_config: dict, optional
            norm_type: specifies type of normalization to use ('layernorm' or 'rmsnorm'). Default is 'layernorm'.
            norm_method: specifies whether to apply normalization before or after attention ('pre-norm' or 'post-norm'). Default is 'pre-norm'.
        dropout_rate : float
            dropout rate.
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_config = norm_config or {}
        self.norm_method = self.norm_config.get('norm_method', 'pre-norm') # 'pre-norm' or 'post-norm' or 'none'
        assert self.norm_method in ['pre-norm', 'post-norm', 'none'], f'norm_method {self.norm_method} not valid'
        self.norm_type = self.norm_config.get('norm_type', 'layernorm')
        self.bias = bias
        self.attn_kwargs = {'n_kv_heads': None, 'add_bias_kv': False}
        if attn_kwargs is not None:
            self.attn_kwargs.update(attn_kwargs)
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = create_norm(self.d_model, self.norm_type) if self.norm_method != 'none' else None
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model,
            add_bias_out=self.bias, dropout=self.dropout_rate, **self.attn_kwargs)
        self.norm2 = create_norm(self.d_model, self.norm_type) if self.norm_method != 'none' else None
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)


    def forward(self, x, need_weights=False):
        if self.norm_method == 'pre-norm':
            y = self._compute_self_attn(self.norm1(x), need_weights=need_weights)
            x = x + y

            y = self._apply_ff_block(self.norm2(x))
            x = x + y
        elif self.norm_method == 'post-norm':
            y = self._compute_self_attn(x, need_weights=need_weights)
            x = self.norm1(x + y)

            x = self.dropout(x)
            y = self._apply_ff_block(x)
            x = self.norm2(x + y)
        else:
            y = self._compute_self_attn(x, need_weights=need_weights)
            x = x + y

            x = self.dropout(x)
            y = self._apply_ff_block(x)
            x = x + y
        return x

    def _compute_self_attn(self, x, need_weights=False):
        x, _ = self.self_attn(query=x, key=x, value=x, is_causal=self.causal,
            need_weights=need_weights)
        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            n_heads_cross: int,
            pos_enc_model_sa = None,
            pos_enc_model_ca = None,
            dff: int = None,
            activation: str = 'relu',
            norm_config: dict = None,
            dropout_rate: float = 0.,
            bias: bool = True,
            causal: bool = False,
            attn_kwargs: dict = None,
            ):
        """
        A Transformer Decoder Block.

        Consists of Self-attention, Cross-attention, Feed-forward block and LayerNorms/Residuals.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads : int
            number of self-attention heads.
        n_heads_cross : int
            number of cross-attention heads.
        pos_enc_model_sa : nn.Module, optional
            positional encoding model to use. Default is None.
        pos_enc_model_ca : nn.Module, optional
            positional encoding model to use. Default is None.
        dff : int
            intermediate dimension of feed-forward block.
        activation : str
            name of activation function to use in feed-forward block.
        norm_config: dict, optional
            norm_type: specifies type of normalization to use ('layernorm' or 'rmsnorm'). Default is 'layernorm'.
            norm_method: specifies whether to apply normalization before or after attention ('pre-norm' or 'post-norm'). Default is 'pre-norm'.
        dropout_rate : float
            dropout rate.
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        attn_kwargs : dict, optional
            keyword arguments for Attention, by default None
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_heads_cross = n_heads_cross
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_config = norm_config or {}
        self.norm_method = self.norm_config.get('norm_method', 'pre-norm') # 'pre-norm' or 'post-norm' or 'none'
        assert self.norm_method in ['pre-norm', 'post-norm', 'none'], f'norm_method {self.norm_method} not valid'
        self.norm_type = self.norm_config.get('norm_type', 'layernorm')
        self.bias = bias
        self.causal = causal
        self.attn_kwargs = {'n_kv_heads': None, 'add_bias_kv': False}
        if attn_kwargs is not None:
            self.attn_kwargs.update(attn_kwargs)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = create_norm(self.d_model, self.norm_type) if self.norm_method != 'none' else None
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model_sa,
            add_bias_out=self.bias, dropout=self.dropout_rate, **self.attn_kwargs)
        self.norm2 = create_norm(self.d_model, self.norm_type) if self.norm_method != 'none' else None
        self.cross_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads_cross, pos_enc_model=pos_enc_model_ca,
            add_bias_out=self.bias, dropout=self.dropout_rate, **self.attn_kwargs)
        self.norm3 = create_norm(self.d_model, self.norm_type) if self.norm_method != 'none' else None
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    def forward(self, x, context):
        if self.norm_method == 'pre-norm':
            x = x + self._compute_self_attn(self.norm1(x))
            x = x + self._compute_cross_attn(self.norm2(x), context)
            x = x + self._apply_ff_block(self.norm3(x))
        elif self.norm_method == 'post-norm':
            x = self.norm1(x + self._compute_self_attn(x))
            x = self.norm2(x + self._compute_cross_attn(x, context))
            x = self.norm3(x + self._apply_ff_block(x))
        else:
            x = x + self._compute_self_attn(x)
            x = x + self._compute_cross_attn(x, context)
            x = x + self._apply_ff_block(x)
        return x

    def _compute_self_attn(self, x, need_weights=False):
        x, _ = self.self_attn(query=x, key=x, value=x,
            is_causal=self.causal, need_weights=need_weights)
        x = self.dropout(x)
        return x

    def _compute_cross_attn(self, x, context, need_weights=False):
        x, _ = self.cross_attn(query=x, key=context, value=context,
            is_causal=False, need_weights=need_weights)

        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x


class FeedForwardBlock(nn.Module):

    def __init__(self,
            embed_dim: int,
            dff: int = None,
            activation: str = 'relu',
            use_bias: bool = False):
        """
        Feed-forward block.

        A 2-layer neural network with activation function in between.

        Parameters
        ----------
        embed_dim : int
            embedding dimension of input.
        dff : int, optional
            size of intermediate layer. if None, 4 * embed_dim.
        activation : str, optional
            name of activation function, by default 'relu'
        use_bias : bool, optional
            whether to use bias in linear layers, by default False
        """

        super().__init__()
        self.embed_dim = embed_dim

        # set dff according to activation function if not given
        if dff is None and activation == 'swiglu':
            self.dff = int(2/3 * 4 * embed_dim)
        elif dff is None:
            self.dff = 4 * embed_dim
        else:
            self.dff = dff

        self.use_bias = use_bias
        self.activation = activation
        if self.activation != 'swiglu':
            self.activation_ = get_activation_function(activation)

        self.linear1 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)
        self.linear2 = nn.Linear(self.dff, self.embed_dim, bias=self.use_bias)
        if self.activation == 'swiglu':
            self.linear3 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)

    def forward(self, x):
        if self.activation == 'swiglu':
            return self.linear2(nn.functional.silu(self.linear1(x)) * self.linear3(x))
        else:
            x = self.linear1(x)
            x = self.activation_(x)
            x = self.linear2(x)
            return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def create_norm(d_model, norm_type):
    if norm_type=='layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type=='rmsnorm':
        return RMSNorm(d_model)
    elif norm_type=='none':
        return  nn.Identity()
    else:
        raise ValueError(f'norm_type {norm_type} not valid')

def get_activation_function(name):
    """gets activation function by its name."""

    activation_dict = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),

        # Gaussian Error Linear Unit: GELU(x) = x * GaussianCDF(x)
        'gelu': nn.GELU(approximate='tanh'),

        # Sigmoid Linear Unit: silu(x) = x * sigmoid(x)
        'silu': nn.SiLU(),

        # Softmax of Linear Units: SoLU(x) = x * softmax(x)
        # (https://transformer-circuits.pub/2022/solu/index.html)
        'solu': lambda x: x * torch.nn.functional.softmax(x, dim=-1),

        # LayerNormed Softmax of Linear Units: LNSoLU(x) = LN(x * softmax(x))
        # Note: here, I am using layernorm functional with no learnable parameters;
        # Note: interpretable activations would be pre-norm post-softmax (i.e., the SoLU part)
        # NOTE: this is equivalent to the original SoLU(x) = LN(x * exp(x)) in terms of final model performance due to scale-invariance of LayerNorm
        'lnsolu': lambda x: torch.nn.functional.layer_norm(x * torch.nn.functional.softmax(x, dim=-1)),

        'softmax': nn.Softmax(dim=-1),
        'identity': nn.Identity(),
        # add more if needed
    }
    if name in activation_dict:
        return activation_dict[name]
    else:
        raise ValueError(f'Activation function {name} not found in {activation_dict.keys()}')