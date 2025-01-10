import torch
import torch.nn as nn
import math

from .transformer_blocks import EncoderBlock
from .positional_encoding import ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding, AlibiPositionalBias, T5RelativePositionBias, RotaryPositionalEmbeddings
from .residual_stream import create_norm

class TransformerLM(torch.nn.Module):
    """
    Transformer Language Model.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary containing the following keys:
        - d_model (int): Dimension of the model.
        - n_heads (int): Number of attention heads.
        - n_layers (int): Number of Transformer layers.
        - dff (int): Dimension of the feed-forward layer.
        - mlp_activation (str): Activation function for the feed-forward layer.
        - norm_config (dict): Configuration for normalization layers.
        - vocab_size (int): Size of the vocabulary.
        - pos_enc_type (str): Type of positional encoding to use.
        - pos_enc_kwargs (dict): Additional arguments for positional encoding.
        - attn_kwargs (dict): Additional arguments for attention layers.

    Methods
    -------
    get_pos_enc_model(attn=True):
        Returns the positional encoding model based on the configuration.
    forward(x):
        Forward pass of the model.
    """

    def __init__(self, model_config):

        super(TransformerLM, self).__init__()
        self.model_config = model_config

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.dff = model_config.dff
        self.mlp_activation = getattr(model_config, 'mlp_activation', 'relu')
        self.norm_config = getattr(model_config, 'norm_config', None)
        self.vocab_size = model_config.vocab_size
        self.pos_enc_type = model_config.pos_enc_type
        self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})
        self.weight_tie_embed_to_token = getattr(model_config, 'weight_tie_embed_to_token', False) # weight tying

        # if using sinusoidal or learned positional encodings, create the positional encoding model
        self.pos_enc_model = self.get_pos_enc_model() if self.pos_enc_type in ['sinusoidal', 'learned'] else None

        self.token_embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)

        self.blocks = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, norm_config=self.norm_config,
            pos_enc_model=self.get_pos_enc_model(attn=True), # positional encoding model for attention (e.g., RoPE, T5, etc.)
            attn_kwargs=self.attn_kwargs, causal=True)
            for _ in range(model_config.n_layers)])

        # if using pre-norm, apply layernorm before final linear layer
        if model_config.norm_config.norm_method == 'pre-norm':
            self.prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm'))
        else:
            self.prelogits_norm = torch.nn.Identity()

        self.embed_to_token_logits = torch.nn.Linear(model_config.d_model, model_config.vocab_size)

        # weight tying between token_embedder and embed_to_token_logits
        if self.weight_tie_embed_to_token:
            self.embed_to_token_logits.weight = self.token_embedder.weight

        # initialize weights
        self.apply(self._init_weights)

        # NOTE: below may not be necessary for HypersphereLERP, since representation is always pinned to unit-norm
        # per-GPT2 paper, scale intialization of output projection and last layer of mlp
        # apply special n_layer-scaled initialization to layers that add to the residual stream
        # (output projection of attention and last layer of mlp)
        # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
        # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
        mlp_special_init_layer = 'linear3' if self.model_config.mlp_activation == 'swiglu' else 'linear2'
        for pn, p in self.named_parameters():
            if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p,
                    mean=0.0, std=0.02 / math.sqrt(2 * self.model_config.n_layers))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_enc_model(self, attn=True):
        if self.pos_enc_type == 'sinusoidal' and not attn:
            return ScaledSinusoidalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'learned' and not attn:
            return AbsolutePositionalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'alibi':
            return AlibiPositionalBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify slopes in pos_enc_kwargs
        elif self.pos_enc_type == 't5':
            return T5RelativePositionBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify num_buckets, max_distance in pos_enc_kwargs (default 32, 128)
        elif self.pos_enc_type == 'rotary':
            return RotaryPositionalEmbeddings(dim=self.d_head, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'none' or ((self.pos_enc_type in ['sinusoidal', 'learned']) and attn):
            return None
        else:
            raise ValueError(f"pos_enc_type {self.pos_enc_type} not recognized")

    def forward(self, x):

        # embed tokens
        x = self.token_embedder(x)

        # if positional encoding model is additive-embedding-based, add it to the input
        if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
            x += self.pos_enc_model(x)

        # apply the Transformer layers
        for encoder in self.blocks:
            x = encoder(x)

        # apply pre-logits normalization (if using pre-norm)
        x = self.prelogits_norm(x)

        # project to logits
        logits = self.embed_to_token_logits(x)

        return logits

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        N = self.get_num_params()
        L, H, Q, T = self.n_layers, self.n_heads, self.d_model//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_enc_type=='pos_emb':
            n_params -= self.layers.positional_embedder.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None):
        """
        Generate max_new_tokens new tokens, conditioning on the input idx.

        Parameters
        ----------
        idx : Tensor[int]
            tensor of shape (batch_size, seq_len) with input tokens.
        max_new_tokens : int
            number of new tokens to generate
        temperature : float, optional
            temperature parameter of softmax, by default 1.0
        top_k : int, optional
            top-k sampling parameter, by default None

        Returns
        -------
        Tensor[int]
            tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens.
        """

        for _ in range(max_new_tokens):
            # crop the sequence if it is longer thanblock_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond) # forward pass
            logits = logits[:, -1, :] / temperature # scale by temperature

            # optionally, crop logits to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence

        return idx

class RecurrentTransformerLM(torch.nn.Module):
    """
    Recurrent Transformer Language Model.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary containing the following keys:
        - d_model (int): Dimension of the model.
        - n_heads (int): Number of attention heads.
        - n_layers (int): Number of Transformer layers.
        - default_n_iters (int): Number of iterations to run the model (by default).
        - dff (int): Dimension of the feed-forward layer.
        - bias (bool): Whether to use bias in the linear layers.
        - mlp_activation (str): Activation function for the feed-forward layer.
        - norm_config (dict): Configuration for normalization layers.
        - vocab_size (int): Size of the vocabulary.
        - pos_enc_type (str): Type of positional encoding to use.
        - pos_enc_kwargs (dict): Additional arguments for positional encoding.
        - attn_kwargs (dict): Additional arguments for attention layers.

    Methods
    -------
    get_pos_enc_model(attn=True):
        Returns the positional encoding model based on the configuration.
    forward(x):
        Forward pass of the model.
    """

    def __init__(self, model_config):

        super().__init__()
        self.model_config = model_config

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.bias = model_config.bias
        self.dff = model_config.dff
        self.mlp_activation = getattr(model_config, 'mlp_activation', 'relu')
        self.norm_config = getattr(model_config, 'norm_config', None)
        self.vocab_size = model_config.vocab_size
        self.pos_enc_type = model_config.pos_enc_type
        self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})
        self.weight_tie_embed_to_token = getattr(model_config, 'weight_tie_embed_to_token', False) # weight tying
        self.default_n_iters = model_config.default_n_iters # number of iterations to run the model

        # if using sinusoidal or learned positional encodings, create the positional encoding model
        self.pos_enc_model = self.get_pos_enc_model() if self.pos_enc_type in ['sinusoidal', 'learned'] else None

        self.token_embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)

        self.blocks = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, norm_config=self.norm_config,
            pos_enc_model=self.get_pos_enc_model(attn=True), # positional encoding model for attention (e.g., RoPE, T5, etc.)
            attn_kwargs=self.attn_kwargs, bias=self.bias, causal=True)
            for _ in range(model_config.n_layers)])

        # if using pre-norm, apply layernorm before final linear layer
        if model_config.norm_config.norm_method == 'pre-norm':
            self.prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm'))
        else:
            self.prelogits_norm = torch.nn.Identity()

        self.embed_to_token_logits = torch.nn.Linear(model_config.d_model, model_config.vocab_size, bias=self.bias)

        # weight tying between token_embedder and embed_to_token_logits
        if self.weight_tie_embed_to_token:
            self.embed_to_token_logits.weight = self.token_embedder.weight

        # initialize weights
        self.apply(self._init_weights)

        # NOTE: below may not be necessary for HypersphereLERP, since representation is always pinned to unit-norm
        # per-GPT2 paper, scale intialization of output projection and last layer of mlp
        # apply special n_layer-scaled initialization to layers that add to the residual stream
        # (output projection of attention and last layer of mlp)
        # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
        # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
        mlp_special_init_layer = 'linear3' if self.model_config.mlp_activation == 'swiglu' else 'linear2'
        for pn, p in self.named_parameters():
            if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p,
                    mean=0.0, std=0.02 / math.sqrt(2 * self.model_config.n_layers))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_enc_model(self, attn=True):
        if self.pos_enc_type == 'sinusoidal' and not attn:
            return ScaledSinusoidalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'learned' and not attn:
            return AbsolutePositionalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'alibi':
            return AlibiPositionalBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify slopes in pos_enc_kwargs
        elif self.pos_enc_type == 't5':
            return T5RelativePositionBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify num_buckets, max_distance in pos_enc_kwargs (default 32, 128)
        elif self.pos_enc_type == 'rotary':
            return RotaryPositionalEmbeddings(dim=self.d_head, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'none' or ((self.pos_enc_type in ['sinusoidal', 'learned']) and attn):
            return None
        else:
            raise ValueError(f"pos_enc_type {self.pos_enc_type} not recognized")

    def forward(self, x, n_iters=None):

        n_iters = self.default_n_iters if n_iters is None else n_iters

        # embed tokens
        x = self.token_embedder(x)

        # if positional encoding model is additive-embedding-based, add it to the input
        if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
            x += self.pos_enc_model(x)

        # apply the Transformer layers
        for iter in range(n_iters):
            for encoder in self.blocks:
                x = encoder(x)

        # apply pre-logits normalization (if using pre-norm)
        x = self.prelogits_norm(x)

        # project to logits
        logits = self.embed_to_token_logits(x)

        return logits

    def forward_skip_embed(self, x, n_iters=None):
        """
        forward call that skips the embedding layer (and adding positional embedding if applicable)

        Parameters
        ----------
        x : Tensor
            input tensor of shape (batch_size, seq_len, d_model)

        Returns
        -------
        Tensor
            logits tensor of shape (batch_size, seq_len, out_vocab_size)
        """

        n_iters = self.default_n_iters if n_iters is None else n_iters

        # apply the Transformer layers
        for iter in range(n_iters):
            for encoder in self.blocks:
                x = encoder(x)

        # apply pre-logits normalization (if using pre-norm)
        x = self.prelogits_norm(x)

        # project to logits
        logits = self.embed_to_token_logits(x)

        return logits

    def forward_skip_output(self, x, n_iters=None):
        """

        Forward call that skips the output layer (i.e., norm + logits)

        Parameters
        ----------
        x : Tensor[int]
            input tensor of token idxs of shape (batch_size, seq_len,)
        n_iters : int, optional
            number of iterations to run recurrent model, by default None

        Returns
        -------
        Tensor
            tensor of shape (batch_size, seq_len, d_model) with model output
        """

        n_iters = self.default_n_iters if n_iters is None else n_iters

        # embed tokens
        x = self.token_embedder(x)

        # if positional encoding model is additive-embedding-based, add it to the input
        if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
            x += self.pos_enc_model(x)

        # apply the Transformer layers
        for iter in range(n_iters):
            for encoder in self.blocks:
                x = encoder(x)

        return x

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        N = self.get_num_params()
        L, H, Q, T = self.n_layers, self.n_heads, self.d_model//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_enc_type=='pos_emb':
            n_params -= self.layers.positional_embedder.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None):
        """
        Generate max_new_tokens new tokens, conditioning on the input idx.

        Parameters
        ----------
        idx : Tensor[int]
            tensor of shape (batch_size, seq_len) with input tokens.
        max_new_tokens : int
            number of new tokens to generate
        temperature : float, optional
            temperature parameter of softmax, by default 1.0
        top_k : int, optional
            top-k sampling parameter, by default None

        Returns
        -------
        Tensor[int]
            tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens.
        """

        for _ in range(max_new_tokens):
            # crop the sequence if it is longer thanblock_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond) # forward pass
            logits = logits[:, -1, :] / temperature # scale by temperature

            # optionally, crop logits to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence

        return idx

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    use_fused = (device_type == 'cuda')
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    print(f"using fused AdamW: {use_fused}")

    return optimizer