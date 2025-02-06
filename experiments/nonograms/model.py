import torch
from torch import nn
from functools import partial

from models.transformer_blocks import EncoderBlock
# from models.positional_encoding import ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding, AlibiPositionalBias, T5RelativePositionBias, RotaryPositionalEmbeddings
from models.residual_stream import ConcatCombine, create_norm
from models.attention_utils import topk_softmax
# from models.ponder import PonderNetworkWrapper
from einops.layers.torch import Rearrange

from utils.utils import AttributeDict, get_cosine_schedule_with_warmup

from nonogram_modules import PositionalEmbedding2D, ConstraintEmbedder

class RecurrentTransformerModel(torch.nn.Module):
    """
    Recurrent Transformer Model.

    This model consists of a stack of Transformer encoder blocks, which are applied recurrently to the input sequence.
    The model can optionally use discrete intermediate states, input recall, and various types of positional encodings.

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
            - norm_method (str): Type of normalization layer ('pre-norm', 'post-norm', etc.).
            - norm_type (str): Type of normalization ('layernorm', 'rmsnorm', etc.).
        - vocab_size (int): Size of the vocabulary.
        - intermediate_vocab_size (int): Size of the vocabulary for intermediate states (if different from final vocab size).
        - pos_enc_type (str): Type of positional encoding to use.
        - pos_enc_kwargs (dict): Additional arguments for positional encoding.
        - attn_kwargs (dict): Additional arguments for attention layers.
        - intermediate_discretization (dict): Configuration for intermediate discretization.
            - discrete_intermediate (bool): Whether to use discrete intermediate states.
            - discretize_map (str): Type of discretization map to use.
            - discretization_map_params (dict): Additional arguments for discretization map.
            - weight_tie_method (str): Method for weight tying between discrete intermediate and final logits.
        - input_recall (bool): Whether to use input recall.
        - input_recall_type (str): Type of input recall ('add' or 'concat').

    """

    def __init__(self, model_config):

        super(RecurrentTransformerModel, self).__init__()
        ## set model config and hyperparams
        self.model_config = model_config

        # base model hyperparameters
        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.dff = model_config.dff
        self.mlp_activation = getattr(model_config, 'mlp_activation', 'relu')
        self.norm_config = getattr(model_config, 'norm_config', None)
        self.out_vocab_size = model_config.vocab_size
        self.intermediate_vocab_size = model_config.get('intermediate_vocab_size', model_config.vocab_size)
        self.default_n_iters = model_config.default_n_iters
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})
        # self.pos_enc_type = model_config.pos_enc_type # NOTE: for now, using fixed positional encoding method: PositionalEmbedding2D
        # self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})

        # intermediate discretization params
        self.discrete_intermediate = model_config.get('intermediate_discretization', {}).get('discrete_intermediate', False)
        self.discretization_map_type = model_config.get('intermediate_discretization', {}).get('discretize_map', None)
        self.discretization_map_params = model_config.get('intermediate_discretization', {}).get('discretization_map_params', {})
        self.weight_tie_method = model_config.get('intermediate_discretization', {}).get('weight_tie_method', 'None')


        ## create modules and layers

        # embedding layer for input tokens (i.e., constraints)
        self.constraint_embedder = ConstraintEmbedder(model_config.constraint_max_val, model_config.max_constraints_per_cell, self.d_model)

        # if using sinusoidal or learned positional encodings, create the positional encoding model
        self.pos_enc_model = PositionalEmbedding2D(model_config.max_x_pos, model_config.max_y_pos, self.d_model)

        # input recall (incorporates original input into hidden state at each recurrent iteration)
        self.input_recall = getattr(model_config, 'input_recall', False)
        self.input_recall_type = getattr(model_config, 'input_recall_type', 'add')
        if self.input_recall_type == 'add':
            self.input_recall_combine = lambda x, y: x + y
        elif self.input_recall_type == 'concat':
            self.input_recall_combine = ConcatCombine(dim=self.d_model)

        # discrete intermediate states (predicts intermediate states as discrete tokens)
        if self.discretization_map_type is not None and self.discrete_intermediate:
            # discretization map for intermediate states (e.g., softmax, hardmax, gumbel-softmax, etc.)
            self.discretization_map = get_discretization_map(self.discretization_map_type, self.discretization_map_params)

            # used for mapping embedded state to discretization logits space (optionally tied with final linear layer)
            emb_to_disc_logits_linear = torch.nn.Linear(self.d_model, self.intermediate_vocab_size)
            emb_to_disc_prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm')) if self.norm_config['norm_method'] == 'pre-norm' else torch.nn.Identity()
            self.emb_to_disc_logits = torch.nn.Sequential(
                emb_to_disc_prelogits_norm,
                emb_to_disc_logits_linear
            )
            # TODO: need pre-norm for this layer? (if using pre-norm, should apply normalization before this layer)

            # used for mapping discrete intermediate state to embedded state (optionally tied with embedding matrix)
            self.disc_to_embed = torch.nn.Linear(self.intermediate_vocab_size, self.d_model)

            # TODO: add support for intermediate_vocab_size > vocab_size
            # how to do it? use max(emb_dim, vocab_size) for token_to_embed, and then project to vocab_size?

        assert not self.discrete_intermediate or self.discretization_map is not None, "Discretization map must be provided for discrete intermediate."

        # each recurrent step involves running a sequence of Transformer blocks (incl. attention + MLP)
        self.encoder = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, norm_config=self.norm_config,
            pos_enc_model=None, # for now, using 2D PositionalEmbeddings (added once at beginning of forward pass) TODO: consider alternatives
            attn_kwargs=self.attn_kwargs)
            for _ in range(model_config.n_layers)])

        # final linear layer to project from model dimension to vocab size
        if self.norm_config['norm_method'] == 'pre-norm':
            # if using pre-norm, we should apply normalization before the final linear layer
            prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm'))
        else:
            prelogits_norm = torch.nn.Identity()

        to_logits_linear = torch.nn.Linear(model_config.d_model, model_config.vocab_size)

        # if want to weight-tie and have different vocab sizes for intermediate and final logits, pad vocab of final logits
        if self.weight_tie_method in ['to_logits', 'all'] and self.out_vocab_size < self.intermediate_vocab_size:
            to_logits_linear = torch.nn.Linear(model_config.d_model, model_config.intermediate_vocab_size)

        self.embed_to_token_logits = torch.nn.Sequential(
            prelogits_norm,
            to_logits_linear
        )

        # weight tying between token_to_embed (mapping discrete tokens to embeddings: Vocab -> Dim) and to_logits_linear (final linear layer: Dim -> Vocab)
        if self.discrete_intermediate:
            # note: we don't weight-tie constraint_embedder, since it has a different vocab in the nonograms task

            if self.weight_tie_method == 'none':
                pass
            elif self.weight_tie_method == 'to_logits':
                self.embed_to_token_logits[0] = self.emb_to_disc_logits[0]
                self.embed_to_token_logits[1] = self.emb_to_disc_logits[1]
                # note: if number of discrete intermediate states is larger, this will increase the number of classes in the final layer (some of which will be unused)
            elif self.weight_tie_method == 'disc-in-out':
                self.emb_to_disc_logits.weight = torch.nn.Parameter(self.disc_to_embed.weight.t())
            elif self.weight_tie_method == 'all':
                self.emb_to_disc_logits.weight = to_logits_linear.weight
                self.emb_to_disc_logits.weight = torch.nn.Parameter(self.disc_to_embed.weight.t())
            else:
                raise ValueError(f"Invalid weight tie method {self.weight_tie_method}.")

        # rearrangers
        self.seq_to_grid = Rearrange('b (x y) d -> b x y d', x=model_config.max_x_pos, y=model_config.max_y_pos)
        self.grid_to_seq = Rearrange('b x y d -> b (x y) d')

        # TODO: special initialization? e.g., GPT-2 initialization or glorot_normal_, etc.
        # by default will be glorot_uniform_ (default for torch.nn.Linear)


    def forward(self, x, n_iters=None, return_intermediate_states=False):

        if n_iters is None:
            n_iters = self.default_n_iters

        input_emb = self.constraint_embedder(x) # shape: [batch_size, x, y, dim]
        x = input_emb # shape: [batch_size, x, y, dim]
        input_emb = self.grid_to_seq(input_emb) # flatten x and y dimensions for use in input recall

        xy_pos_emb = self.pos_enc_model(x) # shape: [batch_size, x, y, dim]
        x += xy_pos_emb # shape: [batch_size, x, y, dim]

        # flatten x and y dimensions
        x = self.grid_to_seq(x) # shape: [batch_size, x * y, dim]

        if return_intermediate_states:
            intermediate_states = dict(logits_states=[], emb_norms=[], delta_norms=[])
            if self.discrete_intermediate:
                for key in ['disc_logits', 'disc_logits_softmax_entropy', 'disc_interm_states']:
                    intermediate_states[key] = []

        for iter in range(n_iters):
            last = iter == n_iters - 1

            x_prev = x

            if self.input_recall:
                x = self.input_recall_combine(x, input_emb)

            x = self.compute_iteration(x)

            if return_intermediate_states:
                intermediate_states['delta_norms'].append((x - x_prev).norm(dim=-1))
                intermediate_states['emb_norms'].append(x.norm(dim=-1))

                iter_logits = self.embed_to_token_logits(x) # shape: [batch_size, x * y, vocab_size]
                iter_logits = self.seq_to_grid(iter_logits) # shape: [batch_size, x, y, vocab_size]

                intermediate_states['logits_states'].append(iter_logits)

            if self.discrete_intermediate and not last: # don't discretize on last iteration
                x = self.emb_to_disc_logits(x) # project to logits space (predict discrete tokens)

                if return_intermediate_states:
                    disc_logits = self.seq_to_grid(x) # shape: [batch_size, x, y, vocab_size]
                    intermediate_states['disc_logits'].append(disc_logits)
                    # logits_states_softmax_entropy = (-x.float().softmax(dim=-1) * x.float().softmax(dim=-1).log2()).sum(dim=-1)
                    logits_states_softmax_entropy = torch.special.entr(disc_logits.softmax(dim=-1)).sum(dim=-1)
                    # note: torch.speecial.entr computes -x * ln(x) elementwise
                    intermediate_states['disc_logits_softmax_entropy'].append(logits_states_softmax_entropy)

                x = self.discretization_map(x)

                if return_intermediate_states:
                    intermediate_states['disc_interm_states'].append(self.seq_to_grid(x))

                x = self.disc_to_embed(x)

        # final predictions
        x = self.embed_to_token_logits(x)
        x = self.seq_to_grid(x)

        if return_intermediate_states:
            return x, intermediate_states

        return x

    def forward_skip_embed(self, x, orig_input, n_iters=1):
        '''
        forward call that skips embedding the input pre-first iteration, and instead continues from the embeddings produced by forward_skip_output.
        used for implementing incremental training procedure.
        '''
        if n_iters is None:
            n_iters = self.default_n_iters

        input_emb = self.constraint_embedder(orig_input) # shape: [batch_size, x, y, dim]
        input_emb = self.grid_to_seq(input_emb) # flatten x and y dimensions for use in input recall

        for iter in range(n_iters):
            last = iter == n_iters - 1

            if self.input_recall:
                x = self.input_recall_combine(x, input_emb)

            x = self.compute_iteration(x)

            if self.discrete_intermediate and not last: # don't discretize on last iteration
                x = self.emb_to_disc_logits(x) # project to logits space (predict discrete tokens)

                x = self.discretization_map(x)

                x = self.disc_to_embed(x)

        # final predictions
        x = self.embed_to_token_logits(x)
        x = self.seq_to_grid(x)

        return x


    def forward_skip_output(self, x, n_iters=1):
        '''
        forward call that skips the final output prediction, ending mid-way through.
        used for implementing incremental training procedure.
        '''

        if n_iters is None:
            n_iters = self.default_n_iters

        input_emb = self.constraint_embedder(x) # shape: [batch_size, x, y, dim]
        x = input_emb # shape: [batch_size, x, y, dim]
        input_emb = self.grid_to_seq(input_emb) # flatten x and y dimensions for use in input recall

        xy_pos_emb = self.pos_enc_model(x) # shape: [batch_size, x, y, dim]
        x += xy_pos_emb # shape: [batch_size, x, y, dim]

        # flatten x and y dimensions
        x = self.grid_to_seq(x) # shape: [batch_size, x * y, dim]

        for iter in range(n_iters):

            if self.input_recall:
                x = self.input_recall_combine(x, input_emb)

            x = self.compute_iteration(x)

            # note: unlike forward, we still discretize on iter == n_iters - 1 because this is not actually the last iteration
            if self.discrete_intermediate:
                x = self.emb_to_disc_logits(x) # project to logits space (predict discrete tokens)


                x = self.discretization_map(x)

                x = self.disc_to_embed(x)

        # skip final prediction and return [batch_size, x * y, dim] tensor

        return x


    def compute_iteration(self, x):
        for encoder in self.encoder:
            x = encoder(x)

        return x

# helper functions

def get_discretization_map(discretization_map_type: str, kwargs: dict):

    if discretization_map_type == "gumbel-softmax":
        return partial(torch.nn.functional.gumbel_softmax, **kwargs)
    if discretization_map_type == "softmax":
        return partial(torch.nn.functional.softmax, dim=-1)
    elif discretization_map_type == "topk-softmax":
        return partial(topk_softmax, **kwargs)
    elif discretization_map_type == "hard":
        return partial(topk_softmax, k=1, **kwargs)
    elif discretization_map_type == "sigmoid":
        return torch.nn.functional.sigmoid
    elif discretization_map_type == "solu":
        return lambda x: x * torch.nn.functional.softmax(x, dim=-1)
    elif discretization_map_type == "relu":
        return torch.nn.functional.relu
    elif discretization_map_type == "linear" or discretization_map_type is None:
        return lambda x: x
    else:
        raise ValueError(f"Discretization map type {discretization_map_type} not valid.")