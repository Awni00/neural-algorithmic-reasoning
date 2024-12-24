from functools import partial

import numpy as np
import wandb
import pandas as pd

import torch
import pytorch_lightning as pl
import lightning
import plotly.express as px

from datetime import datetime

from models.transformer_blocks import EncoderBlock
from models.positional_encoding import ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding, AlibiPositionalBias, T5RelativePositionBias, RotaryPositionalEmbeddings
from models.residual_stream import ConcatCombine, create_norm
from models.attention_utils import topk_softmax

# model_config: temparature annealing (TODO)
# model_config: intermediate_compute_vocab (TODO) [additional vocab for intermediate states] [how to interact with weight-tying??]
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
        - vocab_size (int): Size of the vocabulary.
        - pos_enc_type (str): Type of positional encoding to use.
        - pos_enc_kwargs (dict): Additional arguments for positional encoding.
        - attn_kwargs (dict): Additional arguments for attention layers.
        - intermediate_discretization (dict): Configuration for intermediate discretization.
        - input_recall (bool): Whether to use input recall.
        - input_recall_type (str): Type of input recall ('add' or 'concat').
        - predisc_norm (bool): Whether to apply normalization before discretization.
        - postdisc_norm (bool): Whether to apply normalization after discretization.
        - weight_tie_embed_to_token (bool): Whether to tie the embedding and token projection weights.
        - weight_tie_discrete_interm (bool): Whether to tie the weights for discrete intermediate states.

    Methods
    -------
    get_pos_enc_model(attn=True):
        Returns the positional encoding model based on the configuration.
    forward(x, n_iters=1, skip_embed=False, orig_input=None, skip_output=False, return_intermediate_states=False):
        Forward pass of the model.
    compute_iteration(x):
        Computes a single iteration of the model.
    """

    def __init__(self, model_config):

        super(RecurrentTransformerModel, self).__init__()
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

        self.discrete_intermediate = getattr(getattr(model_config, 'intermediate_discretization', {}), 'discrete_intermediate', False)
        self.discretization_map_type = getattr(getattr(model_config, 'intermediate_discretization', {}), 'discretize_map', None)
        self.discretization_map_params = getattr(getattr(model_config, 'intermediate_discretization', {}), 'discretization_map_params', {})
        if self.discretization_map_type is not None and self.discrete_intermediate:
            # discretization map for intermediate states (e.g., softmax, hardmax, gumbel-softmax, etc.)
            self.discretization_map = get_discretization_map(self.discretization_map_type, self.discretization_map_params)

            # used for mapping discrete intermediate state to embedded state (optionally tied with embedding matrix)
            self.token_to_embed = torch.nn.Linear(model_config.vocab_size, model_config.d_model)

        assert not self.discrete_intermediate or self.discretization_map is not None, "Discretization map must be provided for discrete intermediate."

        # input recall (incorporates original input into hidden state at each recurrent iteration)
        self.input_recall = getattr(model_config, 'input_recall', False)
        self.input_recall_type = getattr(model_config, 'input_recall_type', 'add')
        if self.input_recall_type == 'add':
            self.input_recall_combine = lambda x, y: x + y
        elif self.input_recall_type == 'concat':
            self.input_recall_combine = ConcatCombine(dim=self.d_model)

        # if using sinusoidal or learned positional encodings, create the positional encoding model
        self.pos_enc_model = self.get_pos_enc_model() if self.pos_enc_type in ['sinusoidal', 'learned'] else None

        self.embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)

        # each recurrent step involves running a sequence of Transformer blocks (incl. attention + MLP)
        self.encoder = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, norm_config=self.norm_config,
            pos_enc_model=self.get_pos_enc_model(attn=True), # positional encoding model for attention (e.g., RoPE, T5, etc.)
            attn_kwargs=self.attn_kwargs)
            for _ in range(model_config.n_layers)])
        self.prelogits_norm = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm'))
        self.embed_to_token_logits = torch.nn.Linear(model_config.d_model, model_config.vocab_size)

        # pre-discretization layernorm
        # note: if logits have large magnitude, gradient of softmax will be small, making training difficult
        # layernorm can help with this
        self.pre_disc_ln = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm')) if model_config.predisc_norm else torch.nn.Identity()
        self.post_dic_ln = create_norm(self.d_model, self.norm_config.get('norm_type', 'layernorm')) if model_config.postdisc_norm else torch.nn.Identity()

        # weight tying
        self.weight_tie_embed_to_token = getattr(model_config, 'weight_tie_embed_to_token', False)
        self.weight_tie_discrete_interm = getattr(model_config, 'weight_tie_discrete_interm', False)

        # weight tying between embedder and embed_to_token_logits (mapping embeddings to discrete tokens, for now both in final prediction and intermediate discrete states, if applicable)
        if self.weight_tie_embed_to_token:
            self.embed_to_token_logits.weight = self.embedder.weight

        # weight tying between token_to_embed (mapping discrete tokens to embeddings) and the models embedding layer
        if self.discrete_intermediate and self.weight_tie_discrete_interm:
            self.token_to_embed.weight = torch.nn.Parameter(self.embedder.weight.t())
            # NOTE: for now, we always use embed_to_token_logits to map embedding to discrete intermediate state tokens, regardless of weight tying configuration (i.e., will be weight-tied if weight_tie_embed_to_token is True)
            # self.token_to_embed maps the discrete intermediate state tokens back to the embedding space; whether this is weight-tied with the embedding matrix is determined by weight_tie_discrete_interm
            # TODO: think about this more carefully... e.g., do we want more fine-grained control over weight tying?

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
        else:
            return None

    def forward(self, x, n_iters=1, skip_embed=False, orig_input=None, skip_output=False, return_intermediate_states=False):

        assert not (skip_embed and orig_input is None), "If skip_embed is True, input_emb must be provided."

        if not skip_embed:
            input_emb = self.embedder(x)
            x = input_emb

            if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
                x += self.pos_enc_model(x)
        else:
            input_emb = self.embedder(orig_input)

        if return_intermediate_states:
            intermediate_states = dict(disc_interm_states=[], logits_states=[], emb_norms=[], logits_states_softmax_entropy=[], delta_norms=[])

        for iter in range(n_iters):
            x_prev = x

            if self.input_recall:
                x = self.input_recall_combine(x, input_emb)

            x = self.compute_iteration(x)

            if return_intermediate_states:
                intermediate_states['delta_norms'].append((x - x_prev).norm(dim=-1))
                intermediate_states['emb_norms'].append(x.norm(dim=-1))

            if self.discrete_intermediate:
                x = self.pre_disc_ln(x) # normalize before discretization

                x = self.embed_to_token_logits(x)
                if return_intermediate_states:
                    intermediate_states['logits_states'].append(x)
                    # logits_states_softmax_entropy = (-x.float().softmax(dim=-1) * x.float().softmax(dim=-1).log2()).sum(dim=-1)
                    logits_states_softmax_entropy = torch.special.entr(x.softmax(dim=-1)).sum(dim=-1)
                    # note: torch.speecial.entr computes -x * ln(x) elementwise
                    intermediate_states['logits_states_softmax_entropy'].append(logits_states_softmax_entropy)

                x = self.discretization_map(x)
                if return_intermediate_states:
                    intermediate_states['disc_interm_states'].append(x)

                x = self.token_to_embed(x)

                x = self.post_dic_ln(x) # normalize after discretization

        if not skip_output:
            x = self.prelogits_norm(x)
            x = self.embed_to_token_logits(x)

        if return_intermediate_states:
            return x, intermediate_states

        return x

    def compute_iteration(self, x):
        for encoder in self.encoder:
            x = encoder(x)

        return x

# TODO: add some documentation
# TODO: train config: soft_teacherforcing (i.e., add small loss on intermediate states to encourage teacher-forcing: intuition, soft "teacher-forcing", but with arbitrary data-driven order rather than causal order)
# TODO: make sure implementation of every step is correct... there are several places to make errors in implementation that could mess this up...
# TODO: add more clever data sampling methods (e.g., weight proportional loss for each sequence length/# iters... does multiplicative weights make sense?)
class LitRecurrentModel(pl.LightningModule):
    """
    Lightning Module for training and evaluating the RecurrentTransformerModel.

    This class handles the training, validation, and testing of the Recurrent Transformer Model using PyTorch Lightning.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary for the model.
    data_config : dict
        Configuration dictionary for the data.
    train_config : dict
        Configuration dictionary for the training process.

    Methods
    -------
    forward(x, n_iters):
        Forward pass of the model.
    sample_n_iters():
        Samples the number of iterations for training.
    training_step(batch):
        Defines the training step.
    validation_step(batch):
        Defines the validation step.
    test_step(batch, batch_idx=0, dataloader_idx=0):
        Defines the test step.
    on_test_epoch_end():
        Called at the end of the test epoch to log metrics.
    compute_metrics(batch, logits):
        Computes the metrics for a given batch and logits.
    compute_intermediate_state_metrics(intermediate_states):
        Computes metrics for intermediate states.
    log_metrics(metrics, prefix=None, key_dir=None, **log_kwargs):
        Logs the metrics.
    configure_optimizers():
        Configures the optimizers and learning rate schedulers.
    lr_scheduler_step(scheduler, metric):
        Steps the learning rate scheduler.
    on_before_optimizer_step(optimizer):
        Called before the optimizer step to log gradient norms.
    create_and_log_figs(test_df):
        Creates and logs figures for test metrics.
    """

    def __init__(self, model_config, data_config, train_config):
        super().__init__()
        self.model = RecurrentTransformerModel(model_config)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        self.save_hyperparameters() # save model hyperparameters

        if train_config.compile:
            self.model = torch.compile(self.model)
            print('Model compiled.')

        self.criterion = torch.nn.CrossEntropyLoss()

        # for test-time evaluation
        self.test_step_outputs = []

    def forward(self, x, n_iters):
        return self.model(x, n_iters=n_iters)

    def sample_n_iters(self):
        if np.random.random() < 1 - self.train_config.progressive_training_prob:
            return (self.train_config.train_max_n_iters, 0)

        # randomly sample number of iterations to run model without tracking gradients
        if self.train_config.incremental_training:
            n_nograd_iters = np.random.randint(0, self.train_config.train_max_n_iters)
        else:
            n_nograd_iters = 0

        # number of iterations to run model with tracking gradients
        if self.train_config.progressive_training:
            n_iters = np.random.randint(1, self.train_config.train_max_n_iters - n_nograd_iters + 1)
        else:
            n_iters = self.train_config.train_max_n_iters

        return n_iters, n_nograd_iters

    def training_step(self, batch):

        x, y = batch
        seq_len = x.size(1)

        # algorithm:
        # n_nograd_iters ~ {0, ..., train_max_n_iters - 1}
        # n_train_iters ~ {1, ..., train_max_n_iters - n_nograd_iters}
        # this guarantees that n_train_iters + n_nograd_iters <= train_max_n_iters

        n_iters, n_nograd_iters = self.sample_n_iters()

        # run model without tracking gradients for n_nograd_iters
        # then run model with tracking gradients for n_iters
        if self.train_config.incremental_training and n_nograd_iters > 0:
            orig_input = x
            with torch.no_grad():
                x = self.model(x, n_iters=n_nograd_iters, skip_embed=False, skip_output=True)

            logits = self.model(x, orig_input=orig_input, n_iters=n_iters, skip_embed=True, skip_output=False)

        # run model with tracking gradients for random number n_iters of iters
        else:
            logits = self.model(x, n_iters=n_iters)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        metrics = self.compute_metrics(batch, logits)

        self.log_metrics(metrics, key_dir='train', prog_bar=True)

        self.log('train/n_nograd_iters', n_nograd_iters)
        self.log('train/n_iters', n_iters)
        self.log('train/total_iters', n_nograd_iters + n_iters)

        self.log('train/seq_len', seq_len)

        return loss

    def validation_step(self, batch):

        x, y = batch

        # randomly sample n_iters for validation, as in training
        n_iters, n_nograd_iters = self.sample_n_iters()
        logits, intermediate_states = self.model(x, n_iters=n_iters+n_nograd_iters, return_intermediate_states=True)

        metrics = self.compute_metrics(batch, logits)
        self.log_metrics(metrics, key_dir='val', prog_bar=True)

        intermediate_states_metrics = self.compute_intermediate_state_metrics(intermediate_states)
        self.log_metrics(intermediate_states_metrics, key_dir='val_interm', prog_bar=False)

        return metrics['loss']

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):

        x, y = batch

        ood_length = self.data_config.ood_test_sequence_lengths[dataloader_idx] # length of out-of-distribution sequence

        # first, run model for max number of iterations, tracking intermediate states (to log entropy of logits, embedding norms, etc.)
        logits, intermediate_states = self.model(x, n_iters=self.train_config.test_max_n_iters, return_intermediate_states=True)
        # intermediate_states: Dict[str, List[torch.Tensor]] with keys: disc_interm_states, logits_states, emb_norms. List length = n_iters
        assert len(intermediate_states['emb_norms']) == self.train_config.test_max_n_iters, f"Expected {self.train_config.test_max_n_iters} intermediate states, got {len(intermediate_states['emb_norms'])}"
        if self.model.discrete_intermediate:
            assert len(intermediate_states['logits_states_softmax_entropy']) == self.train_config.test_max_n_iters, f"Expected {self.train_config.test_max_n_iters} intermediate states, got {len(intermediate_states['logits_states_softmax_entropy'])}"

        # for each number of iterations, compute metrics
        n_iterss = range(1, self.train_config.test_max_n_iters+1)
        for n_iters in n_iterss:
            logits = self.model(x, n_iters=n_iters)
            metrics = self.compute_metrics(batch, logits)

            # average embedding norm over batch, sequence length
            metrics['emb_norms'] = intermediate_states['emb_norms'][n_iters - 1].mean()
            metrics['delta_norms'] = intermediate_states['delta_norms'][n_iters - 1].mean()
            if self.model.discrete_intermediate:
                # average entropy of discrete state logits over batch, sequence length
                metrics['logits_states_softmax_entropy'] = intermediate_states['logits_states_softmax_entropy'][n_iters - 1].mean()

            self.test_step_outputs.append(dict(ood_length=ood_length, n_iters=n_iters, metrics=metrics))

        # self.log_metrics(metrics, key_dir='test', prefix=f'L={ood_length}', add_dataloader_idx=False)

        return metrics['loss']

    def on_test_epoch_end(self):

        metrics_list = []
        for L in self.data_config.ood_test_sequence_lengths:
            for n_iters in range(1, self.train_config.test_max_n_iters+1):
                metrics = [output['metrics'] for output in self.test_step_outputs if output['ood_length'] == L and output['n_iters'] == n_iters]
                metrics = {key: torch.tensor([m[key].item() for m in metrics]).mean().tolist() for key in metrics[0].keys()}
                metrics['n_iters'] = n_iters
                metrics['L'] = L

                metrics_list.append(metrics)

        # plot line plots of metrics (metric vs L)
        test_df = pd.DataFrame(metrics_list)
        self.logger.log_table('test/ood_eval', dataframe=test_df)

        self.create_and_log_figs(test_df)

        self.test_step_outputs.clear()  # free memory

    def compute_metrics(self, batch, logits):
        x, y = batch
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        # compute accuracy
        _, predicted = torch.max(logits, -1)
        per_token_acc = (predicted == y).float().mean()
        sequence_acc = (predicted == y).all(dim=1).float().mean()

        metrics = dict(loss=loss, per_token_acc=per_token_acc, sequence_acc=sequence_acc)

        return metrics

    def compute_intermediate_state_metrics(self, intermediate_states):

        intermediate_states_metrics = dict()
        # compute avg entropy over batch, sequence length
        intermediate_states_metrics['avg_emb_norms'] = torch.stack(intermediate_states['emb_norms']).mean()
        intermediate_states_metrics['avg_delta_norms'] = torch.stack(intermediate_states['delta_norms']).mean()
        if self.model.discrete_intermediate:
            intermediate_states_metrics['avg_logits_states_softmax_entropy'] = torch.stack(intermediate_states['logits_states_softmax_entropy']).mean()

        return intermediate_states_metrics


    def log_metrics(self, metrics, prefix=None, key_dir=None, **log_kwargs):
        for key, value in metrics.items():
            key_prefix = ''
            if key_dir is not None:
                key_prefix += f'{key_dir}/'
            if prefix is not None:
                key_prefix += f'{prefix}_'

            key = f'{key_prefix}{key}'
            self.log(key, value, **log_kwargs)

    def configure_optimizers(self):
        # Configure the optimizer.
        optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }

        optimizer_name = self.train_config.optimizer
        if optimizer_name not in optimizer_dict.keys():
            raise ValueError(f"Optimizer {optimizer_name} is not implemented!")
        else:
            optimizer = optimizer_dict[optimizer_name](
                self.parameters(),
                **self.train_config[f'{optimizer_name}_optimizer_config']
            )

        # Configure the learning rate scheduler.
        if self.train_config.lr_scheduler == "cosine":
            cosine_scheduler_config = self.train_config.cosine_scheduler_config
            # scheduler = CosineAnnealingWarmup(
            #     optimizer=optimizer,
            #     warmup_steps=cosine_scheduler_config.warmup_steps,
            #     learning_rate=self.train_config.learning_rate,
            #     min_lr=cosine_scheduler_config.min_lr,
            #     lr_decay_steps=cosine_scheduler_config.lr_decay_steps,
            # )
            raise NotImplementedError("CosineAnnealingWarmup is not implemented yet!")
        # TODO: find implementation of CosineAnnealing with Linear Warmup, or implement and add to utils module
        # e.g. look at torch tune's https://github.com/pytorch/torchtune/blob/32e265d5749fd592711a03247486eafa6c898d94/torchtune/training/lr_schedulers.py#L15
        elif self.train_config.lr_scheduler == "step":
            StepLR_config = self.train_config.StepLR_scheduler_config
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=StepLR_config.step_size,
                gamma=StepLR_config.gamma,
            )
        else:
            # use no scheduler
            scheduler = None
        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

    def lr_scheduler_step(
            self,
            scheduler,
            metric,
    ) -> None:
        scheduler.step()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        """
        This function is called before the optimizer step.
        You can override this function to do something before the optimizer step.

        Args:
            optimizer (torch.optim.Optimizer): the optimizer
        """
        norms = lightning.pytorch.utilities.grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def create_and_log_figs(self, test_df):

        # n_iters vs sequence_acc, color = L
        fig = px.line(test_df, x='n_iters', y='sequence_acc', color='L', title='Sequence Accuracy', labels={'sequence_acc': 'Sequence Accuracy', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        wandb.log({'test/ood_eval/sequence_acc': wandb.Plotly(fig)})

        # n_iters vs per_token_acc, color = L
        fig = px.line(test_df, x='n_iters', y='per_token_acc', color='L', title='Token-wise Accuracy', labels={'per_token_acc': 'Token-wise Accuracy', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        wandb.log({'test/ood_eval/per_token_acc': wandb.Plotly(fig)})

        # heatmap of n_iters vs L vs sequence_acc
        heatmap = test_df.pivot(index='n_iters', columns='L')['sequence_acc']
        fig = px.imshow(heatmap, x=heatmap.columns, y=heatmap.index, title='Sequence Accuracy', zmin=0, zmax=1, origin='lower', color_continuous_scale='Hot')
        fig.add_hline(y=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='bottom left')
        fig.add_vline(x=self.data_config.train_sequence_length, line_dash='dash', line_color='black', annotation_text='train_max_seq_len', annotation_position='bottom right')
        wandb.log({'test/ood_eval/sequence_acc_heatmap': wandb.Plotly(fig)})

        # heatmap of n_iters vs L vs per_token_acc
        heatmap = test_df.pivot(index='n_iters', columns='L')['per_token_acc']
        fig = px.imshow(heatmap, x=heatmap.columns, y=heatmap.index, title='Token-wise Accuracy', zmin=0, zmax=1, origin='lower', color_continuous_scale='Hot')
        fig.add_hline(y=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='bottom left')
        fig.add_vline(x=self.data_config.train_sequence_length, line_dash='dash', line_color='black', annotation_text='train_max_seq_len', annotation_position='bottom right')
        wandb.log({'test/ood_eval/per_token_acc_heatmap': wandb.Plotly(fig)})

        # heatmap of n_iters vs L vs emb_norms
        heatmap = test_df.pivot(index='n_iters', columns='L')['emb_norms']
        fig = px.imshow(heatmap, x=heatmap.columns, y=heatmap.index, title='Embedding Norms', origin='lower', color_continuous_scale='Hot')
        fig.add_hline(y=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='bottom left')
        fig.add_vline(x=self.data_config.train_sequence_length, line_dash='dash', line_color='black', annotation_text='train_max_seq_len', annotation_position='bottom right')
        wandb.log({'test/ood_eval/emb_norms_heatmap': wandb.Plotly(fig)})

        # heatmap of n_iters vs L vs delta_norms
        heatmap = test_df.pivot(index='n_iters', columns='L')['delta_norms']
        fig = px.imshow(heatmap, x=heatmap.columns, y=heatmap.index, title='Delta Norms', origin='lower', color_continuous_scale='Hot')
        fig.add_hline(y=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='bottom left')
        fig.add_vline(x=self.data_config.train_sequence_length, line_dash='dash', line_color='black', annotation_text='train_max_seq_len', annotation_position='bottom right')
        wandb.log({'test/ood_eval/delta_norms_heatmap': wandb.Plotly(fig)})

        # heatmap of n_iters vs L vs logits_states_softmax_entropy
        if self.model.discrete_intermediate:
            heatmap = test_df.pivot(index='n_iters', columns='L')['logits_states_softmax_entropy']
            fig = px.imshow(heatmap, x=heatmap.columns, y=heatmap.index, title='Logits States Softmax Entropy', origin='lower', color_continuous_scale='Hot')
            fig.add_hline(y=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='bottom left')
            fig.add_vline(x=self.data_config.train_sequence_length, line_dash='dash', line_color='black', annotation_text='train_max_seq_len', annotation_position='bottom right')
            wandb.log({'test/ood_eval/logits_states_softmax_entropy_heatmap': wandb.Plotly(fig)})


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

def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Model Config - Train Config - Data Config
    # Name: Seed + Date-Time
    data_str = f'MaxVal{data_config.max_value}-TrainLen{data_config.train_sequence_length}'
    if data_config.get('train_random_sequence_length', False):
        data_str += f'RandLen'
    if data_config.get('include_bos_eos', False):
        data_str += '-BOSEOS'
    model_str = f'L{model_config.n_layers}H{model_config.n_heads}D{model_config.d_model}_{model_config.pos_enc_type}_IR{model_config.input_recall}_WT{model_config.weight_tie_embed_to_token}-{model_config.weight_tie_discrete_interm}'

    # attn_score_fn
    if model_config.attn_kwargs.attn_score_fn != 'softmax':
        model_str += f'_{model_config.attn_kwargs.attn_score_fn}'
        if model_config.attn_kwargs.get('attn_score_fn_params', {}).get('straight_through', False):
            model_str += '-ST'

    if model_config.intermediate_discretization.discrete_intermediate:
        model_str += f'_discinterm-{model_config.intermediate_discretization.discretize_map}'
    else:
        model_str += '_discinterm-NA'
    if model_config.predisc_norm or model_config.postdisc_norm:
        model_str += f'_prepostdiscnorm-{model_config.predisc_norm}-{model_config.postdisc_norm}'

    # train config (progressive training and/or incremental training)
    train_str = ''
    if train_config.progressive_training:
        train_str += 'progressive'
    if train_config.incremental_training:
        train_str += '_incremental'

    group_name = f'{model_str} - {train_str} - {data_str}'

    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if getattr(train_config, 'seed', None) is not None:
        run_name = 'seed-' + str(train_config.seed) + ' - ' + run_name

    # if exceeds 128 characters, save hash instead
    if len(group_name) > 128:
        group_name = 'HASH-' + str(hash(group_name))

    if len(run_name) > 128:
        run_name = 'HASH-' + str(hash(run_name))

    return group_name, run_name
