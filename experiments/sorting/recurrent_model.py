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
from models.misc import ConcatCombine
from models.attention_utils import topk_softmax

class RecurrentTransformerModel(torch.nn.Module):
    def __init__(self, model_config):
        super(RecurrentTransformerModel, self).__init__()
        self.model_config = model_config

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.dff = model_config.dff
        self.mlp_activation = getattr(model_config, 'mlp_activation', 'relu')
        self.vocab_size = model_config.vocab_size
        self.pos_enc_type = model_config.pos_enc_type
        self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})

        self.discrete_intermediate = getattr(getattr(model_config, 'intermediate_discretization', {}), 'discrete_intermediate', False)
        self.discretization_map_type = getattr(getattr(model_config, 'intermediate_discretization', {}), 'discretize_map', None)
        self.discretization_map_params = getattr(getattr(model_config, 'intermediate_discretization', {}), 'discretization_map_params', {})
        if self.discretization_map_type is not None:
            self.discretization_map = get_discretization_amp(self.discretization_map_type, self.discretization_map_params)
        assert self.discretization_map is not None or not self.discrete_intermediate, "Discretization map must be provided for discrete intermediate."

        self.input_recall = getattr(model_config, 'input_recall', False)
        self.input_recall_type = getattr(model_config, 'input_recall_type', 'add')
        if self.input_recall_type == 'add':
            self.input_recall_combine = lambda x, y: x + y
        elif self.input_recall_type == 'concat':
            self.input_recall_combine = ConcatCombine(dim=self.d_model)

        # FIXME: decide whether to share positional encodings across layers
        self.pos_enc_model = self.get_pos_enc_model()

        self.embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.encoder = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, activation=self.mlp_activation, pos_enc_model=self.pos_enc_model, attn_kwargs=self.attn_kwargs)
            for _ in range(model_config.n_layers)])
        self.to_token_logits = torch.nn.Linear(model_config.d_model, model_config.vocab_size)

        self.token_to_embed = torch.nn.Linear(model_config.vocab_size, model_config.d_model)

        # weight tying
        if getattr(model_config, 'weight_tying', False):
            self.to_token_logits.weight = self.embedder.weight

            if self.discrete_intermediate:
                self.token_to_embed.weight = torch.nn.Parameter(self.embedder.weight.t())

    def get_pos_enc_model(self):
        if self.pos_enc_type == 'sinusoidal':
            return ScaledSinusoidalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'learned':
            return AbsolutePositionalEmbedding(dim=self.d_model, **self.pos_enc_kwargs)
        elif self.pos_enc_type == 'alibi':
            return AlibiPositionalBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify slopes in pos_enc_kwargs
        elif self.pos_enc_type == 't5':
            return T5RelativePositionBias(heads=self.n_heads, **self.pos_enc_kwargs) # can specify num_buckets, max_distance in pos_enc_kwargs (default 32, 128)
        elif self.pos_enc_type == 'rotary':
            return RotaryPositionalEmbeddings(dim=self.d_head, **self.pos_enc_kwargs)
        else:
            return None

    def forward(self, x, n_iters=1, skip_embed=False, input_emb=None, skip_output=False):

        assert not (skip_embed and input_emb is None), "If skip_embed is True, input_emb must be provided."

        if not skip_embed:
            input_emb = self.embedder(x)
            x = input_emb

            if any(isinstance(self.pos_enc_model, model) for model in [ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding]):
                x += self.pos_enc_model(x)

        for i in range(n_iters):
            if self.input_recall:
                x = self.input_recall_combine(x, input_emb)

            x = self.compute_iteration(x)

            if self.discrete_intermediate:
                x = self.to_token_logits(x)
                x = self.discretization_map(x)

                x = self.token_to_embed(x)
                # x = torch.matmul(x, self.embedder.weight)
                # NOTE : here, I am "weight-tying" the output of the discretization map to the embedding matrix
                # TODO: make this a separate learnable map? or make weight-tying optional?
        if not skip_output:
            x = self.to_token_logits(x)
        return x

    def compute_iteration(self, x):
        for encoder in self.encoder:
            x = encoder(x)

        return x

class LitRecurrentModel(pl.LightningModule):
    def __init__(self, model_config, data_config, train_config):
        super().__init__()
        self.model = RecurrentTransformerModel(model_config)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        if train_config.compile:
            self.model = torch.compile(self.model)
            print('Model compiled.')

        self.criterion = torch.nn.CrossEntropyLoss()

        # for test-time evaluation
        self.test_step_outputs = []

    def forward(self, x, n_iters):
        return self.model(x, n_iters=n_iters)

    def sample_n_iters(self):
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
        if self.train_config.incremental_training:
            input_emb = self.model.embedder(x)
            with torch.no_grad():
                x = self.model(x, n_iters=n_nograd_iters, skip_embed=False, skip_output=True)

            logits = self.model(x, input_emb=input_emb, n_iters=n_iters, skip_embed=True, skip_output=False)

        # run model with tracking gradients for random number n_iters of iters
        else:
            logits = self.model(x, n_iters=n_iters)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        metrics = self.compute_metrics(batch, logits)

        self.log_metrics(metrics, key_dir='train')

        self.log('train/n_nograd_iters', n_nograd_iters)
        self.log('train/n_iters', n_iters)
        self.log('train/total_iters', n_nograd_iters + n_iters)

        self.log('train/seq_len', seq_len)

        return loss

    def validation_step(self, batch):

        x, y = batch

        # randomly sample n_iters for validation, as in training
        n_iters, n_nograd_iters = self.sample_n_iters()
        logits = self.model(x, n_iters=n_iters+n_nograd_iters)

        metrics = self.compute_metrics(batch, logits)
        self.log_metrics(metrics, key_dir='val')

        return metrics['loss']

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):

        x, y = batch
        n_iterss = range(1, self.train_config.test_max_n_iters+1)

        for n_iters in n_iterss:
            logits = self.model(x, n_iters=n_iters)
            metrics = self.compute_metrics(batch, logits)

            ood_length = self.data_config.ood_test_sequence_lengths[dataloader_idx]

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

def get_discretization_amp(discretization_map_type: str, kwargs: dict):

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
    model_str = f'L{model_config.n_layers}H{model_config.n_heads}D{model_config.d_model}_{model_config.pos_enc_type}_IR{model_config.input_recall}_WT{model_config.weight_tying}'

    # attn_score_fn
    if model_config.attn_kwargs.attn_score_fn != 'softmax':
        model_str += f'_{model_config.attn_kwargs.attn_score_fn}'
        if model_config.attn_kwargs.get('attn_score_fn_params', {}).get('straight_through', False):
            model_str += '-ST'

    if model_config.intermediate_discretization.discrete_intermediate:
        model_str += f'_discinterm-{model_config.intermediate_discretization.discretize_map}'
    else:
        model_str += '_discinterm-NA'

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

    return group_name, run_name
