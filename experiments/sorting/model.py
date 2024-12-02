import torch
import pytorch_lightning as pl
import lightning

from datetime import datetime

from models.transformer_blocks import EncoderBlock
from models.positional_encoding import ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding, AlibiPositionalBias, T5RelativePositionBias, RotaryPositionalEmbeddings


class TransformerModel(torch.nn.Module):
    def __init__(self, model_config):
        super(TransformerModel, self).__init__()
        self.model_config = model_config

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.n_layers = model_config.n_layers
        self.dff = model_config.dff
        self.vocab_size = model_config.vocab_size
        self.pos_enc_type = model_config.pos_enc_type
        self.pos_enc_kwargs = getattr(model_config, 'pos_enc_kwargs', {})
        self.attn_kwargs = getattr(model_config, 'attn_kwargs', {})

        self.input_recall = getattr(model_config, 'input_recall', False)

        self.pos_enc_model = self.get_pos_enc_model()

        self.embedder = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)
        self.encoder = torch.nn.ModuleList([EncoderBlock(
            d_model=self.d_model, n_heads=self.n_heads, dff=self.dff, pos_enc_model=self.pos_enc_model, attn_kwargs=self.attn_kwargs)
            for _ in range(model_config.n_layers)])
        self.linear = torch.nn.Linear(model_config.d_model, model_config.vocab_size)

        # weight tying
        if getattr(model_config, 'weight_tying', False):
            self.linear.weight = self.embedder.weight

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

    def forward(self, x):
        input_emb = self.embedder(x)
        x = input_emb
        for encoder in self.encoder:
            x = encoder(x)
            if self.input_recall:
                x = x + input_emb
                # NOTE: could add linear map before residual connection to input to place input in residual stream
                # could also concatenate rather than add
        x = self.linear(x)
        return x

class LitModel(pl.LightningModule):
    def __init__(self, model_config, data_config, train_config):
        super().__init__()
        self.model = TransformerModel(model_config)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        if train_config.compile:
            self.model = torch.compile(self.model)
            print('Model compiled.')

        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, factored_tokens):
        return self.model(factored_tokens)

    def training_step(self, batch):

        x, y = batch
        logits = self.model(x)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        metrics = self.compute_metrics(batch, logits)

        self.log_metrics(metrics, key_dir='train')

        return loss

    def validation_step(self, batch):

        x, y = batch
        logits = self.model(x)
        metrics = self.compute_metrics(batch, logits)
        self.log_metrics(metrics, key_dir='val')

        return metrics['loss']

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):

        x, y = batch
        logits = self.model(x)
        metrics = self.compute_metrics(batch, logits)

        ood_length = self.data_config.ood_test_sequence_lengths[dataloader_idx]

        self.log_metrics(metrics, key_dir='test', prefix=f'L={ood_length}', add_dataloader_idx=False)

        return metrics['loss']

    def compute_metrics(self, batch, logits):
        x, y = batch
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        # compute accuracy
        _, predicted = torch.max(logits, -1)
        per_token_acc = (predicted == y).float().mean()#.item()
        sequence_acc = (predicted == y).all(dim=1).float().mean()#.item()

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

def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Data Config - Model Config
    # Name: Seed + Date-Time
    data_str = f'MaxVal{data_config.max_value}-TrainLen{data_config.train_sequence_length}'
    model_str = f'L{model_config.n_layers}H{model_config.n_heads}D{model_config.d_model}_{model_config.pos_enc_type}_IR{model_config.input_recall}'
    if model_config.attn_kwargs.attn_score_fn != 'softmax':
        model_str += f'_{model_config.attn_kwargs.attn_score_fn}'
        if model_config.attn_kwargs.get('attn_score_fn_params', {}).get('straight_through', False):
            model_str += '-ST'
    group_name = f'{data_str} - {model_str}'

    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if getattr(train_config, 'seed', None) is not None:
        run_name = 'seed-' + str(train_config.seed) + ' - ' + run_name

    return group_name, run_name