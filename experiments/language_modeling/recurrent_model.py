import torch
import pytorch_lightning as pl
import lightning.pytorch.utilities
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import wandb

from models.language_models import RecurrentTransformerLM, configure_optimizers
from utils.utils import get_cosine_schedule_with_warmup

class LitRecurrentTransformerLM(pl.LightningModule):

    def __init__(self, model_config, train_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.default_n_iters = model_config.default_n_iters

        self.model = RecurrentTransformerLM(model_config)
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Compile model: ', self.train_config.get('compile', False))
        if self.train_config.get('compile', False):
            self.model = torch.compile(self.model)

        print('Use AMP:', train_config.get('amp', False))
        self.ctx_manager = torch.amp.autocast(enabled=(train_config.get('amp', False)), dtype=torch.bfloat16, device_type='cuda')

        # for test-time evaluation
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        n_iters, n_nograd_iters = self.sample_n_iters()

        # run model without tracking gradients for n_nograd_iters
        # then run model with tracking gradients for n_iters
        if self.train_config.incremental_training and n_nograd_iters > 0:
            orig_input = x
            with torch.no_grad(), self.ctx_manager:
                x = self.model(x, n_iters=n_nograd_iters, skip_embed=False, skip_output=True)

            with self.ctx_manager:
                logits = self.model(x, orig_input=orig_input, n_iters=n_iters, skip_embed=True, skip_output=False)

        # run model with tracking gradients for random number n_iters of iters
        else:
            with self.ctx_manager:
                logits = self.model(x, n_iters=n_iters)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        self.log('train/n_nograd_iters', n_nograd_iters)
        self.log('train/n_iters', n_iters)
        self.log('train/total_iters', n_nograd_iters + n_iters)

        self.log('train/loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/ppl', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        self.log('train/tokens', batch_idx * x.size(0) * x.size(1), on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        self.log('val/loss', loss, prog_bar=True, logger=True)
        self.log('val/ppl', torch.exp(loss), prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        self.log('test/loss', loss, prog_bar=True, logger=True)
        self.log('test/ppl', torch.exp(loss), prog_bar=True, logger=True)

        # for each number of iterations, compute metrics
        n_iterss = range(1, self.model_config.test_max_n_iters+1)
        for n_iters in n_iterss:
            logits = self.model(x, n_iters=n_iters)
            metrics = dict(loss=self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1)).item())
            metrics['ppl'] = np.exp(metrics['loss'])

            self.test_step_outputs.append(dict(n_iters=n_iters, metrics=metrics))


        return metrics['loss']

    def sample_n_iters(self):
        if np.random.random() < 1 - self.train_config.progressive_training_prob:
            return (self.default_n_iters, 0)

        # randomly sample number of iterations to run model without tracking gradients
        if self.train_config.incremental_training:
            n_nograd_iters = np.random.randint(0, self.train_config.train_max_n_iters)
        else:
            n_nograd_iters = 0

        # number of iterations to run model with tracking gradients
        if self.train_config.progressive_training:
            n_iters = np.random.randint(1, self.default_n_iters - n_nograd_iters + 1)
        else:
            n_iters = self.default_n_iters

        return n_iters, n_nograd_iters

    def on_test_epoch_end(self):

        metrics_list = []
        for n_iters in range(1, self.model_config.test_max_n_iters+1):
            metrics = [output['metrics'] for output in self.test_step_outputs if output['n_iters'] == n_iters]
            metrics = {key: torch.tensor([m[key] for m in metrics]).mean().tolist() for key in metrics[0].keys()}
            metrics['n_iters'] = n_iters

            metrics_list.append(metrics)

        # plot line plots of metrics (metric vs L)
        test_df = pd.DataFrame(metrics_list)
        self.logger.log_table('test/ood_eval', dataframe=test_df)

        self.create_and_log_figs(test_df)

        self.test_step_outputs.clear()  # free memory

    def create_and_log_figs(self, test_df):
        fig = px.line(test_df, x='n_iters', y='loss', title='Loss vs n_iters')
        fig.add_vline(x=self.default_n_iters, line_dash='dash', line_color='red', annotation_text='train n_iters')
        wandb.log({'test/loss_vs_L': wandb.Plotly(fig)})

        fig = px.line(test_df, x='n_iters', y='ppl', title='PPL vs n_iters')
        fig.add_vline(x=self.default_n_iters, line_dash='dash', line_color='red', annotation_text='train n_iters')
        wandb.log({'test/ppl_vs_L': wandb.Plotly(fig)})



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
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                max_lr=self.train_config[f'{optimizer_name}_optimizer_config']['lr'],
                lr_decay_steps=cosine_scheduler_config.get('lr_decay_steps', self.train_config.n_train_steps),
                min_lr=cosine_scheduler_config.get('min_lr', None),
                warmup_iters=cosine_scheduler_config.get('warmup_steps', None),
            )

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
    # NOTE: configure_optimizer in models.language_models uses slightly different config for tensors of different ranke

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
    # Group: Model Config
    # Name: Seed + Date-Time
    model_str = f'L{model_config.n_layers}T{model_config.default_n_iters}H{model_config.n_heads}D{model_config.d_model}-{model_config.pos_enc_type}-{model_config.norm_config.norm_method}-WT{model_config.weight_tie_embed_to_token}'

    # attn_score_fn
    if model_config.attn_kwargs.attn_score_fn != 'softmax':
        model_str += f'_{model_config.attn_kwargs.attn_score_fn}'
        if model_config.attn_kwargs.get('attn_score_fn_params', {}).get('straight_through', False):
            model_str += '-ST'

    # train config (progressive training and/or incremental training)
    train_str = ''
    if train_config.get('progressive_training', False):
        train_str += 'progressive'
    if train_config.get('incremental_training', False):
        train_str += '_incremental'

    group_name = f'{model_str}' #  - {train_str} - {data_str}

    if train_str:
        group_name += f' - {train_str}'

    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if getattr(train_config, 'seed', None) is not None:
        run_name = 'seed-' + str(train_config.seed) + ' - ' + run_name

    # if exceeds 128 characters, save hash instead
    if len(group_name) > 128:
        group_name = 'HASH-' + str(hash(group_name))

    if len(run_name) > 128:
        run_name = 'HASH-' + str(hash(run_name))

    return group_name, run_name
