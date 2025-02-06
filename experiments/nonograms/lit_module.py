import torch
import pytorch_lightning as pl
import lightning
import wandb
import numpy as np
import pandas as pd
import plotly.express as px

from model import RecurrentTransformerModel
from data_utils import calc_constraint_accuracy

from utils.utils import get_cosine_schedule_with_warmup

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

        print('Compile model: ', self.train_config.get('compile', False))
        if self.train_config.get('compile', False):
            self.model = torch.compile(self.model)

        print('Use AMP:', train_config.get('amp', False))
        self.ctx_manager = torch.amp.autocast(enabled=(train_config.get('amp', False)), dtype=torch.bfloat16, device_type='cuda')

        self.save_hyperparameters() # save model hyperparameters

        if train_config.compile:
            self.model = torch.compile(self.model)
            print('Model compiled.')

        self.criterion = torch.nn.CrossEntropyLoss()

        # for test-time evaluation
        self.test_step_outputs = []

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def sample_n_iters(self):

        if self.train_config.progressive_training and np.random.random() < 1 - self.train_config.progressive_training_prob:
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

        # algorithm:
        # n_nograd_iters ~ {0, ..., train_max_n_iters - 1}
        # n_train_iters ~ {1, ..., train_max_n_iters - n_nograd_iters}
        # this guarantees that n_train_iters + n_nograd_iters <= train_max_n_iters

        n_iters, n_nograd_iters = self.sample_n_iters()

        # run model without tracking gradients for n_nograd_iters
        # then run model with tracking gradients for n_iters
        if self.train_config.incremental_training and n_nograd_iters > 0:
            orig_input = x
            with torch.no_grad(), self.ctx_manager:
                x = self.model.forward_skip_output(x, n_iters=n_nograd_iters)

            with self.ctx_manager:
                logits = self.model.forward_skip_embed(x, orig_input, n_iters=n_iters)

        # run model with tracking gradients for random number n_iters of iters
        else:
            logits = self.model(x, n_iters=n_iters)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        metrics = self.compute_metrics(batch, logits)

        self.log_metrics(metrics, key_dir='train', prog_bar=True)

        self.log('train/n_nograd_iters', n_nograd_iters)
        self.log('train/n_iters', n_iters)
        self.log('train/total_iters', n_nograd_iters + n_iters)

        return loss

    def validation_step(self, batch):

        x, y = batch

        self.model.eval() # set model to evaluation mode

        # randomly sample n_iters for validation, as in training
        n_iters, n_nograd_iters = self.sample_n_iters()
        logits, intermediate_states = self.model(x, n_iters=n_iters+n_nograd_iters, return_intermediate_states=True)

        metrics = self.compute_metrics(batch, logits)

        # calculate constraint accuracy (which may be different from sequence accuracy)
        preds = logits.argmax(dim=-1)
        constraint_acc = calc_constraint_accuracy(x.cpu(), preds.cpu())
        metrics['constraint_acc'] = constraint_acc

        self.log_metrics(metrics, key_dir='val', prog_bar=True)

        intermediate_states_metrics = self.compute_intermediate_state_metrics(intermediate_states)
        self.log_metrics(intermediate_states_metrics, key_dir='val_interm', prog_bar=False)

        return metrics['loss']

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):

        x, y = batch

        self.model.eval() # set model to evaluation mode

        test_split_name = self.data_config.test_split_names[dataloader_idx] # name of test split associated with current batch

        # first, run model for max number of iterations, tracking intermediate states (to log entropy of logits, embedding norms, etc.)
        logits, intermediate_states = self.model(x, n_iters=self.train_config.test_max_n_iters, return_intermediate_states=True)
        # intermediate_states: Dict[str, List[torch.Tensor]] with keys: disc_interm_states, logits_states, emb_norms. List length = n_iters
        assert len(intermediate_states['emb_norms']) == self.train_config.test_max_n_iters, f"Expected {self.train_config.test_max_n_iters} intermediate states, got {len(intermediate_states['emb_norms'])}"
        if self.model.discrete_intermediate:
            assert len(intermediate_states['disc_logits_softmax_entropy']) == self.train_config.test_max_n_iters - 1, f"Expected {self.train_config.test_max_n_iters} intermediate states, got {len(intermediate_states['disc_logits_softmax_entropy'])}"

        # for each number of iterations, compute metrics
        n_iterss = range(1, self.train_config.test_max_n_iters+1)
        for n_iters in n_iterss:
            last_iter = n_iters == self.train_config.test_max_n_iters
            logits = self.model(x, n_iters=n_iters)
            metrics = self.compute_metrics(batch, logits)

            preds = logits.argmax(dim=-1)
            constraint_acc = calc_constraint_accuracy(x.cpu(), preds.cpu())
            metrics['constraint_acc'] = constraint_acc

            # average embedding norm over batch, sequence length
            metrics['emb_norms'] = intermediate_states['emb_norms'][n_iters - 1].mean()
            metrics['delta_norms'] = intermediate_states['delta_norms'][n_iters - 1].mean()
            if self.model.discrete_intermediate and not last_iter:
                # average entropy of discrete state logits over batch, sequence length
                metrics['disc_logits_softmax_entropy'] = intermediate_states['disc_logits_softmax_entropy'][n_iters - 1].mean()

            self.test_step_outputs.append(dict(test_split=test_split_name, n_iters=n_iters, metrics=metrics))

        # self.log_metrics(metrics, key_dir='test', prefix=f'L={ood_length}', add_dataloader_idx=False)

        return metrics['loss']

    def on_test_epoch_end(self):

        metrics_list = []
        for test_split in self.data_config.test_split_names:
            for n_iters in range(1, self.train_config.test_max_n_iters+1):
                metrics = [output['metrics'] for output in self.test_step_outputs if output['test_split'] == test_split and output['n_iters'] == n_iters]
                metrics = {key: torch.tensor([m[key].item() for m in metrics]).mean().tolist() for key in metrics[0].keys()}
                metrics['n_iters'] = n_iters
                metrics['test_split'] = test_split

                metrics_list.append(metrics)

        # plot line plots of metrics (metric vs L)
        test_df = pd.DataFrame(metrics_list)
        self.logger.log_table('test', dataframe=test_df)

        self.create_and_log_figs(test_df)

        self.test_step_outputs.clear()  # free memory



    def compute_metrics(self, batch, logits):
        x, y = batch
        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        # compute accuracy
        _, predicted = torch.max(logits, -1)
        per_token_acc = (predicted == y).float().mean()
        sequence_acc = (predicted == y).all(dim=(1, 2)).float().mean()

        metrics = dict(loss=loss, per_token_acc=per_token_acc, sequence_acc=sequence_acc)

        return metrics

    def compute_intermediate_state_metrics(self, intermediate_states):

        def calc_metrics_mean(metrics):
            if len(metrics) == 0:
                return None
            return torch.stack(metrics).mean()

        intermediate_states_metrics = dict()
        # compute avg entropy over batch, sequence length
        intermediate_states_metrics['avg_emb_norms'] = calc_metrics_mean(intermediate_states['emb_norms'])
        intermediate_states_metrics['avg_delta_norms'] = calc_metrics_mean(intermediate_states['delta_norms'])
        if self.model.discrete_intermediate:
            intermediate_states_metrics['avg_disc_logits_softmax_entropy'] = calc_metrics_mean(intermediate_states['disc_logits_softmax_entropy'])

        return intermediate_states_metrics


    def log_metrics(self, metrics, prefix=None, key_dir=None, **log_kwargs):
        for key, value in metrics.items():
            key_prefix = ''
            if key_dir is not None:
                key_prefix += f'{key_dir}/'
            if prefix is not None:
                key_prefix += f'{prefix}_'

            key = f'{key_prefix}{key}'
            if value is not None:
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
            cosine_scheduler_config = self.train_config.get('cosine_scheduler_config', {})
            max_lr = self.train_config[f'{optimizer_name}_optimizer_config']['lr']
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                max_lr=max_lr,
                min_lr=cosine_scheduler_config.get('min_lr', max_lr * 0.1),
                lr_decay_steps=cosine_scheduler_config.get('lr_decay_steps', self.train_config.n_train_steps),
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

        # n_iters vs sequence_acc, color = test_split
        fig = px.line(test_df, x='n_iters', y='sequence_acc', color='test_split', title='Sequence Accuracy', labels={'sequence_acc': 'Sequence Accuracy', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        fig.show()
        wandb.log({'test/sequence_acc': wandb.Plotly(fig)})

        # n_iters vs per_token_acc, color = test_split
        fig = px.line(test_df, x='n_iters', y='per_token_acc', color='test_split', title='Token-wise Accuracy', labels={'per_token_acc': 'Token-wise Accuracy', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        fig.show()
        wandb.log({'test/per_token_acc': wandb.Plotly(fig)})

        # n_iters vs constraint_acc, color = test_split
        fig = px.line(test_df, x='n_iters', y='constraint_acc', color='test_split', title='Constraint Satisfaction Accuracy', labels={'constraint_acc': 'Constraint Satisfaction Accuracy', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        fig.show()
        wandb.log({'test/constraint_acc': wandb.Plotly(fig)})

        # n_iters vs delta_norms, color = test_split
        fig = px.line(test_df, x='n_iters', y='delta_norms', color='test_split', title='Delta Norms', labels={'delta_norms': 'Delta Norms', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        fig.show()
        wandb.log({'test/delta_norms': wandb.Plotly(fig)})

        # n_iters vs avg_emb_norms, color = test_split
        fig = px.line(test_df, x='n_iters', y='emb_norms', color='test_split', title='Average Embedding Norms', labels={'avg_emb_norms': 'Average Embedding Norms', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        fig.show()
        wandb.log({'test/avg_emb_norms': wandb.Plotly(fig)})

        if self.model.discrete_intermediate:
            # n_iters vs disc_logits_softmax_entropy, color = test_split
            fig = px.line(test_df, x='n_iters', y='disc_logits_softmax_entropy', color='test_split', title='Discrete Logits Softmax Entropy', labels={'disc_logits_softmax_entropy': 'Discrete Logits Softmax Entropy', 'n_iters': 'n_iters'})
            fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
            fig.show()
            wandb.log({'test/disc_logits_softmax_entropy': wandb.Plotly(fig)})


from datetime import datetime
def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Model Config - Train Config - Data Config
    # Name: Seed + Date-Time
    data_str = ''
    # model_str = f'L{model_config.n_layers}H{model_config.n_heads}D{model_config.d_model}_{model_config.pos_enc_type}_IR{model_config.input_recall}_WT{model_config.weight_tie_embed_to_token}-{model_config.weight_tie_discrete_interm}'
    model_str = f'L{model_config.n_layers}T{model_config.default_n_iters}H{model_config.n_heads}D{model_config.d_model}_IR{model_config.input_recall}'
    if model_config.get('norm_method', None) is not None:
        model_str += f'_norm-{model_config.norm_method}'
    if model_config.intermediate_discretization.get('weight_tie_method', None) is not None:
        model_str += f'_WT{model_config.intermediate_discretization.weight_tie_method}'

    # attn_score_fn
    attn_score_fn = model_config.get('attn_kwargs', {}).get('attn_score_fn', None)
    if attn_score_fn is not None and attn_score_fn != 'softmax':
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

    # if exceeds 128 characters, save hash instead
    if len(group_name) > 128:
        group_name = 'HASH-' + str(hash(group_name))

    if len(run_name) > 128:
        run_name = 'HASH-' + str(hash(run_name))

    return group_name, run_name
