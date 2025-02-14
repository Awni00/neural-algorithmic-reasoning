import torch
import pytorch_lightning as pl
import lightning
import wandb
import numpy as np
import pandas as pd
import plotly.express as px

from model import RecurrentTransformerModel
from system2_model import RecurrentSystem2TransformerModel
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
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        if model_config.model_type == 'RecurrentTransformer':
            self.model = RecurrentTransformerModel(model_config)
        elif model_config.model_type == 'RecurrentSystem2Transformer':
            self.model = RecurrentSystem2TransformerModel(model_config)
        else:
            raise ValueError(f"Model type {model_config.model_type} not implemented!")

        # if using delta state regularization or tracking delta state KL loss, set flag
        self.need_delta_state_reg = self.model_config.delta_state_regularization.enable or self.model_config.delta_state_regularization.get('track', False)

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

        # TODO: add option for different sampling schemes (e.g., https://arxiv.org/pdf/2502.05171)
        # e.g., log normal poisson?
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

    def calc_stepwise_kl_div(self, intermediate_states):
        # if using discrete intermediate states, compute KL divergence between consecutive discrete states
        if self.model.discrete_intermediate:
            state_logits = intermediate_states['disc_logits']
        # otherwise, use the predicted logits from the embedding
        else:
            state_logits = intermediate_states['logits_states']

        kl_divs = []

        # compute KL divergence between consecutive state logits
        # regularize so that not too many tokens change their states in a single step
        for prev_logits, logits in zip(state_logits[:-1], state_logits[1:]):
            states = torch.nn.functional.log_softmax(logits, dim=-1) # shape: [B, X, Y, interm_vocab_size]
            target = torch.nn.functional.softmax(prev_logits, dim=-1) # shape: [B, X, Y, interm_vocab_size]
            kl_div = torch.nn.functional.kl_div(states, target, reduction='none').mean(dim=-1) # shape: [B, X, Y]

            kl_divs.append(kl_div)

        if len(kl_divs) == 0:
            return None

        # stepwise, tokenwise KL divergence: Delta_i^{t} = KL(S_i^{t+1} || S_i^{t})
        kl_divs = torch.stack(kl_divs) # shape: (T - 1, B, X, Y)

        return kl_divs

    def compute_regularization_loss(self, intermediate_states, log_prefix=None):
        lamda = self.model_config.delta_state_regularization.lamda

        # stepwise, tokenwise KL divergence: Delta_i^{t} = KL(S_i^{t+1} || S_i^{t})
        kl_divs = self.calc_stepwise_kl_div(intermediate_states) # shape: (T - 1, B, X, Y)

        if kl_divs is None:
            return 0

        # Delta^{t} = mean_{batch} sum_{i,j} Delta_{i,j}^{t}
        stepwise_kl_div = kl_divs.sum(dim=(2,3)) # shape: (T - 1, B)
        stepwise_kl_div = stepwise_kl_div.mean(dim=1) # shape: (T - 1,)

        # maximum single-step change in KL Delta: max_{t} Delta^{t}
        max_step_kl_div = stepwise_kl_div.max() # scalar
        reg_loss = max_step_kl_div

        if log_prefix is not None:
            self.log(f'{log_prefix}/state_change_reg_loss', reg_loss)

        return lamda * reg_loss

    def compute_loss(self, logits, y, intermediate_states=None, log_prefix=None):

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        if log_prefix is not None:
            self.log(f'{log_prefix}/loss', loss)

        if self.need_delta_state_reg:
            reg_loss = self.compute_regularization_loss(intermediate_states, log_prefix=log_prefix)
            if self.model_config.delta_state_regularization.enable:
                loss += reg_loss

                if log_prefix is not None:
                    self.log(f'{log_prefix}/total_loss', loss)

        return loss

    def training_step(self, batch):

        x, y = batch

        intermediate_states = None

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
                x, intermediate_states = self.model.forward_skip_decode(x, n_iters=n_nograd_iters, return_intermediate_states=False)

            with self.ctx_manager:
                logits, intermediate_states = self.model.forward_skip_encode(x, orig_input, n_iters=n_iters, return_intermediate_states=self.need_delta_state_reg)

        # run model with tracking gradients for random number n_iters of iters
        else:
            with self.ctx_manager:
                logits, intermediate_states = self.model(x, n_iters=n_iters, return_intermediate_states=self.need_delta_state_reg)

        loss = self.compute_loss(logits, y, intermediate_states, log_prefix='train')

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
        # TODO: add logging of compute token scores, etc. for intermediate states

        # compute and log loss
        loss = self.compute_loss(logits, y, intermediate_states, log_prefix='val')

        metrics = self.compute_metrics(batch, logits)

        # calculate constraint accuracy (which may be different from sequence accuracy)
        preds = logits.argmax(dim=-1)
        constraint_acc = calc_constraint_accuracy(x.cpu(), preds.cpu())
        metrics['constraint_acc'] = constraint_acc

        self.log_metrics(metrics, key_dir='val', prog_bar=True)

        intermediate_states_metrics = self.compute_intermediate_state_metrics(intermediate_states)
        self.log_metrics(intermediate_states_metrics, key_dir='val_interm', prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx=0, dataloader_idx=0):

        x, y = batch

        self.model.eval() # set model to evaluation mode

        test_split_name = self.data_config.test_split_names[dataloader_idx] # name of test split associated with current batch

        # first, run model for max number of iterations, tracking intermediate states (to log entropy of logits, embedding norms, etc.)
        logits, intermediate_states = self.model(x, n_iters=self.train_config.test_max_n_iters, return_intermediate_states=True)
        # intermediate_states: Dict[str, List[torch.Tensor]] with keys: disc_interm_states, logits_states, emb_norms. List length = n_iters

        if self.need_delta_state_reg:
            delta_kl_divs = self.calc_stepwise_kl_div(intermediate_states) # shape: (T - 1, B, X, Y)
            stepwise_delta_kl = delta_kl_divs.sum(dim=(2,3)).mean(dim=1) # shape: (T - 1,)


        # compute and log loss
        loss = self.compute_loss(logits, y, intermediate_states, log_prefix='test')

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
            metrics['normalized_delta_norms'] = metrics['delta_norms'] / metrics['emb_norms']
            if self.model_config.model_type == 'RecurrentSystem2Transformer':
                metrics['compute_token_scores_entropy'] = intermediate_states['compute_token_scores_entropy'][n_iters - 1].mean()
                metrics['candidate_norms'] = intermediate_states['candidate_norms'][n_iters - 1].mean()
                metrics['token_update_norms'] = intermediate_states['token_update_norms'][n_iters - 1].mean()
                metrics['normalized_token_update_norms'] = intermediate_states['normalized_token_update_norms'][n_iters - 1].mean()
            if self.model.discrete_intermediate and not last_iter:
                # average entropy of discrete state logits over batch, sequence length
                metrics['disc_logits_softmax_entropy'] = intermediate_states['disc_logits_softmax_entropy'][n_iters - 1].mean()
            if self.need_delta_state_reg and n_iters - 1 < len(stepwise_delta_kl):
                metrics['step_delta_kl_div'] = stepwise_delta_kl[n_iters - 1]

            self.test_step_outputs.append(dict(test_split=test_split_name, n_iters=n_iters, metrics=metrics))

        return loss

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

        # compute accuracy
        _, predicted = torch.max(logits, -1)
        per_token_acc = (predicted == y).float().mean()
        sequence_acc = (predicted == y).all(dim=(1, 2)).float().mean()

        metrics = dict(per_token_acc=per_token_acc, sequence_acc=sequence_acc)

        return metrics

    def compute_intermediate_state_metrics(self, intermediate_states):

        def calc_metrics_mean(metrics):
            if len(metrics) == 0:
                return None
            return torch.stack(metrics).mean()

        intermediate_states_metrics = dict()
        # compute avg entropy over batch, sequence length
        metrics = ['emb_norms', 'delta_norms']
        if self.model.discrete_intermediate:
            metrics += ['disc_logits_softmax_entropy']
        if self.model_config.model_type == 'RecurrentSystem2Transformer':
            metrics += ['compute_token_scores_entropy', 'candidate_norms', 'token_update_norms', 'normalized_token_update_norms']
        for metric in metrics:
            intermediate_states_metrics[f'avg_{metric}'] = calc_metrics_mean(intermediate_states[metric])

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

        # n_iters vs normalized_delta_norms, color = test_split
        fig = px.line(test_df, x='n_iters', y='normalized_delta_norms', color='test_split', title='Normalized Delta Norms', labels={'normalized_delta_norms': 'Normalized Delta Norms', 'n_iters': 'n_iters'})
        fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
        fig.show()
        wandb.log({'test/normalized_delta_norms': wandb.Plotly(fig)})

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

        if self.model_config.model_type == 'RecurrentSystem2Transformer':
            # n_iters vs avg_compute_token_scores_entropy, color = test_split
            fig = px.line(test_df, x='n_iters', y='compute_token_scores_entropy', color='test_split', title='Compute Token Scores Entropy', labels={'avg_compute_token_scores_entropy': 'Compute Token Scores Entropy', 'n_iters': 'n_iters'})
            fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
            fig.show()
            wandb.log({'test/compute_token_scores_entropy': wandb.Plotly(fig)})

            # n_iters vs avg_candidate_norms, color = test_split
            fig = px.line(test_df, x='n_iters', y='candidate_norms', color='test_split', title='Candidate Norms', labels={'avg_candidate_norms': 'Candidate Norms', 'n_iters': 'n_iters'})
            fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
            fig.show()
            wandb.log({'test/candidate_norms': wandb.Plotly(fig)})

            # n_iters vs avg_token_update_norms, color = test_split
            fig = px.line(test_df, x='n_iters', y='token_update_norms', color='test_split', title='Token Update Norms', labels={'avg_token_update_norms': 'Token Update Norms', 'n_iters': 'n_iters'})
            fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
            fig.show()
            wandb.log({'test/token_update_norms': wandb.Plotly(fig)})

            # n_iters vs avg_normalized_token_update_norms, color = test_split
            fig = px.line(test_df, x='n_iters', y='normalized_token_update_norms', color='test_split', title='Normalized Token Update Norms', labels={'normalized_token_update_norms': 'Normalized Token Update Norms', 'n_iters': 'n_iters'})
            fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
            fig.show()
            wandb.log({'test/normalized_token_update_norms': wandb.Plotly(fig)})

        if self.need_delta_state_reg:
            # n_iters vs step_delta_kl_div, color = test_split
            fig = px.line(test_df, x='n_iters', y='step_delta_kl_div', color='test_split', title='Stepwise Delta KL Divergence', labels={'step_delta_kl_div': 'Stepwise Delta KL Divergence', 'n_iters': 'n_iters'})
            fig.add_vline(x=self.train_config.train_max_n_iters, line_dash='dash', line_color='black', annotation_text='train_max_n_iters', annotation_position='top right')
            fig.show()
            wandb.log({'test/step_delta_kl_div': wandb.Plotly(fig)})

from datetime import datetime
def get_experiment_name(model_config, data_config, train_config):
    # Format:
    # Group: Model Config - Train Config - Data Config
    # Name: Seed + Date-Time
    data_str = ''
    # model_str = f'L{model_config.n_layers}H{model_config.n_heads}D{model_config.d_model}_{model_config.pos_enc_type}_IR{model_config.input_recall}_WT{model_config.weight_tie_embed_to_token}-{model_config.weight_tie_discrete_interm}'
    model_str = f'{model_config.model_type}-L{model_config.n_layers}T{model_config.default_n_iters}H{model_config.n_heads}D{model_config.d_model}_IR{model_config.input_recall}'
    if model_config.model_type == 'RecurrentSystem2Transformer':
        model_str += f'_CT{model_config.n_compute_tokens}'
    if model_config.get('norm_method', None) is not None:
        model_str += f'_norm-{model_config.norm_method}'
    if model_config.intermediate_discretization.get('weight_tie_method', None) is not None:
        model_str += f'_WT{model_config.intermediate_discretization.weight_tie_method}'
    if model_config.delta_state_regularization.enable:
        model_str += f'_DSR-{model_config.delta_state_regularization.lamda}'

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

    group_name = f'{model_str} - {train_str}' #  - {data_str}

    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if getattr(train_config, 'seed', None) is not None:
        run_name = 'seed-' + str(train_config.seed) + ' - ' + run_name

    # if exceeds 128 characters, save hash instead
    if len(group_name) > 128:
        group_name = 'HASH-' + str(hash(group_name))

    if len(run_name) > 128:
        run_name = 'HASH-' + str(hash(run_name))

    return group_name, run_name
