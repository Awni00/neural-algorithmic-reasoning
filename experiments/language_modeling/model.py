import torch
import pytorch_lightning as pl
import lightning.pytorch.utilities
from datetime import datetime

from models.language_models import TransformerLM, configure_optimizers
from utils.utils import get_cosine_schedule_with_warmup

class LitTransformerLM(pl.LightningModule):

    def __init__(self, model_config, train_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self.model = TransformerLM(model_config)
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Compile model: ', self.train_config.get('compile', False))
        if self.train_config.get('compile', False):
            self.model = torch.compile(self.model)

        print('Use AMP:', train_config.get('amp', False))
        self.ctx_manager = torch.amp.autocast(enabled=(train_config.get('amp', False)), dtype=torch.bfloat16, device_type='cuda')


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        with self.ctx_manager:
            logits = self.model(x)

        loss = self.criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

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
                num_warmup_steps=cosine_scheduler_config.get('num_warmup_steps', int(self.train_config.n_train_steps * 0.05)), # warmup over first 5% of steps
                num_training_steps=self.train_config.n_train_steps
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
    model_str = f'L{model_config.n_layers}H{model_config.n_heads}D{model_config.d_model}-{model_config.pos_enc_type}-{model_config.norm_config.norm_method}-WT{model_config.weight_tie_embed_to_token}'

    # attn_score_fn
    if model_config.attn_kwargs.attn_score_fn != 'softmax':
        model_str += f'_{model_config.attn_kwargs.attn_score_fn}'
        if model_config.attn_kwargs.get('attn_score_fn_params', {}).get('straight_through', False):
            model_str += '-ST'

    group_name = f'{model_str}' #  - {train_str} - {data_str}

    run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if getattr(train_config, 'seed', None) is not None:
        run_name = 'seed-' + str(train_config.seed) + ' - ' + run_name

    # if exceeds 128 characters, save hash instead
    if len(group_name) > 128:
        group_name = 'HASH-' + str(hash(group_name))

    if len(run_name) > 128:
        run_name = 'HASH-' + str(hash(run_name))

    return group_name, run_name
