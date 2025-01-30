import argparse
import yaml
import ast
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

import torchinfo
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import os, sys; sys.path.append(os.path.abspath('../..'))
from utils.utils import AttributeDict, print_gpu_info

from replication_data_utils import Nonogram_Dataset
from viz_utils import plot_nonogram, plot_preds_over_iters, plot_preds_over_iters_with_solution, plot_discrete_interm_iter
from lit_module import LitRecurrentModel, get_experiment_name


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str)
parser.add_argument('--debug', action='store_true', help='Run in debug mode (no logging, no checkpoints).')

args, unknown_args = parser.parse_known_args()

# load model, train, and data config
with open(os.path.join(args.config_dir, 'model_config.yaml')) as f:
    model_config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(args.config_dir, 'train_config.yaml')) as f:
    train_config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(args.config_dir, 'data_config.yaml')) as f:
    data_config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

# update configs with command line arguments (if applicable)
if len(unknown_args) > 0:
    print('='*80)
    print("Received unknown arguments:", unknown_args)
for arg_str in unknown_args:
    for (prefix, config) in [('--train_config.', train_config), ('--data_config.', data_config), ('--model_config.', model_config)]:
        if arg_str.startswith(prefix):
            key, value = arg_str[len(prefix):].split('=')
            value = ast.literal_eval(value)
            key_parts = key.split('.')
            # traverse through the config dict to the second last key
            for k in key_parts[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[key_parts[-1]] = value
            print(f"Updated {prefix}{key} to {value}")

# print configs
print('='*80)
print("Model Config:")
print(model_config)
print('-'*80)
print("Train Config:")
print(train_config)
print('-'*80)
print("Data Config:")
print(data_config)
print('='*80)

# GPU diagnostics
print_gpu_info()

# current device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set the seed for reproducibility
seed = np.random.randint(0, 2**32) # randomly sample a seed
pl.seed_everything(seed) # sets the seed for all random number generators
train_config.seed = seed

# load data
assert data_config.grid_size in [7, 15]
grid_size = data_config.grid_size
max_constraints_per_cell = data_config.max_constraints_per_cell

data_path = f'data/nonograms_{grid_size}.csv'

# update model config with grid size and max constraints per cell
model_config['max_x_pos'] = model_config['max_y_pos'] = model_config['constraint_max_val'] = grid_size
model_config['max_constraints_per_cell'] = max_constraints_per_cell


# ds = Nonogram_Dataset("data/nonograms_7.csv", board_dim=7, max_num_per_hint=4, limit=-1, seed=42)
ds = Nonogram_Dataset(data_path, board_dim=grid_size, max_num_per_hint=max_constraints_per_cell, limit=-1, seed=seed)

class MyNonogramDS(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        constraints = x.reshape(7, 7, -1).transpose(0, 1)
        solution = y.reshape(7, 7)
        return constraints, solution

nonogram_ds = MyNonogramDS(ds)

# train-test split
train_ds, val_ds = torch.utils.data.random_split(nonogram_ds, [0.8, 0.2])

# TODO: if we do OOD generalization and early stopping, don't do early stopping based on in-distribution validation set but on OOD validation set (or at least check difference)

batch_size = train_config.batch_size
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=train_config.num_workers)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=train_config.num_workers)

# create model
litmodel = LitRecurrentModel(model_config, data_config, train_config).cuda()

torchinfo.summary(litmodel.model, input_data=torch.zeros((1,7,7,8), dtype=torch.int), device='cuda')

# logger

# Group: Data Config - Model Config ... Run name: Seed + Date-Time
group_name, run_name = get_experiment_name(model_config, data_config, train_config)
train_config.experiment_run_name = run_name
train_config.experiment_group = group_name

experiment_config = dict(train_config=train_config, model_config=model_config, data_config=data_config)

callbacks = []

lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)

wandb_experiment_run = wandb.init(
    entity=train_config.wandb_config.wandb_entity,
    project=train_config.wandb_config.wandb_project,
    name=train_config.experiment_run_name, group=train_config.experiment_group,
    config=experiment_config)
logger = pl.loggers.WandbLogger(
    experiment=wandb_experiment_run, log_model=False)

# watch model for gradients and parameter evolution
logger.watch(litmodel.model, log='all', log_graph=True)

# callbacks: checkpoint and lr monitor
# checkpoint_dir = f'checkpoints/{train_config.experiment_group}-{train_config.experiment_run_name}'
# checkpoint_callback = ModelCheckpoint(
#     dirpath=checkpoint_dir,
#     filename='{epoch}-{val_loss:.4f}',
#     monitor='val/loss', # this depends on logging in the LightningModule
#     mode='min',
# )
# callbacks.append(checkpoint_callback)

# set matmul precision
if getattr(train_config, 'matmul_precision', None) is not None:
    torch.set_float32_matmul_precision(train_config.matmul_precision)
    print(f'Matmul precision set to {train_config.matmul_precision}.')


# set up trainer
trainer_kwargs = dict(
    max_epochs=train_config.max_epochs,
    logger=logger,
    callbacks=callbacks,
    precision=train_config.precision,
)

if getattr(trainer_kwargs, 'precision', None) is not None:
    print(f'Precision set to {trainer_kwargs.precision}.')


trainer = pl.Trainer(**trainer_kwargs)

# train model
trainer.fit(litmodel, train_dl, val_dl)

# evaluate on validation set
ood_test_result = trainer.test(litmodel, dataloaders=val_dl)

# plot a few samples
samples = np.random.choice(len(val_ds), 5)

litmodel = litmodel.cuda()
for i, sample in enumerate(samples):
    x, y = val_ds[sample]
    x = x.unsqueeze(0).cuda()
    _, intermediates = litmodel.forward(x, n_iters=model_config.default_n_iters, return_intermediate_states=True)
    pred_per_iter = [logits.softmax(dim=-1)[0,:,:,1].cpu().detach().numpy() for logits in intermediates['logits_states']]
    fig = plot_preds_over_iters_with_solution(pred_per_iter, y.cpu().numpy());
    wandb.log({f'sample_preds/{i+1}': wandb.Image(fig)})

    if litmodel.model.discrete_intermediate:
        pred_per_iter = [pred[0].argmax(-1).cpu().detach().numpy() for pred in intermediates['disc_interm_states']]
        fig = plot_discrete_interm_iter(pred_per_iter, y.cpu().numpy());
        wandb.log({f'sample_disc_interm/{i+1}': wandb.Image(fig)})

wandb.finish(quiet=True)