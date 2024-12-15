import argparse
import yaml
import ast
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchinfo

import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import os, sys; sys.path.insert(0, os.path.abspath('../..')) # add project root dir to path
# from model import LitModel, get_experiment_name
from recurrent_model import LitRecurrentModel, get_experiment_name
from utils.utils import AttributeDict, print_gpu_info
from tasks.sorting import SortingDataset


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str, default='data/dag_data.pt')
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

# initialize wandb run
# Group: Data Config - Model Config / Run name: Seed + Date-Time
group_name, run_name = get_experiment_name(model_config, data_config, train_config)

train_config.experiment_run_name = run_name
train_config.experiment_group = group_name

wandb_experiment_run = wandb.init(
    entity=train_config.wandb_config.wandb_entity, project=train_config.wandb_config.wandb_project,
    name=train_config.experiment_run_name, group=train_config.experiment_group)

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


# data
sorting_data_params = dict(
    max_value=data_config.max_value, sequence_length=data_config.train_sequence_length,
    batch_size=train_config.batch_size, random_sequence_length=data_config.train_random_sequence_length, min_sequence_length=data_config.train_min_sequence_length, device=None)
train_ds = SortingDataset(**sorting_data_params, num_samples=data_config.num_train_samples)
train_dataloader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=train_config.num_workers, pin_memory=True)

val_ds = SortingDataset(**sorting_data_params, num_samples=data_config.num_val_samples)
val_dataloader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=train_config.num_workers, pin_memory=True)

ood_test_dss = [
    SortingDataset(max_value=data_config.max_value, sequence_length=ood_seq_len, random_sequence_length=False,
        batch_size=train_config.batch_size, num_samples=data_config.num_test_samples)
    for ood_seq_len in data_config.ood_test_sequence_lengths]
ood_test_dataloaders = [
    DataLoader(ood_test_ds, batch_size=None, shuffle=False, num_workers=train_config.num_workers, pin_memory=True)
    for ood_test_ds in ood_test_dss]


# create model
model_config.vocab_size = data_config.max_value

litmodel = LitRecurrentModel(model_config=model_config, train_config=train_config, data_config=data_config)

# get torchinfo summary
model_summary = torchinfo.summary(litmodel.model, input_data=torch.zeros((1, data_config.train_sequence_length), dtype=torch.long),
    col_names=("input_size", "output_size", "num_params", "params_percent"))
print(model_summary)

model_summary_dict = AttributeDict({
    'Input size (MB)': model_summary.to_megabytes(model_summary.total_input),
    'Params size (MB)': model_summary.to_megabytes(model_summary.total_param_bytes),
    'Forward/backward pass size  (MB)': model_summary.to_megabytes(model_summary.total_output_bytes),
    'Estimated total size (MB)': model_summary.to_megabytes(model_summary.total_output_bytes + model_summary.total_param_bytes + model_summary.total_input),
    'Total Mult-Adds': model_summary.total_mult_adds,

    'trainable_params': model_summary.trainable_params, # note: numbers from torchinfo are not always accurate
    'total_params': model_summary.total_params, # note: numbers from torchinfo are not always accurate

    'num_params': sum(p.numel() for p in litmodel.model.parameters()),
    'num_trainable_params': sum(p.numel() for p in litmodel.model.parameters() if p.requires_grad)
    })


# logger

experiment_config = dict(train_config=train_config, model_config=model_config, data_config=data_config, model_summary=model_summary_dict)
wandb_experiment_run.config.update(experiment_config)

logger = pl.loggers.WandbLogger(
    experiment=wandb_experiment_run,
    config=experiment_config, log_model=train_config.wandb_config.log_model)

if getattr(train_config.wandb_config, 'watch', False):
    logger.watch(litmodel.model, log='all', log_graph=True)

# callbacks: checkpoint and lr monitor
checkpoint_dir = f'checkpoints/{train_config.experiment_group}-{train_config.experiment_run_name}'
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='{epoch}-{val/loss:.4f}',
    monitor='val/loss', # this depends on logging in the LightningModule
    mode='min',
)

lr_monitor = LearningRateMonitor(logging_interval='step')
progbar = pl.callbacks.TQDMProgressBar(refresh_rate=250)
callbacks = [lr_monitor, checkpoint_callback, progbar]

if args.debug:
    callbacks = []
    logger = None

# set matmul precision
if getattr(train_config, 'matmul_precision', None) is not None:
    torch.set_float32_matmul_precision(train_config.matmul_precision)
    print(f'Matmul precision set to {train_config.matmul_precision}.')

trainer_kwargs = dict(
    max_epochs=train_config.max_epochs,
    logger=logger,
    callbacks=callbacks,
    val_check_interval=getattr(train_config, 'val_check_interval', 1.0),
    precision=getattr(train_config, 'precision', None),
)

if getattr(trainer_kwargs, 'precision', None) is not None:
    print(f'Precision set to {trainer_kwargs.precision}.')

trainer = Trainer(**trainer_kwargs)

trainer.fit(litmodel, train_dataloader, val_dataloader)

# # test the model on OOD data
ood_test_result = trainer.test(dataloaders=ood_test_dataloaders, ckpt_path='best')
