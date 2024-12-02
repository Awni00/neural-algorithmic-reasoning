import argparse
import yaml
import ast
import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from lightning.pytorch.utilities.parsing import AttributeDict

import os, sys; sys.path.insert(0, os.path.abspath('../..')) # add project root dir to path
from model import LitModel, get_experiment_name
from models.utils import AttributeDict
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
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("No GPU available, using CPU")

# current device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set the seed for reproducibility
seed = np.random.randint(0, 2**32) # randomly sample a seed
pl.seed_everything(seed) # sets the seed for all random number generators
train_config.seed = seed


train_ds = SortingDataset(max_value=data_config.max_value, sequence_length=data_config.train_sequence_length, batch_size=train_config.batch_size, num_samples=data_config.num_train_samples)
train_dataloader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=train_config.num_workers, pin_memory=True)

val_ds = SortingDataset(max_value=data_config.max_value, sequence_length=data_config.train_sequence_length, batch_size=train_config.batch_size, num_samples=data_config.num_val_samples)
val_dataloader = DataLoader(val_ds, batch_size=None, shuffle=False, num_workers=train_config.num_workers, pin_memory=True)

ood_test_dss = [
    SortingDataset(max_value=data_config.max_value, sequence_length=ood_seq_len, batch_size=train_config.batch_size, num_samples=data_config.num_test_samples)
    for ood_seq_len in data_config.ood_test_sequence_lengths]
ood_test_dataloaders = [
    DataLoader(ood_test_ds, batch_size=None, shuffle=False, num_workers=train_config.num_workers, pin_memory=True)
    for ood_test_ds in ood_test_dss]


# create model
model_config.vocab_size = data_config.max_value

litmodel = LitModel(model_config=model_config, train_config=train_config, data_config=data_config)

# logger
# Group: Data Config - Model Config ... Run name: Seed + Date-Time
group_name, run_name = get_experiment_name(model_config, data_config, train_config)

train_config.experiment_run_name = run_name
train_config.experiment_group = group_name

experiment_config = dict(train_config=train_config, model_config=model_config, data_config=data_config)

logger = pl.loggers.WandbLogger(
    entity=train_config.wandb_config.wandb_entity, project=train_config.wandb_config.wandb_project,
    name=train_config.experiment_run_name, group=train_config.experiment_group,
    config=experiment_config)


# callbacks: checkpoint and lr monitor
# checkpoint_dir = f'checkpoints/{train_config.experiment_group}-{train_config.experiment_run_name}'
# checkpoint_callback = ModelCheckpoint(
#     dirpath=checkpoint_dir,
#     filename='{epoch}-{val_loss:.4f}',
#     monitor='total_loss/val', # this depends on logging in the LightningModule
#     mode='min',
# )

lr_monitor = LearningRateMonitor(logging_interval='step')
# callbacks = [lr_monitor, checkpoint_callback] # no checkpointing for now
callbacks = [lr_monitor]

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
ood_test_result = trainer.test(dataloaders=ood_test_dataloaders)
print(ood_test_result)

# # training loop
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# num_steps = len(train_dataloader)
# log_interval = 100

# total_loss = 0
# correct_tokens = 0
# total_tokens = 0
# correct_sequences = 0
# total_sequences = 0

# for step, (x, y) in enumerate(train_dataloader):
#     x, y = x.to(device), y.to(device)
#     optimizer.zero_grad()
#     y_pred = model(x)
#     loss = criterion(y_pred.view(-1, max_value), y.view(-1))
#     loss.backward()
#     optimizer.step()

#     total_loss += loss.item()
#     _, predicted = torch.max(y_pred, -1)
#     correct_tokens += (predicted == y).sum().item()
#     total_tokens += y.numel()
#     correct_sequences += (predicted == y).all(dim=1).sum().item()
#     total_sequences += y.size(0)

#     if (step + 1) % log_interval == 0:
#         avg_loss = total_loss / (step + 1)
#         token_accuracy = correct_tokens / total_tokens * 100
#         sequence_accuracy = correct_sequences / total_sequences * 100
#         print(f'Step [{step + 1}/{num_steps}], Loss: {avg_loss:.4f}, Token Accuracy: {token_accuracy:.2f}%, Sequence Accuracy: {sequence_accuracy:.2f}%')

