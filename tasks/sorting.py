"""
Data generation for sorting task.

Usage:
>> dataset = SortingDataset(max_value=max_value, sequence_length=sequence_length, batch_size=batch_size, num_samples=num_samples) # NOTE: batching is handled by dataset itself
>> dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def generate_batch(max_value, sequence_length, batch_size):


    x = np.random.randint(0, max_value, size=(batch_size, sequence_length))
    y = np.sort(x, axis=1)

    x, y = torch.from_numpy(x), torch.from_numpy(y)

    # the pytoch implementation appears to be slower, as measured by %timeit on MIG instance
    # x = torch.randint(0, max_value, size=(batch_size, sequence_length))
    # y = torch.sort(x, dim=1).values

    return x, y

class SortingDataset(Dataset):
    def __init__(self, max_value, sequence_length, batch_size, num_samples, device=None):

        self.max_value = max_value
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x, y = generate_batch(max_value=self.max_value, sequence_length=self.sequence_length, batch_size=self.batch_size)
        if self.device is not None:
            x, y = x.to(self.device), y.to(self.device)
        return x, y