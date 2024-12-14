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
    def __init__(self,
        max_value, sequence_length, batch_size, num_samples,
        random_sequence_length=False, min_sequence_length=None, include_bos_eos=False, device=None):

        self.max_value = max_value
        self.sequence_length = sequence_length
        self.random_sequence_length = random_sequence_length
        self.min_sequence_length = min_sequence_length if min_sequence_length is not None else 2
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.include_bos_eos = include_bos_eos
        if self.include_bos_eos:
            self.bos = max_value
            self.eos = max_value + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        if self.random_sequence_length:
            L = np.random.randint(self.min_sequence_length, self.sequence_length + 1)
        else:
            L = self.sequence_length

        x, y = generate_batch(max_value=self.max_value, sequence_length=L, batch_size=self.batch_size)
        if self.include_bos_eos:
            x = torch.cat([torch.full((self.batch_size, 1), self.bos), x, torch.full((self.batch_size, 1), self.eos)], dim=1)
            y = torch.cat([torch.full((self.batch_size, 1), self.bos), y, torch.full((self.batch_size, 1), self.eos)], dim=1)

        if self.device is not None:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
