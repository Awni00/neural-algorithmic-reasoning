import torch
import torch.nn as nn

class ConcatCombine(nn.Module):
    def __init__(self, dim, dim2=None):
        super().__init__()
        self.dim = dim
        self.dim2 = dim2 if dim2 is not None else dim
        self.total_dim = self.dim + self.dim2
        self.combine = nn.Linear(self.total_dim, self.dim, bias = False)

    def forward(self, x, skip):
        concatted_skip = torch.cat((skip, x), dim = -1)
        return self.combine(concatted_skip)