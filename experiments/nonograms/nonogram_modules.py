import torch
from torch import nn

class PositionalEmbedding2D(nn.Module):

    def __init__(self, max_x_pos, max_y_pos, dim):
        super().__init__()
        self.x_embedding = nn.Embedding(max_x_pos, dim)
        self.y_embedding = nn.Embedding(max_y_pos, dim)

    def forward(self, inputs):
        # given [batch_size, x, y] tensor, return [batch_size, x, y, dim] tensor with positional embeddings
        batch_size, x, y, dim = inputs.shape
        x_pos = torch.arange(x, device=inputs.device).unsqueeze(0).expand(batch_size, x)
        y_pos = torch.arange(y, device=inputs.device).unsqueeze(0).expand(batch_size, y)

        x_emb = self.x_embedding(x_pos) # shape: [batch_size, x, dim]
        y_emb = self.y_embedding(y_pos) # shape: [batch_size, y, dim]

        expanded_x_emb = x_emb.unsqueeze(-2).expand(-1, -1, y, -1) # shape: [batch_size, x, y, dim]
        expanded_y_emb = y_emb.unsqueeze(-3).expand(-1, x, -1, -1) # shape: [batch_size, x, y, dim]

        xy_emb = expanded_x_emb + expanded_y_emb

        return xy_emb

class ConstraintEmbedder(nn.Module):

    def __init__(self, constraint_max_val, max_constraints_per_cell, dim):
        super().__init__()
        self.embedder = nn.Embedding(constraint_max_val + 1, dim // (2 * max_constraints_per_cell))

    def forward(self, inputs):
        # given [batch_size, x, y, max_constraints_per_cell] tensor, return [batch_size, x, y, dim] tensor with constraint embeddings
        # concatenation of embeddings for each constraint:
        # max_constraints_per_cell embeddings for row constraints and max_constraints_per_cell embeddings for column constraints
        # each embedding of dimension dim // (2 * max_constraints_per_cell), concatenated to form a single embedding of dimension dim

        batch_size, x, y, _ = inputs.shape
        embedded_constraints = self.embedder(inputs)
        return embedded_constraints.view(batch_size, x, y, -1)