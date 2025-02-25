import numpy as np
import torch
from tqdm import trange

def generate_solution_grid(rows, cols, fill_probability=0.5):
    """Generate a random solution grid with given dimensions."""
    return [[1 if np.random.random() < fill_probability else 0 for _ in range(cols)] for _ in range(rows)]

def infer_constraints(grid, pad=True):
    """Calculate row and column constraints for a given solution grid."""

    def extract_line_constraints(line):
        clues = []
        count = 0
        for cell in line:
            if cell == 1:
                count += 1
            elif count > 0:
                clues.append(count)
                count = 0
        if count > 0:
            clues.append(count)
        return clues or [0]

    rows_constraints = [extract_line_constraints(row) for row in grid]
    cols_constraints = [extract_line_constraints(col) for col in zip(*grid)]

    if pad:
        n, m = len(grid), len(grid[0])
        max_constraints_length = np.ceil(max(n, m) / 2).astype(int)

        def pad_constraints(cons):
            return cons + [0] * (max_constraints_length - len(cons))

        rows_constraints = [pad_constraints(cons) for cons in rows_constraints]
        cols_constraints = [pad_constraints(cons) for cons in cols_constraints]


    return rows_constraints, cols_constraints


def generate_nonogram(rows, cols, fill_probability=0.5):
    """Generate a nonogram puzzle with the specified dimensions."""
    solution_grid = generate_solution_grid(rows, cols, fill_probability)
    rows_clues, cols_clues = infer_constraints(solution_grid)
    return rows_clues, cols_clues, solution_grid

def generate_nonogram_dataset(grid_size, n_samples, fill_probability=0.5):

    X = []
    Y = []
    for _ in trange(n_samples):
        row_constraints, col_constraints, solution = generate_nonogram(grid_size, grid_size, fill_probability)

        # create [grid_size, grid_size, ceil(grid_size / 2) * 2] array of constraints for each cell
        constraints = np.array([[np.concat([col_constraints[col], row_constraints[row]]) for col in range(grid_size)] for row in range(grid_size)])

        X.append(constraints)
        Y.append(solution)

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    nonogram_ds = torch.utils.data.TensorDataset(X, Y)
    return nonogram_ds

def get_row_col_constraints(inputs):
    """
    Translate the input tensor into row and column constraints.

    Args:
        inputs: [n_rows, n_cols, 2 * max_constraints_per_cell]
            For each row, col inputs[row][col] is an integer list of length 2 * max_constraints_per_cell.
            The first max_constraints_per_cell numbers are column constraints.
            The last max_constraints_per_cell numbers are row constraints.
            This is the format generated by `generate_nonogram_dataset`, and is used to initialize the model embeddings.

    Returns:
        row constraints: [n_rows, max_constraints_per_cell]
        col constraints: [n_cols, max_constraints_per_cell]
    """

    n_rows, n_cols = inputs.shape[:2]
    inputs = inputs.numpy().tolist()

    # first four numbers are column constrainsts, last four numbers are row constraints
    col_cons = [inputs[0][col][:4] for col in range(n_cols)]
    row_cons = [inputs[row][0][4:] for row in range(n_rows)]


    # check if the constraints are correct
    for row in range(n_rows):
        for col in range(n_cols):
            assert (inputs[row][col][4:] == row_cons[row])
            assert (inputs[row][col][:4] == col_cons[col])

    return row_cons, col_cons

# helper functions for calculating accuracy of trained model

def check_constraint_satisfied(constraints, solution):
    row_cons, col_cons = get_row_col_constraints(constraints)

    sol_row_cons, sol_col_cons = infer_constraints(solution)

    row_satisfied = all([np.array_equal(a, b) for a, b in zip(row_cons, sol_row_cons)])
    col_satisfied = all([np.array_equal(a, b) for a, b in zip(col_cons, sol_col_cons)])

    return row_satisfied and col_satisfied

def calc_constraint_accuracy(constraints, solutions):
    return np.mean([check_constraint_satisfied(c, s) for c, s in zip(constraints, solutions)])