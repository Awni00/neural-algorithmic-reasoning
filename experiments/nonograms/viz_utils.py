import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_utils import infer_constraints

def plot_preds_over_iters(pred_per_iter, solution, annotate_correctness=True):
    ncols = 4
    nrows = np.ceil(len(pred_per_iter) / ncols).astype(int)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    for iter, (iter_preds, ax) in enumerate(zip(pred_per_iter, axes.flat)):
        ax.pcolormesh(iter_preds, cmap='gray_r', edgecolors='black', linewidth=2)
        ax.invert_yaxis()
        ax.set_title(f'Iteration {iter+1}')
        ax.axis('off')
        ax.set_aspect('equal')

        if annotate_correctness:
            correctness = np.abs(iter_preds - solution)
            correctness_cmap = plt.cm.RdYlGn_r
            for row in range(iter_preds.shape[0]):
                for col in range(iter_preds.shape[1]):
                    color = correctness_cmap(correctness[row, col])
                    ax.plot(col + 0.5, row + 0.5, 'o', color=color, markersize=5, alpha=1)
    for i in range(len(pred_per_iter), nrows*ncols):
        axes.flat[i].axis('off')

    return fig

def plot_preds_over_iters_with_solution(pred_per_iter, solution):
    n_cols = 4
    n_rows = np.ceil(len(pred_per_iter) / n_cols).astype(int)

    fig = plt.figure(figsize=(2*n_cols, 2*(n_rows + 1)))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Create the 4x2 grid of subplots on the left
    ax_top = plt.subplot(gs[0])
    gs_top = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[0])

    for row in range(n_rows):
        for col in range(n_cols):
            iter = row*n_cols + col
            if iter > len(pred_per_iter) - 1:
                break
            ax = plt.Subplot(fig, gs_top[row, col])
            fig.add_subplot(ax)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.pcolormesh(pred_per_iter[iter], cmap='gray_r', edgecolors='black', linewidth=0.1)
            ax.set_title(f'Iteration {iter + 1}')
            ax.axis('off')


    # Create the plot_nonogram on the right
    ax_bottom = plt.subplot(gs[1])
    ax_top.axis('off')
    plot_nonogram(solution, infer_constraints(solution), ax=ax_bottom)


    return fig

def plot_discrete_interm_iter(pred_per_iter, solution, intermediate_vocab_size=None):
    if intermediate_vocab_size is None:
        intermediate_vocab_size = int(np.max(pred_per_iter)) + 1
    cmap = plt.get_cmap('Dark2', intermediate_vocab_size)
    n_rows = 2
    n_cols = np.ceil(len(pred_per_iter) / n_rows).astype(int)

    fig = plt.figure(figsize=(n_cols * 3, (n_rows + 1) * 3))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Create the 4x2 grid of subplots on the left
    ax_top = plt.subplot(gs[0])

    gs_top = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[0])

    for row in range(n_rows):
        for col in range(n_cols):
            iter = row * n_cols + col
            if iter > len(pred_per_iter) - 1:
                break

            ax = plt.Subplot(fig, gs_top[row, col])
            fig.add_subplot(ax)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.pcolormesh(pred_per_iter[iter], cmap=cmap, edgecolors='black', linewidth=0.1)
            ax.set_title(f'Iteration {iter + 1}')
            ax.axis('off')

    # add colorbar to top
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=None), cax=cbar_ax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=None), label='Intermediate Vocab', cax=cbar_ax)
    cbar.ax.set_yticklabels([])


    # Create the plot_nonogram on the right
    ax_bottom = plt.subplot(gs[1])
    ax_top.axis('off')

    plot_nonogram(solution, infer_constraints(solution), ax=ax_bottom)

    return fig


def plot_nonogram(solution_grid, constraints=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot the grid
    for i, row in enumerate(solution_grid):
        for j, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(plt.Rectangle((j, len(solution_grid) - i - 1), 1, 1, color='black'))
            else:
                ax.add_patch(plt.Rectangle((j, len(solution_grid) - i - 1), 1, 1, edgecolor='black', facecolor='white'))

    # Plot row clues
    if constraints is not None:
        rows_clues, cols_clues = constraints
        for i, constraints in enumerate(rows_clues):
            ax.text(-0.5, len(solution_grid) - i - 0.5, ' '.join(map(str, constraints)), ha='right', va='center')

        # Plot column clues
        for j, constraints in enumerate(cols_clues):
            ax.text(j + 0.5, len(solution_grid) + 0.5, ' '.join(map(str, constraints)), ha='center', va='bottom', rotation=90)

    # Set limits and hide axes
    ax.set_xlim(-1, len(solution_grid[0]))
    ax.set_ylim(-1, len(solution_grid))
    ax.axis('off')

    return ax