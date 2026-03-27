import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib import colors as mcolors
from scipy.stats import laplace, lognorm
from scipy.stats import norm as normal_dist

from config import CONFIG  # applies rcParams on import


# --------------------------
# Helper: calculate Credible Interval
# --------------------------
def credible_interval(samples, level=0.95):
    alpha = 1 - level
    lower = torch.quantile(samples, alpha / 2, dim=0)  # shape: [n_init, n_rounds]
    upper = torch.quantile(samples, 1 - alpha / 2, dim=0)  # shape: [n_init, n_rounds]
    return lower, upper


# -------------------------------
# SVI Loss Plot
# -------------------------------
def plot_svi_loss(SVI_losses, y_lable, colors=None, figsize=(5, 4)):

    designs = list(SVI_losses.keys())
    n_designs = len(designs)

    # Choose grid layout automatically
    ncols = min(4, n_designs)
    nrows = math.ceil(n_designs / ncols)

    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # Determine global y-limits
    ymin = min(np.nanmin(SVI_losses[key]) for key in designs)
    ymax = max(np.nanmax(SVI_losses[key]) for key in designs)
    diff = ymax - ymin

    for i, key in enumerate(designs):
        ax = axes[i]
        loss = SVI_losses[key]
        n_steps = loss.shape[0]

        color = colors[key] if colors and key in colors else "#ff7f0e"

        ax.plot(range(n_steps), loss, lw=1.5, color=color)
        ax.set_title(f"{key}")
        ax.set_xlabel("SVI step")
        ax.set_ylabel(y_lable)
        ax.set_ylim(ymin - 0.05 * diff, ymax + 0.05 * diff)
        ax.grid(True, ls="--", alpha=0.3)

    # Remove unused axes if grid not full
    for j in range(n_designs, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n_designs]


def plot_guide_comparison(
    true_marginal, marginal_guide1, marginal_guide2, min_val, max_val, figsize=(12, 5)
):

    num_bins = 100
    bins = np.linspace(min_val, max_val, num_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.3)

    axes[0].set_title("Before Optimization")
    axes[0].hist(true_marginal[..., 6], alpha=0.5, label="Truth", bins=bins)
    axes[0].hist(marginal_guide1[..., 6], alpha=0.5, label="Variational", bins=bins)
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, ls="--", alpha=0.3)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].set_title("After Optimization")
    axes[1].hist(true_marginal[..., 6], alpha=0.5, label="Truth", bins=bins)
    axes[1].hist(marginal_guide2[..., 6], alpha=0.5, label="Variational", bins=bins)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    return fig, axes


# -------------------------------
# Next candidate over all
# -------------------------------
def plot_next_candidates_overall(
    chosen_index, D, candidate_labels=None, colors=None, figsize=(8, 5)
):

    # Total number of candidates = empty + single + pairs
    n_candidates = 1 + D + D * (D - 1) // 2
    values = np.arange(n_candidates)

    # Candidate labels
    if candidate_labels is None:
        candidate_labels = (
            [r"$\emptyset$"]
            + [f"{{{i}}}" for i in range(1, D + 1)]
            + [f"{{{i},{j}}}" for i in range(1, D) for j in range(i + 1, D + 1)]
        )

    # Compute counts
    counts = {}
    for key in chosen_index:
        idx_flat = chosen_index[key].reshape(-1)
        counts[key] = np.array([(idx_flat == v).sum() for v in values])

    designs = list(chosen_index.keys())
    n_designs = len(designs)

    x = np.arange(n_candidates)
    width = 0.8 / n_designs  # total bar width stays constant

    fig, ax = plt.subplots(figsize=figsize)

    for i, design in enumerate(designs):
        offset = (i - (n_designs - 1) / 2) * width
        color = colors[design] if colors and design in colors else None

        ax.bar(x + offset, counts[design], width, label=design, alpha=0.7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(candidate_labels, rotation=90)
    ax.set_xlabel("Candidate design")
    ax.set_ylabel("Total frequency chosen")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig, ax


# -------------------------------
# Frequency for fixed candidate
# -------------------------------
def plot_candidate_frequency(
    chosen_index,
    target_label_index,
    candidate_labels=None,
    rounds_to_plot=None,
    n_rounds=1,
    colors=None,
    figsize=(8, 5),
):

    strategies = list(chosen_index.keys())
    n_strategies = len(strategies)

    n_rounds_total = chosen_index[strategies[0]].shape[1]

    # Default: all rounds
    if rounds_to_plot is None:
        rounds_to_plot = np.arange(n_rounds_total)
    else:
        rounds_to_plot = np.sort(np.unique(rounds_to_plot))

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    block_labels = []
    block_freqs = {key: [] for key in strategies}

    for i in range(len(rounds_to_plot)):
        start = rounds_to_plot[i]
        end = min(start + 5, n_rounds)
        block_labels.append(f"{start + 1}-{end}")

        for key in strategies:
            freq = (chosen_index[key][:, start:end] == target_label_index).mean()
            block_freqs[key].append(freq)

    x_positions = np.arange(len(block_labels))
    width = 0.8 / n_strategies  # keep total bar width constant

    for i, key in enumerate(strategies):
        offset = (i - (n_strategies - 1) / 2) * width
        color = colors[key] if colors and key in colors else None
        ax.bar(
            x_positions + offset,
            block_freqs[key],
            width,
            alpha=0.7,
            color=color,
            label=key,
        )

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Frequency")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(block_labels, rotation=45, ha="right")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    return fig, ax


# -------------------------------
# Joyplot interaction parameters
# -------------------------------
def plot_joyplots(
    locs,
    scales,
    true_values,
    parameter_labels,
    lapl=True,
    round_spacing=1,
    figsize=(8, 5),
):

    # Ensure 2D arrays for consistency
    if locs.ndim == 1:
        locs = locs[:, np.newaxis]
        scales = scales[:, np.newaxis]
        true_values = (
            np.array([true_values])
            if np.isscalar(true_values)
            else np.array(true_values)
        )
        parameter_labels = parameter_labels

    n_rounds, n_params = locs.shape
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if n_params > 1 else [axes]

    # Colormap
    cmap = cm.get_cmap("viridis")
    norm_cmap = mcolors.Normalize(vmin=0, vmax=n_rounds - 1)
    scalarMap = cm.ScalarMappable(norm=norm_cmap, cmap=cmap)

    for idx in range(n_params):
        ax = axes[idx]
        offset = 0
        for r in range(0, n_rounds, 3):
            color = scalarMap.to_rgba(r)
            if lapl:
                x = np.linspace(-1.5, 1.5, 1000)
                y = laplace.pdf(x, loc=locs[r, idx], scale=scales[r, idx])
            else:
                x_min = 0.01
                x_max = np.max(np.exp(locs + 2 * scales).numpy())
                x = np.linspace(x_min, x_max, 1000)
                y = lognorm.pdf(x, s=scales[r, idx], scale=np.exp(locs[r, idx]))
            ax.plot(x, y + offset, color=color, linewidth=1)
            offset += y.max() * round_spacing  # spacing between rounds

        ax.axvline(true_values[idx], color="red", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_title(parameter_labels[idx], fontsize=20)
        ax.set_xlabel("Parameter value")

    # Remove unused axes
    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    # Colorbar for rounds
    cbar = fig.colorbar(
        scalarMap, ax=axes, orientation="vertical", fraction=0.03, pad=0.02
    )
    cbar.set_label("Round")

    return fig, axes


# ------------------------------------------------
# Lineplot parameters (uncert. over process init)
# ------------------------------------------------
def plot_lineplot_init(
    designs,
    true_values,
    parameter_labels,
    colors,
    lapl=True,
    single_plot=True,
    space=False,
    figsize=(8, 5),
):

    # determine number of parameters and rounds
    sample_design = next(iter(designs.values()))
    n_parameters = sample_design.shape[2] if sample_design.ndim == 3 else 1
    n_rounds = sample_design.shape[1]

    n_cols = min(3, n_parameters)
    n_rows = int(np.ceil(n_parameters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_parameters > 1 else [axes]
    rounds = np.arange(n_rounds)

    for idx in range(n_parameters):
        ax = axes[idx]
        for key, locs in designs.items():
            if n_parameters > 1:
                loc = locs[:, :, idx]
            else:
                loc = locs

            if lapl:
                median = torch.quantile(loc, 0.5, dim=0).numpy()
                mean = torch.mean(loc, dim=0).numpy()
                q25 = torch.quantile(loc, 0.25, dim=0).numpy()
                q75 = torch.quantile(loc, 0.75, dim=0).numpy()
            else:
                samples = torch.exp(loc)  # median per initialization for lognormal
                median = torch.quantile(samples, 0.5, dim=0)
                mean = torch.mean(samples, dim=0)
                q25 = torch.quantile(samples, 0.25, dim=0)
                q75 = torch.quantile(samples, 0.75, dim=0)

            ax.plot(rounds + 1, median, color=colors[key], label=key)
            ax.plot(rounds + 1, mean, color=colors[key], linestyle=":")
            ax.fill_between(rounds + 1, q25, q75, color=colors[key], alpha=0.2)

        # True value
        if true_values is not None:
            true_val = true_values[idx] if n_parameters > 1 else true_values
            ax.axhline(true_val, color="black", linestyle="--", label="True value")
        if space:
            ax.set_ylim(0.1, 0.7)
        ax.set_xlabel("Round")
        ax.set_ylabel("Posterior estimate")
        ax.set_title(parameter_labels[idx])
        ax.grid(True, ls="--", alpha=0.3)

    if single_plot:
        axes[0].legend(frameon=False, loc="lower right")
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(handles),
            frameon=False,
        )

    # remove unused axes
    for j in range(n_parameters, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes


def credibility(
    designs,
    parameter_labels,
    colors,
    n_sample=1000,
    pos="upper right",
    figsize=(8, 5),
    single_plot=True,
):

    sample_design = next(iter(designs.values()))[0]

    # detect dimensions
    n_parameters = sample_design.shape[2] if sample_design.ndim == 3 else 1
    n_rounds = sample_design.shape[1]

    # subplot layout
    n_cols = min(3, n_parameters)
    n_rows = int(np.ceil(n_parameters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_parameters > 1 else [axes]

    z = normal_dist.ppf(0.975)  # 95% CI

    for idx in range(n_parameters):
        ax = axes[idx]

        for key, (loc, scale) in designs.items():
            # select parameter
            if n_parameters > 1:
                loc_p = loc[:, :, idx]
                scale_p = scale[:, :, idx]
            else:
                loc_p = loc
                scale_p = scale

            lower = loc_p - z * scale_p
            upper = loc_p + z * scale_p

            zero_excluded = (lower > 0) | (upper < 0)
            prop = zero_excluded.float().mean(dim=0)

            ax.plot(range(1, n_rounds + 1), prop, color=colors[key], label=key)

            full_rounds = np.where(prop.numpy() == 1)[0]
            if len(full_rounds) > 0:
                print(f"{key}, param {idx}: rounds → {full_rounds + 1}")

        ax.set_xlabel("Round")
        ax.set_ylabel("Prop. CrI excl. 0")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(parameter_labels[idx])
        ax.grid(True, ls="--", alpha=0.3)

    # legend handling (same logic as your other function)
    if single_plot:
        axes[0].legend(frameon=False, loc=pos)
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(handles),
            frameon=False,
        )

    # remove unused axes
    for j in range(n_parameters, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes
