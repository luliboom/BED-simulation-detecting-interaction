# main_analysis.py
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CONFIG
from plotting_functions import (
    credibility,
    plot_candidate_frequency,
    plot_guide_comparison,
    plot_joyplots,
    plot_lineplot_init,
    plot_next_candidates_overall,
    plot_svi_loss,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from functions import find_interactive_drugs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------
# Load data
# --------------------------
def load_data(data_dir, truth_path):
    data_dir = Path(data_dir)
    merged_ED = torch.load(data_dir / "svi_merged_ED.pt", map_location=device)
    merged_RD = torch.load(data_dir / "svi_merged_RD.pt", map_location=device)
    merged_PD = torch.load(data_dir / "svi_merged_PD.pt", map_location=device)
    merged_UD = torch.load(data_dir / "svi_merged_UD.pt", map_location=device)
    merged_OD = torch.load(data_dir / "svi_merged_OD.pt", map_location=device)
    eig_path = data_dir / "EIG_results.pt"
    EIG = torch.load(eig_path, map_location=device) if eig_path.exists() else None
    truth = torch.load(truth_path, map_location=device, weights_only=False)
    return merged_ED, merged_RD, merged_PD, merged_UD, merged_OD, EIG, truth


merged_ED, merged_RD, merged_PD, merged_UD, merged_OD, EIG, true_parameters = load_data(
    CONFIG["data_dir"], CONFIG["truth_path"]
)
print("loaded data")


# --------------------------
# Extract common variables
# --------------------------
chosen_index = {
    "UD": merged_UD["chosen_index"].numpy(),
    "RD": merged_RD["chosen_index"].numpy(),
    "PD": merged_PD["chosen_index"].numpy(),
    "ED": merged_ED["chosen_index"].numpy(),
    "OD": merged_OD["chosen_index"].numpy(),
}

SVI_losses = {
    "UD": merged_UD["SVI_losses"].numpy(),
    "RD": merged_RD["SVI_losses"].numpy(),
    "PD": merged_PD["SVI_losses"].numpy(),
    "ED": merged_ED["SVI_losses"].numpy(),
    "OD": merged_OD["SVI_losses"].numpy(),
}

SVI_parameters = {
    "UD": merged_UD["SVI_parameters"].numpy(),
    "RD": merged_RD["SVI_parameters"].numpy(),
    "PD": merged_PD["SVI_parameters"].numpy(),
    "ED": merged_ED["SVI_parameters"].numpy(),
    "OD": merged_OD["SVI_parameters"].numpy(),
}


parameters = {
    "UD": merged_UD["parameters"].detach(),
    "RD": merged_RD["parameters"].detach(),
    "PD": merged_PD["parameters"].detach(),
    "ED": merged_ED["parameters"].detach(),
    "OD": merged_OD["parameters"].detach(),
}

print("extracted common variables")


candidate_designs = merged_ED["candidate_designs"].numpy()
rounds = np.arange(chosen_index["ED"].shape[1])
D = len(true_parameters["beta_d"])
candidate_labels = (
    [r"$\emptyset$"]
    + [f"{{{i}}}" for i in range(1, D + 1)]
    + [f"{{{i},{j}}}" for i in range(1, D) for j in range(i + 1, D + 1)]
)
parameter_labels = [
    f"$\\gamma_{{{i},{j}}}$" for i in range(1, D) for j in range(i + 1, D + 1)
]
interactions = find_interactive_drugs(D, true_parameters["part_add"])
i, j = interactions[0]
target_label = f"{{{i + 1},{j + 1}}}"
target_index = candidate_labels.index(target_label)
nonzero_idx = torch.nonzero(true_parameters["part_add"], as_tuple=True)[0].item()
target_param_label = rf"$\gamma_{{\{{{i + 1},{j + 1}\}}}}$"


# --------------------------
# Helper: save figure
# --------------------------
def save_fig(fig, name):
    fig_dir = Path(CONFIG["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")


# --------------------------
# SVI Loss
# --------------------------
SVI_losses_single = {
    "UD": SVI_losses["UD"][0, 2, :],
    "RD": SVI_losses["RD"][0, 2, :],
    "PD": SVI_losses["PD"][0, 2, :],
    "OD": SVI_losses["OD"][0, 2, :],
    "ED": SVI_losses["ED"][0, 2, :],
}

fig, axes = plot_svi_loss(
    {"ED": SVI_losses["ED"][0, 2, :]},
    y_lable="ELBO loss",
    colors=CONFIG["colors"],
    figsize=(8, 5),
)
save_fig(fig, "svi_convergence.pdf")

fig, axes = plot_svi_loss(
    {"OD": SVI_losses["OD"][0, 2, :]},
    y_lable="ELBO loss",
    colors=CONFIG["colors"],
    figsize=(8, 5),
)
save_fig(fig, "svi_convergence_OD.pdf")

SVI_parameters_single = {
    r"location $\gamma_{2,4}$": SVI_parameters["ED"][0, 2, :, 4],
    r"scale $\gamma_{2,4}$": SVI_parameters["ED"][0, 2, :, 9],
    r"location $\sigma$": SVI_parameters["ED"][0, 2, :, 6],
    r"scale $\sigma$": SVI_parameters["ED"][0, 2, :, 11],
}

fig, axes = plot_svi_loss(
    SVI_parameters_single, y_lable="Parameter value", figsize=(16, 5)
)
save_fig(fig, "svi_parameters.pdf")


# --------------------------
# EIG Loss
# --------------------------
if EIG is not None:
    min_val = min(
        EIG["initial guide samples"].min(),
        EIG["true marginal samples"].min(),
        EIG["optimized guide samples"].min(),
    )
    max_val = max(
        EIG["initial guide samples"].max(),
        EIG["true marginal samples"].max(),
        EIG["optimized guide samples"].max(),
    )

    fig, axes = plot_guide_comparison(
        EIG["true marginal samples"].detach().numpy(),
        EIG["initial guide samples"].detach().numpy(),
        EIG["optimized guide samples"].detach().numpy(),
        min_val.detach().numpy(),
        max_val.detach().numpy(),
        figsize=(12, 5),
    )
    save_fig(fig, "guide_comp.pdf")


# --------------------------
# Next Candidate over all
# --------------------------
fig, axes = plot_next_candidates_overall(
    chosen_index, D, candidate_labels, colors=CONFIG["colors"], figsize=(8, 5)
)
save_fig(fig, "next_cand_overall.pdf")

# -------------------------------
# Frequency for fixed candidate
# -------------------------------
n_rounds = chosen_index["ED"].shape[1]
rounds_to_plot = list(range(0, n_rounds, 5))
fig, ax = plot_candidate_frequency(
    chosen_index,
    target_index,
    candidate_labels,
    rounds_to_plot=rounds_to_plot,
    n_rounds=n_rounds,
    colors=CONFIG["colors"],
    figsize=(8, 5),
)
plt.show()
save_fig(fig, "freq_2_4.pdf")


# -------------------------------
# Joyplot interaction parameters
# -------------------------------
n_param_total = parameters["ED"].shape[2] // 2
n_combinations = D * (D - 1) // 2

locs_intpar_ED = parameters["ED"][:, :, :n_combinations]
locs_intpar_UD = parameters["UD"][:, :, :n_combinations]
locs_intpar_RD = parameters["RD"][:, :, :n_combinations]
locs_intpar_PD = parameters["PD"][:, :, :n_combinations]
locs_intpar_OD = parameters["OD"][:, :, :n_combinations]

scales_intpar_ED = parameters["ED"][
    :, :, n_param_total : n_param_total + n_combinations
]
scales_intpar_UD = parameters["UD"][
    :, :, n_param_total : n_param_total + n_combinations
]
scales_intpar_RD = parameters["RD"][
    :, :, n_param_total : n_param_total + n_combinations
]
scales_intpar_PD = parameters["PD"][
    :, :, n_param_total : n_param_total + n_combinations
]
scales_intpar_OD = parameters["OD"][
    :, :, n_param_total : n_param_total + n_combinations
]

locs_intpar = {
    "UD": locs_intpar_UD,
    "RD": locs_intpar_RD,
    "PD": locs_intpar_PD,
    "OD": locs_intpar_OD,
    "ED": locs_intpar_ED,
}

scales_intpar = {
    "UD": scales_intpar_UD,
    "RD": scales_intpar_RD,
    "PD": scales_intpar_PD,
    "OD": scales_intpar_OD,
    "ED": scales_intpar_ED,
}


fig, axes = plot_joyplots(
    locs_intpar_ED[2],
    scales_intpar_ED[2],
    true_parameters["beta_dd"].numpy(),
    parameter_labels,
    lapl=True,
    figsize=(12, 10),
)
save_fig(fig, "joyplot_interaction_param.pdf")


# -------------------------------
# Joyplot sigma parameter
# -------------------------------

locs_sigma_ED = parameters["ED"][:, :, n_param_total - 1]
locs_sigma_RD = parameters["RD"][:, :, n_param_total - 1]
locs_sigma_PD = parameters["PD"][:, :, n_param_total - 1]
locs_sigma_UD = parameters["UD"][:, :, n_param_total - 1]
locs_sigma_OD = parameters["OD"][:, :, n_param_total - 1]
scales_sigma_ED = parameters["ED"][:, :, -1]
scales_sigma_RD = parameters["RD"][:, :, -1]
scales_sigma_PD = parameters["PD"][:, :, -1]
scales_sigma_UD = parameters["UD"][:, :, -1]
scales_sigma_OD = parameters["OD"][:, :, -1]

locs_sigma = {
    "UD": locs_sigma_UD,
    "RD": locs_sigma_RD,
    "PD": locs_sigma_PD,
    "OD": locs_sigma_OD,
    "ED": locs_sigma_ED,
}

fig, axes = plot_joyplots(
    locs_sigma_ED[2],
    scales_sigma_ED[2],
    np.exp(true_parameters["sigma"]),
    [r"$\sigma$"],
    lapl=False,
    figsize=(8, 5),
)
save_fig(fig, "joyplot_sigma.pdf")


# -------------------------------
# Lineplot interaction parameters
# -------------------------------

keys = ["UD", "RD", "PD", "ED"]
fig, axes = plot_lineplot_init(
    designs={k: locs_intpar[k] for k in keys},
    true_values=true_parameters["beta_dd"],
    parameter_labels=parameter_labels,
    lapl=True,
    colors=CONFIG["colors"],
    single_plot=False,
    figsize=(12, 10),
)
save_fig(fig, "lineplot_interaction_param.pdf")

keys_OD = ["OD", "ED"]
fig, axes = plot_lineplot_init(
    designs={k: locs_intpar[k] for k in keys_OD},
    true_values=true_parameters["beta_dd"],
    parameter_labels=parameter_labels,
    lapl=True,
    colors=CONFIG["colors"],
    single_plot=False,
    figsize=(12, 10),
)
save_fig(fig, "lineplot_interaction_param_OD.pdf")

fig, axes = plot_lineplot_init(
    designs={k: scales_intpar[k] for k in keys_OD},
    true_values=None,
    parameter_labels=parameter_labels,
    lapl=True,
    colors=CONFIG["colors"],
    single_plot=False,
    figsize=(12, 10),
)
save_fig(fig, "lineplot_interaction_param_OD_scales.pdf")

# -------------------------------
# Lineplot sigma
# -------------------------------

fig, axes = plot_lineplot_init(
    designs={k: locs_sigma[k] for k in keys},
    true_values=np.exp(true_parameters["sigma"]),
    parameter_labels=[r"$\sigma$"],
    colors=CONFIG["colors"],
    lapl=False,
    single_plot=True,
    figsize=(8, 5),
)
save_fig(fig, "lineplot_sigma.pdf")

fig, axes = plot_lineplot_init(
    designs={k: locs_sigma[k] for k in keys_OD},
    true_values=np.exp(true_parameters["sigma"]),
    parameter_labels=[r"$\sigma$"],
    colors=CONFIG["colors"],
    lapl=False,
    single_plot=True,
    space=True,
    figsize=(8, 5),
)
save_fig(fig, "lineplot_sigma_OD.pdf")


# -------------------------------
# Lineplot {2,4} parameter
# -------------------------------


locs_24 = {
    "UD": locs_intpar_UD[:, :, target_index - D - 1],
    "RD": locs_intpar_RD[:, :, target_index - D - 1],
    "PD": locs_intpar_PD[:, :, target_index - D - 1],
    "OD": locs_intpar_OD[:, :, target_index - D - 1],
    "ED": locs_intpar_ED[:, :, target_index - D - 1],
}


fig, axes = plot_lineplot_init(
    designs={k: locs_24[k] for k in keys},
    true_values=true_parameters["beta_dd"].numpy()[target_index - D - 1],
    parameter_labels=" ",
    colors=CONFIG["colors"],
    lapl=True,
    single_plot=True,
    figsize=(8, 5),
)
save_fig(fig, "lineplot_24.pdf")

fig, axes = plot_lineplot_init(
    designs={k: locs_24[k] for k in keys_OD},
    true_values=true_parameters["beta_dd"].numpy()[target_index - D - 1],
    parameter_labels=" ",
    colors=CONFIG["colors"],
    lapl=True,
    single_plot=True,
    figsize=(8, 5),
)
save_fig(fig, "lineplot_24_OD.pdf")


# -------------------------------
# Credibility Interval
# -------------------------------
designs_24 = {
    "UD": (
        locs_intpar_UD[:, :, target_index - D - 1],
        scales_intpar_UD[:, :, target_index - D - 1],
    ),
    "RD": (
        locs_intpar_RD[:, :, target_index - D - 1],
        scales_intpar_RD[:, :, target_index - D - 1],
    ),
    "PD": (
        locs_intpar_PD[:, :, target_index - D - 1],
        scales_intpar_PD[:, :, target_index - D - 1],
    ),
    "OD": (
        locs_intpar_OD[:, :, target_index - D - 1],
        scales_intpar_OD[:, :, target_index - D - 1],
    ),
    "ED": (
        locs_intpar_ED[:, :, target_index - D - 1],
        scales_intpar_ED[:, :, target_index - D - 1],
    ),
}

fig, axes = credibility(
    designs=designs_24,
    parameter_labels=" ",
    colors=CONFIG["colors"],
    pos="lower right",
    figsize=(8, 5),
)
save_fig(fig, "credibility.pdf")


designs_intpar = {
    "UD": (locs_intpar_UD, scales_intpar_UD),
    "RD": (locs_intpar_RD, scales_intpar_RD),
    "PD": (locs_intpar_PD, scales_intpar_PD),
    "OD": (locs_intpar_OD, scales_intpar_OD),
    "ED": (locs_intpar_ED, scales_intpar_ED),
}


fig, axes = credibility(
    designs=designs_intpar,
    parameter_labels=parameter_labels,
    colors=CONFIG["colors"],
    figsize=(12, 10),
)
save_fig(fig, "credibility2.pdf")
