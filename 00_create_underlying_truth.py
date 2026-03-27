import os

import numpy as np
import torch

import config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Reproducibility
SEED = 4
torch.manual_seed(SEED)

# Model dimensions
D = 4
n_combinations = D * (D - 1) // 2
n_single_effect = 0
n_interactions = 1  # Between 0 and n_comb

beta_0 = torch.rand(1, device=device)

part_eff = torch.zeros(D, device=device)
part_eff[:n_single_effect] = 1
part_eff = part_eff[torch.randperm(D, device=device)]
beta_d = part_eff

part_add = torch.zeros(int(n_combinations), device=device)  # [n_comb]
part_add[:n_interactions] = 1
part_add = part_add[torch.randperm(n_combinations, device=device)]
beta_dd = part_add

sigma = np.log(0.4)


data_dir = config.CONFIG["data_dir"]

# Underlying truth
truth_file = os.path.join(data_dir, "underlying_truth.pt")

# Save all in a dictionary
torch.save(
    {
        "beta_0": beta_0,
        "beta_d": beta_d,
        "beta_dd": beta_dd,
        "sigma": sigma,
        "part_add": part_add,
        "part_eff": part_eff,
    },
    truth_file,
)

print(
    {
        "beta_0": beta_0,
        "beta_d": beta_d,
        "beta_dd": beta_dd,
        "sigma": sigma,
        "part_add": part_add,
        "part_eff": part_eff,
    }
)

print(f"Truth file created: {truth_file}")
