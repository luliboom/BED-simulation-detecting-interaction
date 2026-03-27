import argparse
import math
import os
import random
import sys

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from functions import (
    create_candidates,
    get_combinations,
    get_prior,
    kl_exact,
    run_experiment,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Euler batch script with parameters")
parser.add_argument(
    "--seed", type=int, default=1, help="Random seed for reproducibility"
)
parser.add_argument("--n_rounds", type=int, default=1, help="Number of rounds")
parser.add_argument("--start_lr_SVI", type=float, default=1e-03, help="start lr SVI")
parser.add_argument("--end_lr_SVI", type=float, default=1e-06, help="end lr SVI")
parser.add_argument(
    "--num_steps_SVI", type=int, default=5000, help="Number of steps SVI"
)
args = parser.parse_args()

# Initialization
D = 4
n_combinations = D * (D - 1) // 2
n_train = 1 + D + n_combinations  # [n_no_pert + n_single + n_comb]

n_rounds = args.n_rounds
num_steps_SVI = args.num_steps_SVI
start_lr_SVI = args.start_lr_SVI
end_lr_SVI = args.end_lr_SVI

num_steps_EIG, start_lr_EIG, end_lr_EIG = 1000, 0.5, 0.001
optimizer_EIG = pyro.optim.ExponentialLR(
    {
        "optimizer": torch.optim.Adam,
        "optim_args": {"lr": start_lr_EIG},
        "gamma": (end_lr_EIG / start_lr_EIG) ** (1 / num_steps_EIG),
    }
)
# Seed
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
pyro.set_rng_seed(SEED)
torch.manual_seed(SEED)


# underlying truth
_batch_dir = config.CONFIG["data_dir"]
truth_file = os.path.join(_batch_dir, "underlying_truth.pt")
if not os.path.exists(truth_file):
    raise FileNotFoundError(
        f"Truth file not found. Set BATCH_DIR env var to the directory containing underlying_truth.pt or run 00_create_underlying_truth.py first."
    )

true_parameters = torch.load(truth_file, map_location=device, weights_only=False)

true_0 = true_parameters["beta_0"]
true_d = true_parameters["beta_d"]
true_dd = true_parameters["beta_dd"]
true_sigma = np.exp(true_parameters["sigma"])

# -------------------------------
# Model
# -------------------------------


def make_model(loc_dd, scale_dd, loc_sigma, scale_sigma):

    def model(x, y=None, *args, **kwargs):

        dd = get_combinations(x, D, device=device)

        beta_dd = pyro.sample("beta_dd", dist.Laplace(loc_dd, scale_dd).to_event(1))
        sigma = pyro.sample("sigma", dist.LogNormal(loc_sigma, scale_sigma))
        mean = true_0 + x @ true_d + dd @ beta_dd

        with pyro.plate_stack("data", x.shape[:-1]):
            pyro.sample("y", dist.Normal(mean, sigma), obs=y)

    return model


# -------------------------------
# Initialization
# -------------------------------

candidate_designs = create_candidates(D, device=device)

xs = torch.empty(0, D, device=device)
ys = torch.empty(0, device=device)

loc_dd, scale_dd, loc_sigma, scale_sigma = get_prior(D, device=device)
prior_model = make_model(loc_dd, scale_dd, loc_sigma, scale_sigma)


checkpoints = [0, n_rounds // 2, n_rounds - 1]


svi_losses = torch.empty(len(checkpoints), num_steps_SVI, device=device)
SVI_parameters = torch.empty(
    len(checkpoints), num_steps_SVI, 2 * (n_combinations + 1), device=device
)
chosen_index = torch.zeros(n_rounds, device=device)
parameters = torch.empty((n_rounds, 2 * (n_combinations + 1)))

# Initialize old parameters from the prior (for KL computation)
old_locs_kl = {"beta_dd": loc_dd, "sigma": loc_sigma}
old_scales_kl = {"beta_dd": scale_dd, "sigma": scale_sigma}

for experiment in range(n_rounds):
    print(f"Round: {experiment + 1}")

    ig = torch.tensor(-np.inf, device=device)

    for i in range(n_train):
        # -------------------------------
        # Step 1: Run experiment
        # -------------------------------
        y = run_experiment(
            candidate_designs[i], true_0, true_d, true_dd, true_sigma, D, device=device
        )
        xs_tilde = torch.cat([xs, candidate_designs[i].unsqueeze(0)], dim=0)
        ys_tilde = torch.cat([ys, y], dim=0)

        # -------------------------------
        # Step 2: SVI (Learn the posterior using all experiments seen so far)
        # -------------------------------

        pyro.clear_param_store()
        conditioned_model = pyro.condition(prior_model, {"y": ys_tilde})
        auto_guide = AutoNormal(conditioned_model)
        optimizer_SVI = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": start_lr_SVI},
                "gamma": (end_lr_SVI / start_lr_SVI) ** (1 / num_steps_SVI),
            }
        )
        svi = SVI(
            conditioned_model,
            auto_guide,
            optimizer_SVI,
            loss=Trace_ELBO(num_particles=5),
        )
        svi_losses_tilde = torch.empty(num_steps_SVI, device=device)
        svi_parameters_tilde = torch.empty(
            num_steps_SVI, 2 * (n_combinations + 1), device=device
        )
        # Run SVI steps for the current round
        for step in range(num_steps_SVI):
            loss = svi.step(xs_tilde)

            # Only store the loss if this round is a checkpoint
            if experiment in checkpoints:
                # Extract Parameters
                param_store = pyro.get_param_store()
                loc_dd = param_store["AutoNormal.locs.beta_dd"].detach().clone()
                scale_dd = param_store[
                    "AutoNormal.scales.beta_dd"
                ].detach().clone() / math.sqrt(2)
                loc_sigma = param_store["AutoNormal.locs.sigma"].detach().clone()
                scale_sigma = param_store["AutoNormal.scales.sigma"].detach().clone()

                # Save Losses and Parameteters over steps
                svi_losses_tilde[step] = loss
                svi_parameters_tilde[step] = torch.cat(
                    [
                        loc_dd.flatten(),
                        loc_sigma.flatten(),
                        scale_dd.flatten(),
                        scale_sigma.flatten(),
                    ],
                    dim=0,
                )

        # -------------------------------
        # Step 3: Get new parameters
        # -------------------------------
        param_store = pyro.get_param_store()
        parameters_tilde = torch.empty((2 * (n_combinations + 1)))

        loc_dd = param_store["AutoNormal.locs.beta_dd"].detach().clone()
        scale_dd = param_store[
            "AutoNormal.scales.beta_dd"
        ].detach().clone() / math.sqrt(2)

        loc_sigma = param_store["AutoNormal.locs.sigma"].detach().clone()
        scale_sigma = param_store["AutoNormal.scales.sigma"].detach().clone()

        parameters_tilde = torch.cat(
            [
                loc_dd.flatten(),
                loc_sigma.flatten(),
                scale_dd.flatten(),
                scale_sigma.flatten(),
            ],
            dim=0,
        )

        # -------------------------------
        # Step 4: Calculate IG
        # -------------------------------

        new_locs_kl = {"beta_dd": loc_dd, "sigma": loc_sigma}
        new_scales_kl = {"beta_dd": scale_dd, "sigma": scale_sigma}
        kl_value = kl_exact(old_locs_kl, old_scales_kl, new_locs_kl, new_scales_kl)

        if ig < kl_value:
            ig = kl_value

            locs = {"beta_dd": loc_dd.clone(), "sigma": loc_sigma.clone()}
            scales = {"beta_dd": scale_dd.clone(), "sigma": scale_sigma.clone()}

            chosen_index[experiment] = i
            chosen_xs = xs_tilde
            chosen_ys = ys_tilde
            chosen_parameters = parameters_tilde
            if experiment in checkpoints:
                svi_losses[checkpoints.index(experiment)] = svi_losses_tilde
                SVI_parameters[checkpoints.index(experiment)] = svi_parameters_tilde

    # -------------------------------
    # Step 5: Update model
    # -------------------------------
    old_locs_kl = locs
    old_scales_kl = scales
    xs = chosen_xs
    ys = chosen_ys
    parameters[experiment] = chosen_parameters

# -------------------------------
# Step 6: Save results
# -------------------------------

out_dir = os.path.join(_batch_dir or ".", "results_OD")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f"svi_seed_{SEED}.pt")
torch.save(
    {
        "SVI_losses": svi_losses,
        "SVI_parameters": SVI_parameters,
        "chosen_index": chosen_index,
        "parameters": parameters,
        "candidate_designs": candidate_designs,
    },
    out_file,
)

print(f"Saved results to {out_file}")
