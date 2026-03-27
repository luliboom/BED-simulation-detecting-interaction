import argparse
import math
import os
import random

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

import config
from functions import (
    create_candidates,
    get_combinations,
    get_prior,
    next_candidate,
    return_all,
    run_experiment,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Parse command-line arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Euler batch script with parameters")
parser.add_argument(
    "--seed", type=int, default=1, help="Random seed for reproducibility"
)
parser.add_argument("--n_rounds", type=int, default=1, help="Number of rounds")
parser.add_argument(
    "--start_lr_SVI", type=float, default=1e-03, help="start learning rate"
)
parser.add_argument("--end_lr_SVI", type=float, default=1e-06, help="end learning rate")
parser.add_argument("--num_steps_SVI", type=int, default=10000, help="Number of steps")
parser.add_argument("--procedure", type=str, default="ED", help="ED, PD, RD or UD")
args = parser.parse_args()

# -------------------------------
# Initialization
# -------------------------------
D = 4
n_combinations = D * (D - 1) // 2
n_train = 1 + D + n_combinations


n_rounds = args.n_rounds
num_steps_SVI = args.num_steps_SVI
start_lr_SVI = args.start_lr_SVI
end_lr_SVI = args.end_lr_SVI
procedure = args.procedure

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

        dd = get_combinations(x, D)

        beta_dd = pyro.sample("beta_dd", dist.Laplace(loc_dd, scale_dd).to_event(1))
        sigma = pyro.sample("sigma", dist.LogNormal(loc_sigma, scale_sigma))
        mean = true_0 + x @ true_d + dd @ beta_dd

        with pyro.plate_stack("data", x.shape[:-1]):
            pyro.sample("y", dist.Normal(mean, sigma), obs=y)

    return model


# -------------------------------
# Marginal Guide
# -------------------------------


def marginal_guide(design, observation_labels, target_labels):
    q_mean = pyro.param("q_mean", torch.zeros(design.shape[-2], device=device))
    q_sigma = pyro.param(
        "q_sigma",
        torch.ones(design.shape[-2], device=device),
        constraint=constraints.interval(0.01, 100.0),
    )

    pyro.sample("y", dist.Normal(q_mean, q_sigma))


# -------------------------------
# Run SVI
# -------------------------------

# Initialization
candidate_designs = create_candidates(D, device=device)

xs = torch.empty(0, D, device=device)
ys = torch.empty(0, device=device)

# Prior Model
loc_dd, scale_dd, loc_sigma, scale_sigma = get_prior(D, device=device)
prior_model = make_model(loc_dd, scale_dd, loc_sigma, scale_sigma)
current_model = prior_model

checkpoints = [0, n_rounds // 2, n_rounds - 1]
order = torch.randperm(n_combinations)


# Initialize Tensors for plotting
SVI_losses = torch.empty(len(checkpoints), num_steps_SVI, device=device)
SVI_parameters = torch.empty(
    len(checkpoints), num_steps_SVI, 2 * (n_combinations + 1), device=device
)
chosen_index = torch.zeros(n_rounds, device=device)
parameters = torch.empty((n_rounds, 2 * (n_combinations + 1)))


for experiment in range(n_rounds):
    print(f"Round: {experiment + 1}")

    # -------------------------------
    # Step 1: Chose next experiment
    # -------------------------------
    x_index, x_next = next_candidate(
        procedure,
        current_model,
        marginal_guide,
        candidate_designs,
        n_train,
        optimizer_EIG,
        num_steps_EIG,
        order,
        experiment,
        D,
        device=device,
    )
    chosen_index[experiment] = x_index

    # -------------------------------
    # Step 2: Run the next experiment
    # -------------------------------
    y = run_experiment(x_next, true_0, true_d, true_dd, true_sigma, D, device=device)

    xs = torch.cat([xs, x_next.unsqueeze(0)], dim=0)
    ys = torch.cat([ys, y], dim=0)

    # -------------------------------
    # Step 3: SVI: Learn Posterior
    # -------------------------------

    pyro.clear_param_store()
    conditioned_model = pyro.condition(prior_model, {"y": ys})
    auto_guide = AutoNormal(conditioned_model)
    optimizer_SVI = pyro.optim.ExponentialLR(
        {
            "optimizer": torch.optim.Adam,
            "optim_args": {"lr": start_lr_SVI},
            "gamma": (end_lr_SVI / start_lr_SVI) ** (1 / num_steps_SVI),
        }
    )

    svi = SVI(
        conditioned_model, auto_guide, optimizer_SVI, loss=Trace_ELBO(num_particles=5)
    )

    # Check if the current round is a checkpoint
    if experiment in checkpoints:
        checkpoint_idx = checkpoints.index(experiment)

    # Run SVI steps for the current round
    for i in range(num_steps_SVI):
        loss = svi.step(xs)

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
            SVI_losses[checkpoint_idx, i] = loss
            SVI_parameters[checkpoint_idx, i] = torch.cat(
                [
                    loc_dd.flatten(),
                    loc_sigma.flatten(),
                    scale_dd.flatten(),
                    scale_sigma.flatten(),
                ],
                dim=0,
            )

    # -------------------------------
    # Step 4: Get new parameters
    # -------------------------------

    param_store = pyro.get_param_store()

    loc_dd = param_store["AutoNormal.locs.beta_dd"].detach().clone()
    scale_dd = param_store["AutoNormal.scales.beta_dd"].detach().clone() / math.sqrt(2)

    loc_sigma = param_store["AutoNormal.locs.sigma"].detach().clone()
    scale_sigma = param_store["AutoNormal.scales.sigma"].detach().clone()

    parameters[experiment] = torch.cat(
        [
            loc_dd.flatten(),
            loc_sigma.flatten(),
            scale_dd.flatten(),
            scale_sigma.flatten(),
        ],
        dim=0,
    )

    # -------------------------------
    # Step 5: Update model
    # -------------------------------

    current_model = make_model(loc_dd, scale_dd, loc_sigma, scale_sigma)


# -------------------------------
# Save Results
# -------------------------------
return_all(
    procedure,
    SVI_losses,
    SVI_parameters,
    chosen_index,
    parameters,
    candidate_designs,
    SEED,
    base_dir=_batch_dir,
)
