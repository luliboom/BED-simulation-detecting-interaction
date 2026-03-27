import torch
import pyro
from pyro.contrib.oed.eig import marginal_eig
import pyro.distributions as dist
from itertools import combinations
import os


# -------------------------------
# Prior Informations
# -------------------------------

def get_prior(D, device="cpu"):
    loc_dd = torch.zeros(int(D*(D-1)/2), device=device)
    scale_dd = 0.3*torch.ones(int(D*(D-1)/2), device=device)
    loc_sigma = torch.tensor([-1.0], device=device) #-1 for 0.4
    scale_sigma = 0.5 * torch.tensor([1.0], device=device)

    return loc_dd, scale_dd, loc_sigma, scale_sigma

# -------------------------------
# Get Combinations 
# -------------------------------

def get_combinations(d, D, device="cpu"):

    i, j = torch.triu_indices(D, D, offset=1, device=device) # All possible pairs D drugs
    dd = d[..., i] * d[..., j]

    return dd

# -------------------------------
# Run Experiment
# -------------------------------

def run_experiment(x, beta_0, beta_d, beta_dd, sigma, D, device="cpu"):
    dd = get_combinations(x, D)
    mean = beta_0 + x @ beta_d + dd @ beta_dd
    epsilon = torch.randn_like(mean, device=device) * sigma
    y = mean + epsilon 
    return y

# -------------------------------
# Create Candidates
# -------------------------------

def create_candidates(D, device='cpu'):
    n_candidates = 1 + D + D*(D-1)//2
    d_tensor = torch.zeros((n_candidates, D), device=device)

    empty = [()]
    singles = [(i,) for i in range(D)]
    pairs = list(combinations(range(D), 2))
    all_candidates = empty + singles + pairs

    for i, candidate in enumerate(all_candidates):
        if len(candidate) > 0:
            d_tensor[i, list(candidate)] = 1

    return d_tensor



# -------------------------------
# Sample from Marginal
# -------------------------------

def sample_marginal_guide(guide, design, observation_labels, target_labels, num_samples=10000):
    guide_samples = []
    for _ in range(num_samples):
        q_trace = pyro.poutine.trace(guide).get_trace(design, observation_labels, target_labels)
        y_vals = torch.stack([q_trace.nodes[l]['value'] for l in observation_labels], dim=0)
        guide_samples.append(y_vals)
    guide_samples = torch.stack(guide_samples).squeeze()
    return guide_samples

def sample_true_marginal(model, design, observation_labels, num_samples= 10000):
    true_samples = []
    for _ in range(num_samples):
        trace = pyro.poutine.trace(model).get_trace(design)
        y_vals = torch.stack([trace.nodes[l]['value'] for l in observation_labels], dim=0)
        true_samples.append(y_vals)
    true_samples = torch.stack(true_samples).squeeze()
    return true_samples
           

# -------------------------------
# Sample next candidate
# -------------------------------

def next_candidate(procedure, model, marginal_guide, candidate_designs, n_candidates, optimizer, num_steps, order, experiment, D, target_labels=None, device="cpu"):

    if target_labels is None:
        target_labels = ["beta_dd", "sigma"]

    if procedure == "ED":
        eig = marginal_eig(model,
                           candidate_designs,
                           ["y"],
                           target_labels,
                           num_samples=10000,
                           num_steps = num_steps,
                           guide = marginal_guide,
                           optim = optimizer,
                           return_history = False,
                           final_num_samples=10000)

        x_index = torch.argmax(eig)
        
    elif procedure == "PD":
        x_index = order[experiment%(D*(D-1)//2)] + D + 1

    elif procedure == "RD":
        x_index = torch.randint(D + 1, n_candidates, (), device=device)
    
    else:
        x_index = torch.randint(0, n_candidates, (), device=device)
        
    x_next = candidate_designs[x_index]  # [D]

    return x_index, x_next



# -------------------------------
# Calculate IG
# -------------------------------

def kl_exact(old_locs, old_scales, new_locs, new_scales):
    kl_total = torch.tensor(0.0)
    for site in old_locs:
        if site == "sigma":
            d = dist.Normal
        else:
            d = dist.Laplace
        kl_total = kl_total + torch.distributions.kl_divergence(
            d(new_locs[site], new_scales[site]),
            d(old_locs[site], old_scales[site])
        ).sum()
    return kl_total

# -------------------------------
# Return for plotting
# -------------------------------

def return_all(procedure, SVI_losses, SVI_parameters, chosen_index, parameters, candidate_designs, SEED, base_dir="."):
    out_dir = os.path.join(base_dir, f"results_{procedure}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"svi_seed_{SEED}.pt")
    torch.save({"SVI_losses": SVI_losses,
                "SVI_parameters": SVI_parameters,
                "chosen_index": chosen_index,
                "parameters": parameters,
                "candidate_designs": candidate_designs},
                out_file)
    print(f"saved {out_file}")


# -------------------------------
# Find interactive drugs
# -------------------------------

def find_interactive_drugs(D, part_add):
    interactions = []
    idx = 0
    for i in range(D):
        for j in range(i+1, D):
            if part_add[idx] == 1:
                interactions.append((i,j))
            idx += 1
    return interactions