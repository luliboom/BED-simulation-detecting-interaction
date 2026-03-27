# BED Simulation: Detecting Interactions

This is a simulation study for **Bayesian experimental design (ED)** to detect drug interactions. The project compares ED to four baseline strategies (UD, RD, PD, OD) for sequentially selecting drug combination experiments, using Pyro for Bayesian Experimental Design (ED) and Stochastic Variational Inference (SVI). 

## Installation

Requires Python 3.13.2. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

All subsequent commands use `uv run`, so no manual environment activation is needed.

## Running the Pipeline

Scripts are meant to be run in order. They can be run both locally or on a Server. 

### Locally

This is only used for testing the code. The typical workflow is:

```bash
# Step 0: Set Batch directory and generate ground truth parameters (run once)
export BATCH_DIR=/path/to/output/dir
uv run 00_create_underlying_truth.py

# Step 1.1: Run sequential SVI: Experimental design (ED), Uniform design (UD), Restricted design (RD), Permutation design (PD), Optimal design (OD)
uv run 01_SVI.py --seed 1 --n_rounds 1 --procedure ED --num_steps_SVI 10
uv run 01_SVI.py --seed 1 --n_rounds 1 --procedure UD --num_steps_SVI 10 
uv run 01_SVI.py --seed 1 --n_rounds 1 --procedure RD --num_steps_SVI 10 
uv run 01_SVI.py --seed 1 --n_rounds 1 --procedure PD --num_steps_SVI 10 
uv run 01_SVI_OD.py --seed 1 --n_rounds 1 --num_steps_SVI 10

# Step 2: Merge results across seeds
uv run 02_merge_files.py        # RD
uv run 02_merge_files.py --ED   # ED
uv run 02_merge_files.py --UD   # UD
uv run 02_merge_files.py --PD   # PD
uv run 02_merge_files.py --OD   # OD

# Step 3: Generate all analysis plots
uv run 03_plotting.py

# Utility: Visualize prior distributions
uv run prior_visualization.py
```

### Key CLI arguments

Both `01_SVI.py` and `01_SVI_OD.py` share:
- `--seed`: random seed (1–1000 in batch runs)
- `--n_rounds`: number of sequential experimental rounds
- `--num_steps_SVI`: SVI optimisation steps per round
- `--start_lr_SVI` / `--end_lr_SVI`: learning rate schedule for SVI

`01_SVI.py` only:
- `--procedure`: `ED` (experimental design), `UD` (uniform random), `RD` (restricted design), `PD` (permutation design)

### Running on the ETH Euler Cluster (SLURM)

The script [python_residuals_evaluation.sh](python_residuals_evaluation.sh) submits a SLURM job array over 1000 seeds. It uses `uv run` directly, so no manual environment activation is needed.

**Step 0: Set the output directory** via `BATCH_DIR` and generate the ground truth (run once):

```bash
export BATCH_DIR=/path/to/output/dir
uv run 00_create_underlying_truth.py
```

**Step 1: Submit a procedure** (e.g. ED, UD, RD, PD, OD):

```bash
DESIGN=ED sbatch python_residuals_evaluation.sh
DESIGN=UD sbatch python_residuals_evaluation.sh
DESIGN=RD sbatch python_residuals_evaluation.sh
DESIGN=PD sbatch python_residuals_evaluation.sh
OD=1 sbatch python_residuals_evaluation.sh
```


Job output logs go to `res_outfiles/res_<jobid>_<seed>.out/err`. 

**Step 2–3:** After all jobs complete, run locally or on the cluster:

```bash
uv run 02_merge_files.py   # merges all procedures found in BATCH_DIR
uv run 03_plotting.py
```

## Architecture

### Model

The study models a drug combination experiment with `D=4` drugs. The response is:

```
y = beta_0 + x @ beta_d + dd @ beta_dd + epsilon
```

where `x` is a `D`-dimensional vector indicating which drugs are applied, `dd` contains all pairwise product terms, and `epsilon ~ Normal(0, sigma)`. The unknown parameters are `beta_dd` (interaction effects, Laplace prior) and `sigma` (noise, LogNormal prior). `beta_0` and `beta_d` are treated as known from the ground truth.

### Candidate designs

There are `1 + D + D*(D-1)/2 = 11` candidates (no drug, each single drug, each pair). Created by `create_candidates()` in [functions.py](functions.py).

### Procedure comparison

| Procedure | Selection strategy |
|---|---|
| ED | Experimental Design: Maximize Expected Information Gain (EIG) via `marginal_eig` from `pyro.contrib.oed` |
| UD | Uniform Design: Uniformly chosoes candidates across all types |
| RD | Restricted Design: Uniformly chooses candidates restricted to type combination |
| PD | Permuted Design: Permute candidates of type combination, fix order and select candidates along the fixed order |
| OD | Optimal Design: Tries all candidates, picks the one maximizing Information gain (KL divergence) |


### Additional files

- [functions.py](functions.py): shared utilities — prior setup, candidate generation, experiment simulation, EIG-based candidate selection, KL divergence, result saving
- [config.py](config.py): centralised paths and matplotlib defaults; imported by plotting scripts
- [plotting_functions.py](plotting_functions.py): all plot implementations called by `03_plotting.py`

