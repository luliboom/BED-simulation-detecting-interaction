import argparse
import os

import torch

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parser = argparse.ArgumentParser(description="Merger")
parser.add_argument("--ED", action="store_true", help="Use experimental design")
parser.add_argument("--PD", action="store_true", help="Use permutation design")
parser.add_argument("--UD", action="store_true", help="Use uniform design")
parser.add_argument("--OD", action="store_true", help="Use optimal design")
args = parser.parse_args()
ED = args.ED
PD = args.PD
UD = args.UD
OD = args.OD

base_dir = config.CONFIG["data_dir"]

print(f"base_dir: {base_dir}")

if ED:
    procedure = "ED"
elif PD:
    procedure = "PD"
elif UD:
    procedure = "UD"
elif OD:
    procedure = "OD"
else:
    procedure = "RD"

data_dir = os.path.join(base_dir, f"results_{procedure}")


results_list = []

seedfiles = [
    f for f in os.listdir(data_dir) if f.startswith("svi_seed_") and f.endswith(".pt")
]
print(f"Found {len(seedfiles)} seed files in {data_dir}")
for seedfile in seedfiles:
    file_path = os.path.join(data_dir, seedfile)
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        try:
            data = torch.load(file_path, map_location=device)
            results_list.append(data)
        except Exception as e:
            print(f"Skipping corrupted file: {file_path} ({e})")
    else:
        print(f"File not found: {file_path}")

# Merge dictionaries
merged_results = {}
for key in results_list[0].keys():
    first_val = results_list[0][key]

    # Case 1: plain tensor → stack
    if torch.is_tensor(first_val):
        merged_results[key] = torch.stack([res[key] for res in results_list], dim=0)

    # Case 2: dict of tensors (e.g. parameters)
    elif isinstance(first_val, dict):
        merged_results[key] = {}
        for subkey in first_val.keys():
            merged_results[key][subkey] = torch.stack(
                [res[key][subkey] for res in results_list], dim=0
            )

# Check shapes
for k, v in merged_results.items():
    if torch.is_tensor(v):
        print(f"{k}: {v.shape}")
    elif isinstance(v, dict):
        for sk, sv in v.items():
            print(f"{k}.{sk}: {sv.shape}")
    else:
        print(f"{k}: list of length {len(v)}")

output_file = os.path.join(base_dir, f"svi_merged_{procedure}.pt")
torch.save(merged_results, output_file)
print(f"Merged results saved to {output_file}")
