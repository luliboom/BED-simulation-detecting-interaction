# config.py
import os
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent  # up to pyro_project

_batch_dir_env = os.environ.get("BATCH_DIR", "./data")
# create batch directory if it doesn't exist
if _batch_dir_env and not os.path.exists(_batch_dir_env):
    os.makedirs(_batch_dir_env)

if _batch_dir_env:
    _data_dir = Path(_batch_dir_env)
    _figures_dir = _data_dir / "figures"
    _truth_path = _data_dir / "underlying_truth.pt"
else:
    _data_dir = _SCRIPT_DIR
    _figures_dir = _SCRIPT_DIR / "figures"
    _truth_path = _PROJECT_ROOT / "underlying_truth.pt"

CONFIG = {
    "data_dir": str(_data_dir),
    "figures_dir": str(_figures_dir),
    "truth_path": str(_truth_path),
    "colors": {
        "ED": "#ff7f0e",
        "RD": "#1f77b4",
        "PD": "#009E73",
        "UD": "#CC79A7",
        "OD": "#4D4D4D",
    },
    "fontsize": 16,
    "figsize": (7, 5),
    "dpi": 300,
    "grid_alpha": 0.3,
}

plt.rcParams.update(
    {
        "figure.figsize": CONFIG["figsize"],
        "figure.dpi": CONFIG["dpi"],
        "savefig.dpi": CONFIG["dpi"],
        "font.size": CONFIG["fontsize"],
        "axes.labelsize": CONFIG["fontsize"],
        "axes.titlesize": 20,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "grid.alpha": CONFIG["grid_alpha"],
    }
)
