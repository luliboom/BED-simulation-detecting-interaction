from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace, lognorm

from config import CONFIG


def save_fig(name, **kwargs):
    path = Path(CONFIG["figures_dir"]) / f"{name}.pdf"
    plt.savefig(path, **kwargs)
    print(f"Figure saved to {path}")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.3)

loc_dd, scale_dd = 0, 0.3
dist_dd = laplace(loc_dd, scale_dd)
x_dd = np.linspace(-1.5, 1.5, 1000)
pdf_dd = dist_dd.pdf(x_dd)

loc_sigma, scale_sigma = -1, 0.5
dist_sigma = lognorm(s=scale_sigma, scale=np.exp(loc_sigma))
x_sigma = np.linspace(0.001, 1.5, 1000)
pdf_sigma = dist_sigma.pdf(x_sigma)


axes[0].set_title(r"$\gamma_{i,j}$")
axes[0].plot(x_dd, pdf_dd)
axes[0].set_xlabel(r"$\gamma_{i,j}$")
axes[0].set_ylabel("Density")
axes[0].grid(True, ls="--", alpha=0.3)


axes[1].set_title(r"$\sigma$")
axes[1].plot(x_sigma, pdf_sigma)
axes[1].set_xlabel(r"$\sigma$")
axes[1].set_ylabel("Density")
axes[1].grid(True, ls="--", alpha=0.3)

fig.tight_layout()
save_fig("prior")
