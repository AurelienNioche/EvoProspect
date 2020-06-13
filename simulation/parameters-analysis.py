# -----------------------------------------------------------------------------
# An evolutionary perspective on the prospect theory
# Copyright 2020 Nicolas P. Rougier & Aurélien Nioche
# Released under the BSD two-clauses license
# -----------------------------------------------------------------------------
import os
import tqdm
import parameters
import numpy as np


data = parameters.default()
parameters.dump(data)
locals().update(data)
np.random.seed(seed)

dat_filename = "data/population-gain.npy"
prm_filename = "data/population-gain-parameters.json"
fig_filename = "figs/parameters-analysis.pdf"


score = np.load(dat_filename)
score = (score - score.min())/(score.max() - score.min())

# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt


fig = plt.figure(figsize = (6,6))
ax =  plt.subplot(1,1,1)

from scipy.ndimage.filters import gaussian_filter
score = gaussian_filter(score, 1.0)
median = np.median(score)
C = plt.contour(score, levels=[median,],
                extent=[pmin, pmax, vmin, vmax],
                vmin=0, vmax=1, origin="lower", colors="black", zorder=50)
ax.set_title("Parameter sensitivity as mean of final population", weight="bold")
ax.set_xlabel("α (probability weighting function)" )
ax.set_ylabel("β (utility function)")

marker_size = 15

data = parameters.default()
locals().update(data)

for i, selection_rate in enumerate(np.arange(5,100,5)/100):
    p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
    p = p % (selection_rate, mutation_rate, mixture_rate)
    filename = "data/simulation-final-population" + p + ".npy"
    agent_f = np.load(filename)
    X, Y = agent_f[:,0], agent_f[:,1]
    label = None
    if i == 0: label="selection rate"
    ax.scatter([X.mean()], [Y.mean()], s=marker_size, alpha=.5, label=label,
               facecolor="black", edgecolor="white", marker="X", lw=1)
    ax.text(X.mean(), Y.mean()-.015, "%.2f" % selection_rate,
            transform = ax.transData, ha="center", va="top",
            size=4,  color="black", alpha=.5)


data = parameters.default()
locals().update(data)
for i,mutation_rate in enumerate(np.array([1,2,3,4,5,10,20,50])/100):
    p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
    p = p % (selection_rate, mutation_rate, mixture_rate)
    filename = "data/simulation-final-population" + p + ".npy"
    agent_f = np.load(filename)
    X, Y = agent_f[:,0], agent_f[:,1]
    label = None
    if i == 0: label="mutation rate"
    ax.scatter([X.mean()], [Y.mean()], s=marker_size, alpha=.5, label=label,
               facecolor="red", edgecolor="white", marker="o", lw=1)
    ax.text(X.mean(), Y.mean()-.015, "%.2f" % mutation_rate,
            transform = ax.transData, ha="center", va="top",
            size=4,  color="red", alpha=.5)

data = parameters.default()
locals().update(data)
for i,mixture_rate in enumerate(np.arange(5,50,5)/100):
    p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
    p = p % (selection_rate, mutation_rate, mixture_rate)
    filename = "data/simulation-final-population" + p + ".npy"
    agent_f = np.load(filename)
    X, Y = agent_f[:,0], agent_f[:,1]
    label = None
    if i == 0: label="mixture rate"
    ax.scatter([X.mean()], [Y.mean()], s=marker_size, alpha=.5,  label=label,
               facecolor="blue", edgecolor="white", marker="D", lw=1)

    ax.text(X.mean(), Y.mean()-.015, "%.2f" % mixture_rate,
            transform = ax.transData, ha="center", va="top",
            size=4,  color="blue", alpha=.5)

    
ax.axhline(0, lw=0.75, ls="--", color="black")
ax.axvline(1, lw=0.75, ls="--", color="black")
ax.legend(frameon=False)

plt.text(0.01, .75, "risk averse", rotation=90, transform=ax.transAxes,
         ha="left", va="center", fontsize="small")
plt.text(0.01, .25, "risk seeking", rotation=90, transform=ax.transAxes,
         ha="left", va="center", fontsize="small")

plt.text(0.25, 0.01, "low ++ / high --", rotation=0, transform=ax.transAxes,
         ha="center", va="bottom", fontsize="small")
plt.text(0.75, 0.01, "low -- / high ++", rotation=0, transform=ax.transAxes,
         ha="center", va="bottom", fontsize="small")

plt.savefig(fig_filename)
plt.show()
