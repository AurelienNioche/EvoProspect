# -----------------------------------------------------------------------------
# An evolutionary perspective on the prospect theory
# Copyright 2020 Nicolas P. Rougier & Aurélien Nioche
# Released under the BSD two-clauses license
# -----------------------------------------------------------------------------
import os
import json
import time
import subprocess
import parameters
import numpy as np

# Se parameters and initialize
data = parameters.default()
parameters.dump(data)
locals().update(data)
np.random.seed(seed)

dat_filename = "gain-analysis.npy"
prm_filename = "gain-analysis.json"
fig_filename = "monkey-agent-comparison.pdf"

if not os.path.exists(dat_filename):
    raise Exception("Datafile not found. Please run 'gain-analysis.py'")

# Load and normalize score
score = np.load(dat_filename)
data = parameters.load(prm_filename)
locals().update(data)
score = (score - score.min())/(score.max() - score.min())


# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (12, 6))
ax =  plt.subplot(1,2,1, aspect=1)

from scipy.ndimage.filters import gaussian_filter
gscore = gaussian_filter(score, 1.0)
median = np.median(score)
C = ax.contour(gscore, levels=[median,],
                extent=[pmin, pmax, vmin, vmax], 
                vmin=100, vmax=140, origin="lower", colors="0.25", zorder=50)
ax.set_title("Monkeys fitted behavior", weight="bold")
ax.set_xlabel("Probability distortion")
ax.set_ylabel("Value distortion")

# Display monkeys
monkeys = np.genfromtxt('monkeys.csv', delimiter=',', dtype=None, names=True)
X, Y = monkeys["gaindistortion"], monkeys["gainrisk_aversion"]
ax.scatter(X, Y, s=25, facecolor="blue",
           edgecolor="blue", marker="o", label="gain")
ax.scatter([X.mean()], [Y.mean()], s=75, facecolor="blue",
           edgecolor="white", marker="X", lw=1)

for i, label in enumerate(monkeys["name"]):
    ax.text(X[i]-0.02, Y[i],
             label.decode(), size="small",
            va = "center", ha="right", color="blue")

X, Y = monkeys["lossdistortion"], monkeys["lossrisk_aversion"]
ax.scatter(X, Y, s=25, facecolor="red",
           edgecolor="red", marker="o", label="loss")
ax.scatter([X.mean()], [Y.mean()], s=75, facecolor="red",
           edgecolor="white", marker="X", lw=1)
for i, label in enumerate(monkeys["name"]):
    label = label.decode()
    dy = 0.0 if label != "Yoh" else 0.025
    ax.text(X[i]-0.02, Y[i]+dy, label, size="small",
            va = "center", ha="right", color="red")
ax.legend(frameon=False)
ax.set_xlim(pmin, pmax), ax.set_ylim(vmin, vmax)


ax = plt.subplot(1,2,2, aspect=1)
agent_i = np.load("agent-simulation-initial.npy")
agent_f = np.load("agent-simulation-final.npy")
from scipy.ndimage.filters import gaussian_filter
gscore = gaussian_filter(score, 1.0)
median = np.median(score)
C = ax.contour(gscore, levels=[median,],
                extent=[pmin, pmax, vmin, vmax],
                vmin=100, vmax=140, origin="lower", colors="0.25", zorder=50)
ax.set_title("Agents' behavior", weight="bold")
ax.set_xlabel("Probability distortion")

# Display agents
X, Y = agent_i[:,0], agent_i[:,1]
ax.scatter(X, Y, s=10, facecolor="black", edgecolor="None", marker="o",
           label="initial", alpha=.125)

X, Y = agent_f[:,0], agent_f[:,1]
ax.scatter(X, Y, s=30, facecolor="black", edgecolor="None", marker="o")
ax.scatter(X, Y, s=20, facecolor="white", edgecolor="None", marker="o")
ax.scatter(X, Y, s=20, facecolor=".5", edgecolor="None",
           marker="o", label="final", alpha=.25)

ax.scatter([X.mean()], [Y.mean()], s=75,
           facecolor="black", edgecolor="white", marker="X", lw=1)
X, Y = monkeys["gaindistortion"], monkeys["gainrisk_aversion"]
ax.scatter([X.mean()], [Y.mean()], s=75,
           facecolor="blue", edgecolor="white", marker="X", lw=1)
X, Y = monkeys["lossdistortion"], monkeys["lossrisk_aversion"]
ax.scatter([X.mean()], [Y.mean()], s=75,
           facecolor="red", edgecolor="white", marker="X", lw=1)
ax.legend(frameon=True, loc="upper right")
ax.set_xlim(pmin, pmax), ax.set_ylim(vmin, vmax)
ax.set_yticks([])

plt.savefig(fig_filename)
plt.show()
