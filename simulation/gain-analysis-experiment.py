# -----------------------------------------------------------------------------
# An evolutionary perspective on the prospect theory
# Copyright 2020 Nicolas P. Rougier & Aurélien Nioche
# Released under the BSD two-clauses license
# -----------------------------------------------------------------------------
import os
import tqdm
import parameters
import numpy as np

def P_distortion(X, alpha, beta=1):
    return np.exp(-beta*(-np.log(X))**alpha)

def V_distortion(X, alpha):
    return X**(1-alpha)

def simulate(lottery, agent, n_trial):
    n_lottery, n_agent = len(lottery), len(agent)

    # Choose n_trial random couple of lotteries for each of the n_agent agents
    # index starts at 1 because we don't want the lottery 0 (p=0 and v=+inf)
    L = np.random.randint(n_lottery, size=(n_agent*n_trial, 2))
    L0, L1 = L[..., 0], L[..., 1]

    # Computing choice for each agent and each lottery
    I = np.repeat(np.arange(n_agent), n_trial)
    U0 = agent["p"][I, L0] * agent["v"][I, L0] * (n_lottery-1)
    U1 = agent["p"][I, L1] * agent["v"][I, L1] * (n_lottery-1)
    U = U0-U1
    C = np.zeros(n_agent*n_trial, dtype=int)
    C[U <= 0] = 1
    L = lottery[L[np.arange(n_agent*n_trial), C]]
    P = np.random.uniform(0, 1, n_agent*n_trial)
    G = ((P < L["p"]) * L["v"]).reshape(n_agent, n_trial)
    return G


def compute(n_lottery, n_agent, n_trial, gridsize, selection_rate):
    # Generate lotteries that can be played

    # lottery["p"][0] = 1e-15
    # lottery["p"][lottery["p"]<1e-15] = 1e-15

    # lottery["v"][1:] = 0.5 / lottery["p"][1:] / (n_lottery - 1)
    # lottery["v"][1:] = np.random.random(n_lottery-1)#(f / lottery["p"][1:]) + np.random.normal(0, f/2, n_lottery-1)
    p = np.linspace(0.2, 1, n_lottery)

    x = 1/P_distortion(p, 1.5)

    # x1 = 1/p**2
    #x2 = 1/p**0.5

    # x = np.array([x1[i] if i % 2 == 0 else x2[i] for i in range(n_lottery)])

    x /= x.max()

    lottery = np.zeros(n_lottery, dtype=[("p", float), ("v", float)])
    lottery["p"] = p
    lottery["v"] = x

    # lottery["v"] = np.random.random(n_lottery)  # lottery["v"][1:] =  1/lottery["p"][1:] / (n_lottery-1)
    # lottery["v"][0] = 0

    # Generate all possible agents and make them play
    agent = np.zeros((n_agent, n_lottery), dtype=[("p", float), ("v", float)]) 
    Ap = np.linspace(pmin, pmax, gridsize)
    Av = np.linspace(vmin, vmax, gridsize)
    score = np.zeros((gridsize, gridsize))
    for i in tqdm.tqdm(range(gridsize)):
        for j in range(gridsize):
            agent["p"] = P_distortion(lottery["p"], Ap[i])
            agent["v"] = V_distortion(lottery["v"], Av[j])
            score_ = simulate(lottery, agent, n_trial)
            score_ = score_.mean(axis=-1)
            score_ = score_[np.argsort(-score_)]
            score[j, i] = score_[:int(selection_rate*n_agent)].mean()
    return score


# Set parameters and initialize
data = parameters.default()
data["seed"] = 12
data["gridsize"] = 50
data["n_lottery"] = 1000
data["n_trial"] = 100
parameters.dump(data)
locals().update(data)
np.random.seed(seed)

recompute = True # ! Force recompute or not
dat_filename = "data/population-gain.npy"
prm_filename = "data/population-gain-parameters.json"
fig_filename = "figs/gain-analysis.pdf"


if recompute:
    score = compute(n_lottery, n_agent, n_trial, gridsize, selection_rate)
    parameters.save(prm_filename, data)
    np.save(dat_filename, score)
else:
    score = np.load(dat_filename)
    data = parameters.load(prm_filename)
    locals().update(data)

# Normalize score
# score = (score - score.min())/(score.max() - score.min())

# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize = (14, 6))
grid = gridspec.GridSpec(ncols=6, nrows=3, figure=fig)

ax =  fig.add_subplot(grid[:3, :3])
im = ax.imshow(score, extent=[pmin, pmax, vmin, vmax],
               rasterized=True, interpolation="none", origin="lower")

from scipy.ndimage.filters import gaussian_filter
score = gaussian_filter(score, 1.0)
median = np.median(score)
C = plt.contour(score, levels=[median,],
                extent=[pmin, pmax, vmin, vmax],
                vmin=100, vmax=140, origin="lower", colors="white", zorder=50)

ax.set_title("Mean gain of the luckiest individuals", weight="bold")

ax.set_xlabel("α (probability function, red)" )
ax.set_ylabel("β (utility function, blue)")
axins = inset_axes(ax,
                   width="3%",  # width = 5% of parent_bbox width
                   height="33%",  # height : 50%
                   loc='upper left',
                   bbox_to_anchor=(1.025, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
cbar = plt.colorbar(im, cax=axins,  orientation="vertical")#, ticks=[0,median,1])
cbar.ax.tick_params(labelsize="small")

ax.axhline(0, lw=0.75, ls="--", color="white")
ax.axvline(1, lw=0.75, ls="--", color="white")
plt.text(0.01, .75, "risk averse", rotation=90, transform=ax.transAxes,
         ha="left", va="center", fontsize="small", color="white", weight="bold")
plt.text(0.01, .25, "risk seeking", rotation=90, transform=ax.transAxes,
         ha="left", va="center", fontsize="small", color="white", weight="bold")

plt.text(0.25, 0.01, "low ++ / high --", rotation=0, transform=ax.transAxes,
         ha="center", va="bottom", fontsize="small", color="white", weight="bold")
plt.text(0.75, 0.01, "low -- / high ++", rotation=0, transform=ax.transAxes,
         ha="center", va="bottom", fontsize="small", color="white", weight="bold")


# Display some representative points
Ap = np.linspace(pmin, pmax, 3+2)[1: -1]
Av = np.linspace(vmin, vmax, 3+2)[1: -1]
index = 0

for i,v in enumerate(Av[::-1]):
    for j,p in enumerate(Ap):
        ax.scatter([p], [v], s=25,
                   facecolor="white", edgecolor="None", marker="o")
        ax.text(p, v, "\n"+chr(ord('A') + index), transform=ax.transData,
               va = "top", ha="center", color="white", weight="bold")

        ax2 =  fig.add_subplot(grid[i, 3+j], frameon=False)
        ax2.text(0, 1, chr(ord('A') + index), transform=ax2.transAxes,
                va = "top", ha="left", color="black", weight="bold")
        ax2.set_xlim([0,1]), ax2.set_xticks([])
        ax2.set_ylim([0,1]), ax2.set_yticks([])

        X = np.linspace(0,1,100)
        X[0] = 1e-15

        ax2.plot(X, X, color=".5",  lw=1, ls = "--")
        ax2.plot(X, P_distortion(X, p),
                 color="red",  lw=1.5, alpha=.5, clip_on=False)
        ax2.plot(X, V_distortion(X, v),
                 color="blue", lw=1.5, alpha=.5, clip_on=False)
        index += 1


plt.savefig(fig_filename)
plt.show()
