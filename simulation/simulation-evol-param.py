# -----------------------------------------------------------------------------
# An evolutionary perspective on the prospect theory
# Copyright 2020 Nicolas P. Rougier & Aurélien Nioche
# Released under the BSD two-clauses license
# -----------------------------------------------------------------------------
import os
import tqdm
import parameters
import numpy as np
from string import ascii_lowercase


def P_distortion(X, alpha, beta=1):
    return np.exp(-beta * (-np.log(X)) ** alpha)


def V_distortion(X, alpha):
    return X ** (1 - alpha)


def play(lottery, agent, n_trial):
    n_lottery, n_agent = len(lottery), len(agent)

    # Choose n_trial random couple of lotteries for each of the n_agent agents
    # index starts at 1 because we don't want the lottery 0 (p=0 and v=+inf)
    L = np.random.randint(1, n_lottery, (n_agent * n_trial, 2))
    L0, L1 = L[..., 0], L[..., 1]

    # Computing choice for each agent and each lottery
    I = np.repeat(np.arange(n_agent), n_trial)
    U0 = agent["p"][I, L0] * agent["v"][I, L0] * (n_lottery - 1)
    U1 = agent["p"][I, L1] * agent["v"][I, L1] * (n_lottery - 1)
    U = U0 - U1
    C = np.zeros(n_agent * n_trial, dtype=int)
    C[U <= 0] = 1
    L = lottery[L[np.arange(n_agent * n_trial), C]]
    P = np.random.uniform(0, 1, n_agent * n_trial)
    G = ((P < L["p"]) * L["v"]).reshape(n_agent, n_trial)
    return G


def evolve(lottery, agent, param, score, selection_rate, mutation_rate,
           mixture_rate):
    sorting = np.argsort(-score)
    param = param[sorting]
    n_agent, n_param = param.shape
    param_ = np.zeros((2, n_agent // 2, 2))

    # Select best agents (selection rate)
    size = int(selection_rate * n_agent)
    I = np.random.randint(0, size, (n_agent // 2, 2))

    # Create children (mix rate)
    I1, I2 = I[:, 0], I[:, 1]
    A = np.random.uniform(0.0, mixture_rate, (n_agent // 2, 1))
    param_[0] = (1 - A) * param[I1] + A * param[I2]
    param_[1] = (1 - A) * param[I2] + A * param[I1]
    param = param_.reshape(param.shape)

    # Mutation of a few individuals (mutation rate)
    size = int(mutation_rate * n_agent)
    I = np.random.choice(n_agent, size=size, replace=False)

    param[I, 0] = np.random.uniform(pmin, pmax, size)
    param[I, 1] = np.random.uniform(vmin, vmax, size)
    for i in range(len(param)):
        Ap, Av = param[i]
        agent["p"][i] = P_distortion(lottery["p"], Ap)
        agent["v"][i] = V_distortion(lottery["v"], Av)

    return agent, param


def simulation(filename):
    # Generate lotteries that can be played
    lottery = np.zeros(n_lottery, dtype=[("p", float), ("v", float)])
    lottery["p"] = np.arange(n_lottery) / (n_lottery - 1)
    lottery["v"][1:] = 1 / lottery["p"][1:] / (n_lottery - 1)
    lottery["v"][0] = 0

    # Generate agents
    if 'dist' not in globals() or dist == "uniform":
        print("using uniform dist")
        Av = np.random.uniform(vmin, vmax, n_agent)
        Ap = np.random.uniform(pmin, pmax, n_agent)
    elif dist == "normal":
        print("using normal dist")
        Av = np.random.normal(loc=np.mean([vmax, vmin]),
                              scale=0.10 * (vmax - vmin),
                              size=n_agent)
        Ap = np.random.normal(loc=np.mean([pmin, pmax]),
                              scale=0.10 * (pmax - pmin),
                              size=n_agent)
    else:
        raise ValueError

    agent = np.zeros((n_agent, n_lottery), dtype=[("p", float), ("v", float)])
    param = np.zeros((n_agent, 2))
    X = np.arange(0, n_lottery) / (n_lottery - 1)
    X[0] = 1e-12
    for i in range(n_agent):
        agent["p"][i] = P_distortion(lottery["p"], Ap[i])
        agent["v"][i] = V_distortion(lottery["v"], Av[i])
        param[i] = Ap[i], Av[i]

    param_mean = np.zeros((n_epoch+1, 2))
    param_std = np.zeros((n_epoch+1, 2))
    gain_mean = np.zeros(n_epoch+1)
    gain_std = np.zeros(n_epoch+1)

    param_mean[0] = np.mean(param, axis=0)
    param_std[0] = np.std(param, axis=0)

    for epoch in tqdm.trange(n_epoch):
        score = play(lottery, agent, n_trial)
        score = score.mean(axis=-1)
        agent, param = evolve(lottery, agent, param, score,
                              selection_rate, mutation_rate, mixture_rate)

        param_mean[epoch+1] = np.mean(param, axis=0)
        param_std[epoch+1] = np.std(param, axis=0)

        size = int(selection_rate * n_agent)
        best_score = np.argsort(-score)[:size]

        gain_mean[epoch+1] = np.mean(best_score)
        gain_std[epoch+1] = np.std(best_score)

    data_to_save = {
        "param_mean": param_mean,
        "param_std": param_std,
        "gain_mean": gain_mean,
        "gain_std": gain_std
    }

    np.savez(filename, **data_to_save)


# Set parameters and initialize
data = parameters.default()
parameters.dump(data)
locals().update(data)
np.random.seed(seed)

recompute = False  # ! Force recompute or not
dat_filename = f"data/simulation-evol-param.npz"
prm_filename = f"data/simulation-parameters-evol-param.json"
fig_filename = f"figs/simulation-evol-param.png"

if recompute:

    parameters.save(prm_filename, data)

    simulation(dat_filename)
    loaded_data = np.load(dat_filename)

else:
    loaded_data = np.load(dat_filename)


print(loaded_data.keys())

param_mean = loaded_data['param_mean']
param_std = loaded_data["param_std"]
gain_mean = loaded_data["gain_mean"]
gain_std = loaded_data["gain_std"]

print(gain_mean)

# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt


def plot(ax, x_mean, x_std, xlabel, ylabel, idx_letter=None):

    ax.plot(np.arange(len(x_mean)), x_mean)
    ax.fill_between(np.arange(len(x_mean)), x_mean-x_std, x_mean+x_std,
                    alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.text(-0.1, 1.1, ascii_lowercase[idx_letter],
            transform=ax.transAxes, size=20, weight='bold')


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
plot(axes[0], param_mean[:, 0], param_std[:, 0], "epoch", r"$\alpha$-value",
     idx_letter=0)
plot(axes[1], param_mean[:, 1], param_std[:, 1], "epoch", r"$\beta$-value",
     idx_letter=1)
plot(axes[2], gain_mean[:], gain_std[:], "epoch", "mean gain of 20% best scorers",
     idx_letter=2)

plt.tight_layout()
plt.savefig(fig_filename, dpi=300)
plt.show()
