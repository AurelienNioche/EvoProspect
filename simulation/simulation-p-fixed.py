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

def play(lottery, agent, n_trial):
    n_lottery, n_agent = len(lottery), len(agent)

    # Choose n_trial random couple of lotteries for each of the n_agent agents
    # index starts at 1 because we don't want the lottery 0 (p=0 and v=+inf)
    L = np.random.randint(1, n_lottery, (n_agent*n_trial, 2))
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


def evolve(lottery, agent, param, score, selection_rate, mutation_rate, mixture_rate):

    sorting = np.argsort(-score)
    param = param[sorting]
    n_agent, n_param = param.shape
    param_ = np.zeros((2, n_agent//2, 2))

    # Select best agents (selection rate)
    size = int(selection_rate*n_agent)
    I = np.random.randint(0, size, (n_agent//2, 2))

    # Create children (mix rate)
    I1, I2 = I[:, 0], I[:, 1]
    A = np.random.uniform(0.0, mixture_rate, (n_agent//2, 1))
    param_[0] = (1-A)*param[I1] + A*param[I2]
    param_[1] = (1-A)*param[I2] + A*param[I1]
    param = param_.reshape(param.shape)

    # Mutation of a few individuals (mutation rate)
    size = int(mutation_rate*n_agent)
    I = np.random.choice(n_agent, size=size, replace=False)

    param[I, 0] = np.random.uniform(pmin, pmax, size)
    param[I, 1] = np.random.uniform(vmin, vmax, size)
    for i in range(len(param)):
        Ap, Av = param[i]
        agent["p"][i] = P_distortion(lottery["p"], Ap)
        agent["v"][i] = V_distortion(lottery["v"], Av)

    return agent, param

def simulation(filename1, filename2):
    # Generate lotteries that can be played
    lottery = np.zeros(n_lottery, dtype=[("p",float), ("v",float)])
    lottery["p"] = np.arange(n_lottery)/(n_lottery-1)
    lottery["v"][1:] = 1/lottery["p"][1:] / (n_lottery-1)
    lottery["v"][0] = 0

    # Generate agents
    Av = np.random.uniform(vmin, vmax, n_agent)
    Ap = np.random.uniform(pmin, pmax, n_agent)
    agent = np.zeros((n_agent, n_lottery), dtype=[("p", float), ("v", float)])
    param = np.zeros((n_agent, 2))
    X = np.arange(0, n_lottery)/(n_lottery-1)
    X[0] = 1e-12
    for i in range(n_agent):
        agent["p"][i] = P_distortion(lottery["p"], Ap[i])
        agent["v"][i] = V_distortion(lottery["v"], Av[i])
        param[i] = Ap[i], Av[i]
    agent_i = agent.copy()
    param_i = param.copy()
    np.save(filename1, param_i)
    for epoch in tqdm.trange(n_epoch):
        score = play(lottery, agent, n_trial)
        score = score.mean(axis=-1)
        agent, param = evolve(lottery, agent, param, score,
                              selection_rate, mutation_rate, mixture_rate)
    agent_f = agent.copy()
    param_f = param.copy()
    np.save(filename2, param_f)


# Set parameters and initialize
data = parameters.default()
data["pmin"] = data["pmax"] = 1.0
parameters.dump(data)
locals().update(data)
np.random.seed(seed)

recompute = False # ! Force recompute or not
dat_filename_1 = "data/p-fixed/simulation-initial-population"
dat_filename_2 = "data/p-fixed/simulation-final-population"
prm_filename = "data/p-fixed/simulation-parameters.json"
fig_filename = "figs/simulation-p-fixed-results.pdf"

if recompute:

    data = parameters.default()
    locals().update(data)
    parameters.save(prm_filename, data)

    data = parameters.default()
    data["pmin"] = data["pmax"] = 1.0
    locals().update(data)
    for selection_rate in np.arange(5,100,5)/100:
        np.random.seed(seed)
        p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
        p = p % (selection_rate, mutation_rate, mixture_rate)
        filename1 = dat_filename_1 + p + ".npy"
        filename2 = dat_filename_2 + p + ".npy"
        print(p)
        simulation(filename1, filename2)

    print()
    data = parameters.default()
    data["pmin"] = data["pmax"] = 1.0
    locals().update(data)
    for mutation_rate in np.array([1,2,3,4,5,10,20,50])/100:
        np.random.seed(seed)
        p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
        p = p % (selection_rate, mutation_rate, mixture_rate)
        filename1 = dat_filename_1 + p + ".npy"
        filename2 = dat_filename_2 + p + ".npy"
        print(p)
        simulation(filename1, filename2)

    print()
    data = parameters.default()
    data["pmin"] = data["pmax"] = 1.0
    locals().update(data)
    for mixture_rate in np.arange(5,50,5)/100:
        np.random.seed(seed)
        p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
        p = p % (selection_rate, mutation_rate, mixture_rate)
        filename1 = dat_filename_1 + p + ".npy"
        filename2 = dat_filename_2 + p + ".npy"
        print(p)
        simulation(filename1, filename2)
        

else:
    data = parameters.default()
    data["pmin"] = data["pmax"] = 1.0
    locals().update(data)
    p = "(selection_rate=%.2f,mutation_rate=%.2f,mixture_rate=%.2f)"
    p = p % (selection_rate, mutation_rate, mixture_rate)
    filename1 = dat_filename_1 + p + ".npy"
    filename2 = dat_filename_2 + p + ".npy"
    param_i = np.load(filename1)
    param_f = np.load(filename2)

    # Evaluate final score
    lottery = np.zeros(n_lottery, dtype=[("p",float), ("v",float)])
    lottery["p"] = np.arange(n_lottery)/(n_lottery-1)
    lottery["v"][1:] = 1/lottery["p"][1:] / (n_lottery-1)
    lottery["v"][0] = 0
    agent = np.zeros((n_agent, n_lottery), dtype=[("p", float), ("v", float)])
    param = np.zeros((n_agent, 2))
    X = np.arange(0, n_lottery)/(n_lottery-1)
    X[0] = 1e-12

    for i in range(n_agent):
        Ap, Av = param_i[i] 
        agent["p"][i] = P_distortion(lottery["p"], Ap)
        agent["v"][i] = V_distortion(lottery["v"], Av)
    score = play(lottery, agent, n_trial)    
    np.savetxt("initial-v-variable-p-linear.csv", score, delimiter=",")

    for i in range(n_agent):
        Ap, Av = param_f[i] 
        agent["p"][i] = P_distortion(lottery["p"], Ap)
        agent["v"][i] = V_distortion(lottery["v"], Av)
    score = play(lottery, agent, n_trial)
    np.savetxt("final-v-variable-p-linear.csv", score, delimiter=",")
    
    import sys
    sys.exit()

    
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Display score
def plot(ax, P, title, label="probabibility", display_mean=False):
    X = np.linspace(0, 1, n_lottery, endpoint=True)
    X[0] = 1e-15
    Y = np.zeros((len(P), len(X)))

    for i in range(len(Y)):
        if label == "value":
            Y[i] = V_distortion(X, P[i])
        else:
            Y[i] = P_distortion(X, P[i])
        ax.plot(X, Y[i], color="C1", lw=.75, alpha=0.1)
    if display_mean:
        if label == "value":
            M = V_distortion(X, np.mean(P,axis=0))
        else:
            M = P_distortion(X, np.mean(P,axis=0))
        ax.plot(X, M, color="k")
    ax.plot(X, X, color="k", lw=.75, ls="--")
    M1, M2 = np.min(Y, axis=0), np.max(Y, axis=0)
    ax.plot(X, M1, color=".25", linewidth=1.0, linestyle="--")
    ax.plot(X, M2, color=".25", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Actual %s" % label)
    ax.set_ylabel("Estimated %s" % label)
    ax.spines['left'].set_bounds(0, 1)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.text(1, 0, title,  weight="bold",
            ha="right", va="top", transform=ax.transData)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plot(axes[0], param_i[:,1], "Initial population", "value")
plot(axes[1], param_f[:,1],
     "Final population", "value", display_mean=True)

plt.savefig(fig_filename)
plt.show()
