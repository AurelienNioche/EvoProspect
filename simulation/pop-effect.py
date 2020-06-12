import numpy as np
import matplotlib.pyplot as plt

n_trial = 1000
n_agent = 1000
ratio = 0.75
n_best = int(ratio*n_agent)

G1 = np.zeros((n_agent, n_trial))
G2 = np.zeros((len(G1), n_trial))

P1 = np.random.uniform(0.01, 0.50, (n_agent, n_trial))
X1 = 1/P1
G1 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P1) * X1
print(np.mean(G1))
G1 = G1[np.argsort(-np.sum(G1,-1))]
G1 = G1[:n_best]

P2 = np.random.uniform(0.01, 1.0, (n_agent, n_trial))
X2 = 1/P2
G2 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P2) * X2
print(np.mean(G2))
G2 = G2[np.argsort(-np.sum(G2,-1))]
G2 = G2[:n_best]

for i in range(len(G1)):
    G1[i] = np.cumsum(G1[i])/np.arange(1,n_trial+1)
    G2[i] = np.cumsum(G2[i])/np.arange(1,n_trial+1)

plt.plot(np.mean(G1[:,10:], axis=0), color="red", lw=1.5)
plt.plot(np.mean(G2[:,10:], axis=0), color="blue", lw=1.5)
plt.show()