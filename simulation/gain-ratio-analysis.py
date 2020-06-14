import tqdm
import numpy as np
import matplotlib.pyplot as plt

def P_distortion(X, alpha): return np.exp(-(-np.log(X))**alpha)
def V_distortion(X, alpha): return X**(1-alpha)


# np.random.seed(123)
n_trial = 1000
n_agent = 10000
ratio = 0.20
n_best = int(ratio*n_agent)
start = 10

P = np.zeros((2, n_agent, n_trial))
P[0] = np.random.uniform(0.001, 1.0, (n_agent, n_trial))
P[1] = np.random.uniform(0.001, P[0], (n_agent, n_trial))

X = np.zeros((2, n_agent, n_trial))
X[0] = 1/P[0]
X[1] = 1/P[1]

G0 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P[0]) * X[0]
G1 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P[1]) * X[1]
G0_ = []
G1_ = []
for i in tqdm.trange(start,n_trial):
    g0 = ((G0[np.argsort(-np.sum(G0[:,:i],axis=-1))])[:n_best,:i]).mean()
    g1 = ((G1[np.argsort(-np.sum(G1[:,:i],axis=-1))])[:n_best,:i]).mean()
    G0_.append(g0)
    G1_.append(g1)

fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('axes', -0.05))
ax.spines['bottom'].set_position(('axes', -0.05))

ax.set_ylim(1.0, 2.5)
ax.set_ylabel("mean gain")
ax.set_xlim(start, n_trial)
ax.set_xlabel("number of trials")
X = np.arange(start, n_trial)

plt.plot(X, G0_, color='0.5', label="risk seeking")
plt.plot(X, G1_, color='0.0', label="risk averse")

def plot_ratio(x, y0, y1):
    ratio = y1/y0
    plt.scatter([x,x], [y0, y1], s=25, edgecolor="black", facecolor="white",
                lw=.5, zorder=50, clip_on=False)
    plt.plot([x,x], [y0,1.75], color="black", zorder=25, clip_on=False, lw=.75, ls="--")
    plt.text(x, 1.80, "%.2f" % ratio, va="bottom", ha="center", clip_on=False)

xticks = [start, 100, 200, 500, 1000]
plt.xticks(xticks)
for i in xticks[1:]:
    i = i-1-start
    plot_ratio(X[i], G0_[i], G1_[i])
plt.legend(frameon=False)
plt.title("Mean gain of 20% best scorers as a function of the number of trials")
    
plt.savefig("gain-ratio-analysis.pdf")
plt.savefig("gain-ratio-analysis.png", dpi=300)
plt.show()



#X[0] = np.random.uniform(0.0, 1.0, (n_agent, n_trial))
#X[1] = np.random.uniform(0.0, 1.0, (n_agent, n_trial))


# print("Player 1: play low probabilities")
# G = (np.random.uniform(0, 1, (n_agent,n_trial)) < P[0]) * X[0]
# print("Mean gain (all):", G.mean())
# print("Mean gain (best):", 
#       ((G[np.argsort(-np.sum(G,axis=-1))])[:n_best]).mean())
# print()

# print("Player 2: play high probabilities")
# G = (np.random.uniform(0, 1, (n_agent,n_trial)) < P[1]) * X[1]
# print("Mean gain (all):", G.mean())
# print("Mean gain (best):", 
#       ((G[np.argsort(-np.sum(G,axis=-1))])[:n_best]).mean())
# print()

# print("Player 3: play best expected value")
# P_ = np.where(P[0]*X[0] > P[1]*X[1], P[0], P[1])
# X_ = np.where(P[0]*X[0] > P[1]*X[1], X[0], X[1])
# G = (np.random.uniform(0, 1, (n_agent,n_trial)) < P_) * X_
# print("Mean gain (all):", G.mean())
# print("Mean gain (best):", 
#       ((G[np.argsort(-np.sum(G,axis=-1))])[:n_best]).mean())
# print()


# alpha = 0.5
# print("Player 4: play best expected value with probability distortion (%.1f)" % alpha)
# P0, P1 = P_distortion(P[0], alpha), P_distortion(P[1], alpha)
# P_ = np.where(P0*X[0] > P1*X[1], P[0], P[1])
# X_ = np.where(P0*X[0] > P1*X[1], X[0], X[1])
# G = (np.random.uniform(0, 1, (n_agent,n_trial)) < P_) * X_
# print("Mean gain (all):", G.mean())
# print("Mean gain (best):", 
#       ((G[np.argsort(-np.sum(G,axis=-1))])[:n_best]).mean())
# print()

# alpha = 1.5
# print("Player 5: play best expected value with probability distortion (%.1f)" % alpha)
# P0, P1 = P_distortion(P[0], alpha), P_distortion(P[1], alpha)
# P_ = np.where(P0*X[0] > P1*X[1], P[0], P[1])
# X_ = np.where(P0*X[0] > P1*X[1], X[0], X[1])
# G = (np.random.uniform(0, 1, (n_agent,n_trial)) < P_) * X_
# print("Mean gain (all):", G.mean())
# print("Mean gain (best):", 
#       ((G[np.argsort(-np.sum(G,axis=-1))])[:n_best]).mean())
# print()


# G = np.zeros((n_agent, n_trial))
# G2 = np.zeros((n_agent, n_trial))

# P1 = np.random.uniform(0.0, 1.0, (n_agent, n_trial))
# X1 = np.random.uniform(0.0, 1.0, (n_agent, n_trial))
# # X1 = 1/P1

# P2 = np.random.uniform(P1, 1.0, (n_agent, n_trial))
# X2 = np.random.uniform(0.0, 1.0, (n_agent, n_trial))
# # X2 = 1/P2



# G1 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P1) * X1
# print("Choose low p (all):", G1.mean())
# print("Choose low p (best):",
#       ((G1[np.argsort(-np.sum(G1,axis=-1))])[:n_best]).mean())
# print()
# G2 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P2) * X2
# print("Choose high p (all):", G2.mean())
# print("Choose high p (best):",
#       ((G2[np.argsort(-np.sum(G2,axis=-1))])[:n_best]).mean())

# print()
# G2 = (np.random.uniform(0, 1, (n_agent,n_trial)) < P_distortion(P2,-0.8)) * X2
# print("Choose high p (all):", G2.mean())
# print("Choose high p (best):",
#       ((G2[np.argsort(-np.sum(G2,axis=-1))])[:n_best]).mean())


#     g1 = ((G1[np.argsort(-np.sum(G1[:,:i],axis=-1))])[:n_best,:i]).mean()
#     g2 = ((G2[np.argsort(-np.sum(G2[:,:i],axis=-1))])[:n_best,:i]).mean()

# G = []
# for i in tqdm.trange(10,n_trial):
#     g1 = ((G1[np.argsort(-np.sum(G1[:,:i],axis=-1))])[:n_best,:i]).mean()
#     g2 = ((G2[np.argsort(-np.sum(G2[:,:i],axis=-1))])[:n_best,:i]).mean()
#     # print(g1,g2)
#     G.append(g1/g2)
# plt.plot(G)
# plt.axhline(1, color="black")
# plt.show()
    
# print(np.mean(G1))
# G1 = G1[np.argsort(-np.sum(G1,-1))]
# G1 = G1[:n_best]
# print(np.mean(G1))
      
# print(np.mean(G2))
# G2 = G2[np.argsort(-np.sum(G2,-1))]
# G2 = G2[:n_best]
# print(np.mean(G2))

# for i in range(len(G1)):
#     G1[i] = np.cumsum(G1[i])/np.arange(1,n_trial+1)
#     G2[i] = np.cumsum(G2[i])/np.arange(1,n_trial+1)

# plt.plot(np.mean(G1[:,10:]/G2[:,10:], axis=0), color="red", lw=1.5)
# # plt.plot(np.mean(G2[:,10:], axis=0), color="blue", lw=1.5)
plt.show()


