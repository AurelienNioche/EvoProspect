import matplotlib.pyplot as plt
import numpy as np

def P_distortion(X, alpha=1.0):
    return np.exp(-(-np.log(X))**alpha)

def V_distortion(X, alpha=0.0):
    return X**(1-alpha)

n = 10000
# p = np.linspace(0.01, 1.00, n)
# ev1 = np.ones(n)
# ev2 = np.linspace(0.01, 1, n)
# ev3 = np.linspace(2, 1, n)
#
# v1 = ev1/p
# v2 = ev2/p
# v3 = ev3/p
#
# v = 1/P_distortion(p, 0.5)

p = np.random.uniform(0.01, 1.00, n)
p.sort()
v = np.random.uniform(0.01, 1.00, n)

v /= v.max()

# plt.plot(p, v1)
# plt.plot(p, v2)
# plt.plot(p, v3)
# plt.plot(p, P_distortion(p) * V_distortion(v1))
# plt.plot(p, P_distortion(p) * V_distortion(v), label="neutral")
# plt.plot(p, P_distortion(p, 1.5) * V_distortion(v), label="disto proba no-KT")

#plt.ylabel("SEU")E
# plt.scatter(p, p*v, label="EV", alpha=0.5, s=10)

for y in np.linspace(0.1, 1, 6):
    plt.plot(p, P_distortion(p, 0.15) * y, label="SEU disto proba KT")
# plt.hist(p*v)
plt.xlabel("ev")
plt.legend()
plt.show()