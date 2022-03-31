import numpy as np
from particle_smoothers.pbals_sv import pbals_sv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# set parameters
phi = 0.98
sigma = np.sqrt(0.1)
beta = np.sqrt(0.7)
T = 300

# build hmm
# gaussian transitions
U = np.random.normal(0, 1, size = T + 1)
V = np.random.normal(0, 1, size = T + 1)

# initiate zs and xs
z = np.zeros(shape = T + 1)
y = np.zeros(shape = T + 1)

# build hmm 
z[0] = np.random.normal(0, sigma / np.sqrt((1 - phi**2)))
y[0] = beta * np.exp(z[0]/2) * V[0]
for t in range(1, T + 1):
    z[t] = phi * z[t - 1] + sigma * U[t]
    y[t] = beta * np.exp(z[t]/2) * V[t]

# set algorithm parameters
N = 400 # number of samples
M = 2 # precision parameter of backward samples
delta = 1000 # max number of rejection
epsilon = 0.001 # tolerance parameter

# run particle-based smoother
mu_pbalms = pbals_sv(phi, sigma, beta, T, N, M, delta, epsilon, y)

# plot
plt.plot(mu_pbalms, label = "pbals")
plt.plot(z, label = "true values")
plt.xlabel('t')
plt.ylabel('estimates')
plt.legend(loc = "best")
plt.show()
