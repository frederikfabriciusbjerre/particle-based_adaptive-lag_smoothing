import numpy as np
from particle_smoothers.pbals_lg import pbals_lg
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from kalman_filter_and_smoothers.kalman_smoother_information_param import kalman_smoother

# set parameters
A = 0.95
Q = 0.5
H = 0.5
R = 2
T = 100

# build hmm
# gaussian transitions
U1 = np.random.normal(0, 1, size = T + 1)
V1 = np.random.normal(0, 1, size = T + 1)

# initiate zs and xs
z = np.zeros(shape = T + 1)
y = np.zeros(shape = T + 1)

# build hmm 
z[0] = Q * U1[0]
y[0] = H * z[0] + R * V1[0] # observation is z plus noise
for t in range(1, T + 1):
    z[t] = A * z[t - 1] + Q * U1[t]
    y[t] = H * z[t] + R * V1[t]

# set algorithm parameters
N = 400 # number of samples
M = 2 # precision parameter of backward samples
delta = 1000 # max number of rejection
epsilon = 0.001 # tolerance parameter

# run particle-based smoother
mu_pbalms = pbals_lg(A, Q, H, R, T, N, M, delta, epsilon, y)

# run kalman smoother
mu_correct = kalman_smoother(A, Q, H, R, T, y)

# plot
plt.plot(mu_pbalms, label = "palms")
plt.plot(mu_correct, label = "kalman")
plt.xlabel('t')
plt.ylabel('estimates')
plt.legend(loc = "best")
plt.show()

# print MSE
print(mean_squared_error(mu_correct, mu_pbalms))
