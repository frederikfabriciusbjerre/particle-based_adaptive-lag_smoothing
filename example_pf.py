import numpy as np
from particle_filters.pf_lg import pf_lg
from kalman_filter_and_smoothers.kalman_als import kalman_filter
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

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

# run particle filter
mu_pf = pf_lg(A, Q, H, R, N, T, y)

# run kalman filter
mu_kf, _ = kalman_filter(A, Q*Q, H, R*R, T, y)

# plot
plt.plot(mu_pf, label = "particle filter")
plt.plot(mu_kf, label = "kalman filter")
plt.xlabel('t')
plt.ylabel('estimates')
plt.legend(loc = "best")
plt.show()

# print MSE
print(mean_squared_error(mu_kf, mu_pf))
