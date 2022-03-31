import numpy as np
import scipy.stats as stats

def pf_lg(A, Q, H, R, N, T, y):
    """
    Particle Filter for a linear Gaussian HMM / SSM build like so:
    
    Z_t = A * Z_{t-1} + Q * U_t
    Y_t = H * Z_t + R * V_t

    where U and V are standard Gaussian variables.
    """
    # initialization
    x = np.zeros(shape = (T + 1, N))
    w = np.zeros(shape = (T + 1, N))
    estimate = np.zeros(shape = T + 1)
        
    # noise
    epsilon = np.random.normal(0, 1, size = (T + 1, N))
    
    # sample x0, calculate w0
    x[0, :] = Q * epsilon[0,:]

    w[0, :] = stats.norm.pdf(y[0], loc = H * x[0, :], scale = R)
    
    for t in range(1, T + 1):
        
        # sample xi
        x[t, :] = A * x[t-1, :] + Q * epsilon[t, :]
        
        # calculate density value
        u_t = stats.norm.pdf(y[t], loc = H * x[t, :], scale = R)

        # calculate weight
        w[t, :] = np.multiply(w[t-1, :], u_t) 
        
        # normalization of weights
        w[t, :] *= 1/np.sum(w[t, :])
        
        # estimate
        estimate[t] = np.sum(np.multiply(w[t, :], x[t, :]))
        
        # resampling
        x[t, :] = np.random.choice(x[t, :], size = N, p = w[t, :])
        
        # reset weights
        w[t, :] = 1/N
        
    return estimate
