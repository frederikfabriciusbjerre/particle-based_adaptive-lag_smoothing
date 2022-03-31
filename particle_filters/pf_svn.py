import numpy as np
import scipy.stats as stats

def pf_sv(phi, sigma, beta, N, T, y):
    """
    Particle filter for a stochastic volatility model build like so:

    Z_{t+1} = phi * Z_t + sigma * U_t
    Y_t = beta * exp(Z_t/2) * V_t

    where U and V are standard Gaussian variables. 
    """
    # initialization
    x = np.zeros(shape = (T + 1, N))
    w = np.zeros(shape = (T + 1, N))
    estimate = np.zeros(shape = T + 1)
    
    # sample x0, calculate w0
    x[0, :] = np.random.normal(0, sigma / np.sqrt(1 - phi**2), size = N)
    w[0, :] = stats.norm.pdf(y[0], loc = 0, scale = beta * np.exp(x[0, :]/2))
    
    # noise
    U = np.random.normal(0, 1, size = (T + 1, N))
   
    for t in range(1, T + 1):
        # sample xi
        x[t, :] = phi * x[t-1, :] + sigma * U[t, :]

        # pdf function value
        u_t = stats.norm.pdf(y[t], loc = 0, scale = beta * np.exp(x[t, :]/2))

        # calculate weight
        w[t, :] = w[t-1, :] * u_t  
        
        # normalization of weights
        w[t, :] *= 1/np.sum(w[t, :])
        
        # estimate
        estimate[t] = np.sum(w[t, :] * x[t, :])
        
        # resampling
        x[t, :] = np.random.choice(x[t, :], size = N, p = w[t, :])

        # reset weights
        w[t, :] = 1/N

    return estimate
