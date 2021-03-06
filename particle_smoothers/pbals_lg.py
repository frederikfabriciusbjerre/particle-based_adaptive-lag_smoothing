import numpy as np
import scipy.stats as stats

def rejection_sampling_lg(x, w, i, m, t, max_pdf, A, delta):
    """
    Rejection sampling for linear Gaussian HMM
    """
    # initilization of set of indexes returned
    S = np.zeros(shape = m)

    # draw 
    J_list = np.random.choice(np.arange(len(w)), size = delta, p = w)
    
    # initilization of counter
    k = 0
    n = 0
    while k < m and n < delta:
        # draw
        J = J_list[n]
        
        # calculate pdf value
        mu = A * x[t-1, J]
        se = (x[t, i] - mu)**2
        pdf_val = (max_pdf**se)
        
        # rejection step
        if np.random.binomial(size = 1, n = 1, p = pdf_val) == 1:
            S[k] = J
            k += 1
        n += 1
    
    # if all samples are not found
    if k < m:
        return None
    else:
        return S.astype(int)
    
def pbals_lg(A, sigma_u, H, sigma_v, T, N, M, delta, epsilon, y):
    
    """
    Offline implementation of Particle-Based Adaptive-Lag Smoother as seen in the article
    "Particle-Based Adaptive-Lag Online Marginal Smoothing in General State-Space Models"
    for linear Gaussian HMM
    """

    # initialization
    S   = np.array([0]) # set of active estimators
    x   = np.zeros(shape = (T + 1, N)) # xi in algorithm setps
    w   = np.zeros(shape = (T + 1, N)) # weight
    tau = np.zeros(shape = (T + 1, T + 1, N)) # t stat estimates
    estimate = np.zeros(shape = T + 1)
    t   = 0
    
    # calculate constant from gaussian pdf to optimize computing time
    # in rejection sampling step
    max_pdf = np.exp(-1/(2 * (sigma_u**2)))
    
    # noise
    U = np.random.normal(0, 1, size = (T + 1, N))
    
    # first estimate and weight
    x[0,:] = sigma_u * U[0, :]
    w[0,:] = stats.norm.pdf(y[0], loc = H * x[0,:], scale = sigma_v)
    
    # set tau_(0|0)^i for all i
    tau[0, 0, :] = x[0, :]
    
    for t in range(1, T + 1):
        # draw indexes
        idx = np.random.choice(N, size = N, p = w[t-1, :]/np.sum(w[t-1, :]))
        
        # sample x
        x[t, :] = A * x[t-1, :][idx] + sigma_u * U[t, :]
        
        # calc u_t
        u_t = stats.norm.pdf(y[t], loc = H * x[t, :], scale = sigma_v)
        
        # calculate weight
        w[t, :] = u_t  
        
        # normalization of weights
        w[t,:] /= np.sum(w[t, :])
  
        # weights for sampling J
        w_sampling = w[t-1,:] / np.sum(w[t-1,:])
        
        for i in range(N):
            
            # find indices
            indices = rejection_sampling_lg(x, w_sampling, i, M, t, max_pdf, A, delta)
            
            # if indices are type None
            # calculate tau directly
            if indices is None:
                for s in S: 
                    q = stats.norm.pdf(x[t,i], loc = A * x[t-1, :], scale = sigma_u)
                    wq = np.multiply(w[t-1, :], q[:])
                    tau[s, t, i] = np.sum(wq / (np.sum(wq)) * tau[s, t-1, :])
            
            # calculate tau like in article
            else:     
                for s in S:
                    tau[s, t, i] = np.average(tau[s, t - 1, :][indices])
        
            # initialize tau
            tau[t, t, i] = x[t, i]
        
        # new active estimator
        S = np.append(S, t)
        
        for s in S:
            
            # calculate variance
            tau_w_sum = np.sum(np.multiply((w[t,:]), tau[s, t, :]))
            val = tau[s, t, :] - tau_w_sum
            val_sq = np.square(val)
            sigma_estimate = np.sum((w[t, :]) * (val_sq))
            
            if sigma_estimate < epsilon:
                
                # calculate output
                estimate[s] = np.sum(np.multiply((w[t, :]), tau[s, t, :]))
                
                # remove estimator from active estimators
                S = np.setdiff1d(S, s)
    
    # if S is not empty after last iteration
    if len(S) > 0:
        for s in S:
            estimate[s] = np.sum(w[t, :] * tau[s, t, :])
            
    return estimate
