import numpy as np

# kalman filter algorithm is made by Niels Richard Hansen, University of Copenhagen

def kalman_filter(A, Q, H, R, T, x):
    """
    made by Niels Richard Hansen, University of Copenhagen
    Kalman Filter algorithm working for a linear Gaussian HMM / SSM build like so:
    
    Z_t = A * Z_{t-1} + Q * U_t
    Y_t = H * Z_t + R * V_t

    where U and V are standard Gaussian variables. 
    """
    # initialize mu and Sigma
    mu = np.zeros(shape = T + 1)
    Sigma = np.zeros(shape = T + 1)
    
    # calculate K^(t)
    Kt = (Q * H) / (H * Q * H + R)
    mu[0] = Kt * x[0]
    Sigma[0] = (1 - Kt * H) * Q
    for t in range(T):
        mut = A * mu[t]
        Sigmat = A * Sigma[t] * A + Q
        Kt = Sigmat * H / (H * Sigmat * H + R)
        mu[t + 1] = mut + Kt * (x[t + 1] - H * mut)
        Sigma[t + 1] = (1 - Kt * H) * Sigmat
    return ((mu, Sigma))


def kalman_als(A, sigma_u, H, sigma_v, T, epsilon, y):
    """
    Kalman filter version of the Adaptive-Lag Smoother as seen in the article
    "Particle-Based Adaptive-Lag Online Marginal Smoothing in General State Space Models"
    working for a linear Gaussian HMM / SSM build like so:
    
    Z_t = A * Z_{t-1} + Q * U_t
    Y_t = H * Z_t + R * V_t

    where U and V are standard Gaussian variables.
    Note that the estimates will become less precise as we approach the end of sequence.
    """
    # initialize alpha, beta, mu, T_s|t, and sigma
    alpha = np.zeros(shape = (T+1,T+1)) 
    beta = np.zeros(shape = (T+1,T+1))
    sigma = np.zeros(shape = (T+1, T+1))
    mu = np.zeros(shape = (T+1, T+1))
    tau = np.zeros(shape = (T+1, T+1)) # named T_s|t in pseudo code
    output = np.zeros(shape = T+1)
    S   = np.array([]) # set of active estimators
    estimate = np.zeros(shape = T+1)
    
    # as filter takes variance as input:
    sigma_u = sigma_u * sigma_u
    sigma_v = sigma_v * sigma_v
    
    (filter_mu, filter_sigma) = kalman_filter(A, sigma_u, H, sigma_v, T, y)
    
    tau[0,0] = y[0]
    for i in range(0, T+1):
        alpha[i,i] = 1
        beta[i,i] = 0
    for t in range(0, T+1):
        #print(t)
        mu_t = filter_mu[t]
        sigma_t = filter_sigma[t]

        for s in S:
            # if S is empty s will be an empty list
            if type(s) is list:
                pass
            
            # calculate tau
            s = s.astype(int)
            tau[s, t] = alpha[s,t] * y[t] + beta[s,t]
        
        S = np.append(S, t)
        # calculate tau
        # assumption: alpha = 1 and beta = 0 for all h_s
        tau[t,t] = y[t]
        
        if t != T:
            # calculate mu and sigma
            sigma[t, t+1]= 1/(A * (1/sigma_u) * A + (1/sigma_t))
            mu[t,t+1] = sigma[t, t+1] * (A * (1/sigma_u) * y[t+1] + (1/sigma_t) * mu_t)
        
        
        for s in S:
            s = s.astype(int)
            
            # calculate variance
            variance = alpha[s,t] * sigma_t * alpha[s,t]
            if variance < epsilon:
                # calculate output
                estimate[s] = tau[s,t]
                
                # remove s from active set of estimators
                S = np.setdiff1d(S, s)
                
            # calculate alpha and beta for t+1
            if t != T:             
                alpha[s,t+1] = alpha[s,t] * sigma[t, t+1] * A * (1/sigma_u)
                beta[s,t+1]  = alpha[s,t] * sigma[t, t+1] * A * (1/sigma_t) * mu_t + beta[s,t]

        # if S is not empty after last iteration 
        if len(S) > 0:
            for s in S:
                s = s.astype(int)
                estimate[s] = tau[s,t]
    return estimate
