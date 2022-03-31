# made by Niels Richard Hansen, University of Copenhagen

def J_diag(A, Q, H, R, T):
    """
    Helper function finding diagonal matrix of information parameter, J
    Made by Niels Richard Hansen, University of Copenhagen
    """
    J = np.zeros(shape = T + 1)
    J[T] = 1 / Q + H ** 2 / R
    for t in range(T):
        J[t] = (1 + A ** 2) / Q + H ** 2 / R
    return(J)

def h_vec(x, H, R):
    """
    Helper function finding information parameter, h
    Made by Niels Richard Hansen, University of Copenhagen
    """
    return(x * H / R)

def forward(J, h, A, Q): 
    """
    Forward pass, kalman filter
    """
    AQ = - A / Q
    T = len(h) - 1
    J_forward = np.zeros(shape = T + 1)
    h_forward = np.zeros(shape = T + 1)
    for t in range(T):
        Jt = J[t] + J_forward[t]
        ht = h[t] + h_forward[t]
        J_forward[t + 1] = - AQ * AQ / Jt
        h_forward[t + 1] = - AQ * ht / Jt
    return(J_forward, h_forward)

def backward(J, h, A, Q): 
    """
    Backward pass, kalman smoother in combination with forward
    Made by Niels Richard Hansen, University of Copenhagen
    """
    AQ = - A / Q
    T = len(h) - 1
    J_backward = np.zeros(shape = T + 1)
    h_backward = np.zeros(shape = T + 1)
    for t in reversed(range(1, T + 1)):
        Jt = J[t] + J_backward[t]
        ht = h[t] + h_backward[t]
        J_backward[t - 1] = - AQ * AQ / Jt
        h_backward[t - 1] = - ht * AQ / Jt
    return(J_backward, h_backward)

def kalman_smoother(A, Q, H, R, T, y):
    """
    Wrapper function taking Q and R as standard deviations in the same fashion as 
    the particle-based smoothers in this repo.
    """
    # gets kalman smoother, but has input which is standard deviation
    Q = Q**2
    R = R**2

    J = J_diag(A, Q, H, R, T)
    h = h_vec(y, H, R)
    (J_f, h_f) = forward(J, h, A, Q)
    (J_b, h_b) = backward(J, h, A, Q)
    Sigma = 1  (J + J_f + J_b)
    mu = Sigma * (h + h_f + h_b)
    return mu
