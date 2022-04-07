# Particle-Based Filtering and Smoothing for Hidden Markov Models
as seen in the article https://arxiv.org/abs/1812.10939

## Kalman Implementations
### Kalman Filter and Smoother
These functions are made by Niels Richard Hansen, Professor in Statistics at University of Copenhagen, and are used for testing particle-based methods. They assume a linear Gaussian hidden Markov model / state space model of the form:

```math
Z_{t+1} = A * Z_{t} + Q U_{t+1} \\
Y_{t} = H * Z_{t} + R V_{t}
```
where 
```math
A, H, Q, R \in \mathbb{R}
```

and $`U, V`$ are standard Gaussian variables. Initialize $`Z_0 = Q * U_0`$ and $`Y_0 = H * Z_0 + \sigma_V \cdot V_{0}`$. 

As input the `kalman_filter` takes variance ($Q^2, R^2$), but the `kalman_smoother` wrapper function takes standard deviation as input. 

### Kalman Adaptive-Lag Smoother
This is an adaptive-lag version of the smoother made to work on the linear Gaussian HMM structure seen above. An implementation can be found in file `kalman_als.py`. As inputs it takes $`A, Q, H, R, T, y`$, where $`T`$ is the length of the observation vector $`y`$, as with the Kalman filter and smoother, but it also takes the input `epsilon`, which is a tolerance parameter for the variance. To understand this method in depth, read the article referred above. 

To test this method against a Kalman smoother, see `example_kalman.py`

## Particle-Based Implementations
### Particle-filter
The particle filter method will not be explained in depth here, but was based on the presentation of the algorithm made in the book "Computational Statistics" by Givens and Hoeting. It is made to work on the linear Gaussian HMM from above and a stochastic Volatility model with the following structure:

```math
Z_{t+1} = \gamma * Z_t + \sigma *  U_{t+1} \\
Y_{t} = \beta * exp(X_{t}/2)V_{t}
```
again where $`U, V`$ are standard Gaussian variables. Implementations can be found in `particle_filters`, and for a test of this method against a Kalman filter, see `example_pf.py`

### Particle-Based Adaptive-Lag Smoother
This method is introduced in the article referred to in the beginning of the article. The implementations can be found in `particle_smoothers` and are made to work on a linear Gaussian HMM and a stochastic volatility model. (Here, _h_, is assumed to be the id function, if you read into the article.)

To see tests of these methods, we refer to `example_lg.py` and `example_sv.py`. 
