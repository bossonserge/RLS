# Theory: Learning Optimal Probing Signals

This document develops the theoretical foundations for designing optimal excitation signals
for parameter estimation. The ultimate goal is a toy problem where the probing signal is
learned via reinforcement learning and benchmarked against analytical solutions.

Work through the sections in order. Each section has a **Read** pointer, a **Key insight**
to internalize, a **Hands-on** exercise, and a **Check question**.

---

## 1. Linear Estimation and Least Squares

### Theory

Consider the scalar linear regression model:

$$y_t = \theta \cdot \phi_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

where θ is an unknown scalar parameter, φ_t is a known input (regressor), and y_t is the
noisy observation. The ordinary least squares (OLS) estimate after T observations minimizes
the sum of squared residuals [1]:

$$\hat{\theta}_T = \frac{\sum_{t=1}^T \phi_t y_t}{\sum_{t=1}^T \phi_t^2}$$

This can be computed recursively without storing all past data. The **recursive least squares
(RLS)** update [1] is:

$$\hat{\theta}_t = \hat{\theta}_{t-1} + K_t \left( y_t - \phi_t \hat{\theta}_{t-1} \right)$$

$$K_t = \frac{P_{t-1} \phi_t}{\sigma^2 + \phi_t^2 P_{t-1}}$$

$$P_t = (1 - K_t \phi_t) P_{t-1}$$

where P_t is the estimation error variance. Notice that P_t depends on the inputs φ_t — the
choice of input directly affects how fast the estimate converges.

### Read

- Ljung, L. *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999.
  Chapter 2 (least squares) and Chapter 11 (recursive methods). [1]

### Key insight

The estimation error variance P_t is determined entirely by the sequence of inputs {φ_t},
not by the observations {y_t}. Better inputs lead to smaller P_T regardless of noise
realizations.

### Hands-on

Implement RLS in Python. Generate synthetic data with θ = 2.5, σ = 0.5, T = 50.

1. Run RLS with φ_t = 1 (constant input). Plot θ̂_t and P_t over time.
2. Run RLS with φ_t ~ Uniform(-1, 1) (random input). Plot the same.
3. Overlay both P_t trajectories on one plot.