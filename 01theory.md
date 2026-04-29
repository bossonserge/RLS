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

**Question:** Which input leads to faster convergence of P_t? Can you explain why from the
RLS equations?

---

## 2. Bayesian Estimation and Posterior Covariance

### Theory

Bayesian estimation treats θ as a random variable with a prior distribution. For the linear
Gaussian model above with a Gaussian prior θ ~ N(θ̂_0, P_0), the posterior after observing
y_1, ..., y_T is also Gaussian [2]:

$$p(\theta \mid y_{1:T}) = \mathcal{N}(\hat{\theta}_T, P_T)$$

The posterior mean θ̂_T is the minimum mean square error (MMSE) estimate, and P_T captures
remaining uncertainty. The updates are [2]:

$$\hat{\theta}_t = \hat{\theta}_{t-1} + P_{t-1} \phi_t \left( \sigma^2 + \phi_t^2 P_{t-1} \right)^{-1} \left( y_t - \phi_t \hat{\theta}_{t-1} \right)$$

$$P_t^{-1} = P_{t-1}^{-1} + \frac{\phi_t^2}{\sigma^2}$$

This is the Kalman filter for a static parameter [3]. Unlike OLS, the Bayesian approach
carries the full distribution — not just a point estimate — which is essential for reasoning
about what to measure next.

### Read

- Bishop, C. M. *Pattern Recognition and Machine Learning*, Springer, 2006. Section 3.3
  (Bayesian linear regression). [2]
- Kalman, R. E. "A new approach to linear filtering and prediction problems," *ASME Journal
  of Basic Engineering*, vol. 82, pp. 35–45, 1960. [3]

### Key insight

The posterior distribution is the complete summary of what is known about θ given the data.
The posterior variance P_T tells you not just how good your estimate is, but how much more
you could gain from additional measurements.

### Hands-on

Extend your RLS code to track and plot the full posterior N(θ̂_t, P_t).

1. At t = 0, 5, 15, 30, plot the posterior density p(θ | y_{1:t}) as a Gaussian curve.
2. Overlay the true θ = 2.5 as a vertical line.
3. Now use 10 identical observations with φ_t = 1. Describe what the posterior looks like
   after t = 10 vs t = 50.


**Question:** What happens to the posterior if all inputs are identical in this scalar model?
Why does the estimate still improve, but with diminishing returns, and what does this reveal
about when input diversity matters?

---

## 3. Bridge: Posterior Covariance and Fisher Information

### Theory

The **Fisher information** I(θ; u_t) measures how much a single observation y_t = f(θ, u_t) + ε
tells us about θ [4]:

$$\mathcal{I}(\theta; u_t) = \mathbb{E}\left[ \left( \frac{\partial}{\partial \theta} \log p(y_t \mid \theta, u_t) \right)^2 \right]$$

For the linear Gaussian model y_t = θ · φ_t + ε, this evaluates to:

$$\mathcal{I}(\theta; \phi_t) = \frac{\phi_t^2}{\sigma^2}$$

The **Cramér-Rao lower bound** states that no unbiased estimator can achieve a variance
smaller than I^{-1} [4]. For the Bayesian posterior update:

$$P_t^{-1} = P_{t-1}^{-1} + \mathcal{I}(\theta; \phi_t)$$

The posterior precision is the prior precision plus the accumulated Fisher information. The
two frameworks are identical in the linear Gaussian case — Bayesian estimation achieves the
Cramér-Rao bound exactly [2].

### Read

- Kay, S. M. *Fundamentals of Statistical Signal Processing, Vol. I: Estimation Theory*,
  Prentice Hall, 1993. Chapter 3 (Cramér-Rao bound) and Chapter 15 (Fisher information
  matrix). [4]

### Key insight

In the linear Gaussian model, the Bayesian posterior covariance and the inverse Fisher
information are the same object. Designing informative experiments is equivalent to
maximizing the Fisher information accumulated over the experiment.

### Hands-on

Verify the identity numerically.

1. Generate a random sequence of inputs φ_1, ..., φ_T ~ Uniform(-2, 2).
2. Compute P_T^{-1} via recursive Bayesian update.
3. Compute Σ_t φ_t² / σ² + P_0^{-1} directly.
4. Verify that both quantities are equal to machine precision.


**Question:** Derive the posterior covariance update P_t = f(P_{t-1}, φ_t, σ) algebraically
from the precision update P_t^{-1} = P_{t-1}^{-1} + φ_t²/σ². Confirm your derivation
matches your simulation.

---

## 4. Fisher Information and D-Optimal Experiment Design

### Theory

**Optimal experiment design (OED)** asks: given a budget of T measurements, which inputs
{u_1, ..., u_T} minimize the final estimation uncertainty [5]?

Several optimality criteria exist. **D-optimality** minimizes det(P_T), which for the scalar
case reduces to minimizing P_T itself [5]. Substituting the precision update:

$$P_T^{-1} = P_0^{-1} + \frac{1}{\sigma^2} \sum_{t=1}^T \phi_t^2$$

Maximizing P_T^{-1} subject to |u_t| ≤ u_max means: maximize Σ φ_t². For a linear model
where φ_t = u_t, the solution is trivial — saturate the input at every step [5].

The problem becomes non-trivial when the Fisher information depends on the unknown θ. For a
nonlinear observation model y_t = f(θ, u_t) + ε:

$$\mathcal{I}(\theta; u_t) = \frac{1}{\sigma^2} \left( \frac{\partial f}{\partial \theta}(θ, u_t) \right)^2$$

The optimal u_t now depends on θ — unknown. **Locally optimal design** substitutes the
current estimate θ̂_t, giving a greedy policy [5]:

$$u_t^* = \arg\max_{u} \left( \frac{\partial f}{\partial \theta}(\hat{\theta}_t, u) \right)^2$$

### Read

- Pukelsheim, F. *Optimal Design of Experiments*, SIAM, 2006. Chapter 1 (introduction and
  criteria). [5]
- Atkinson, A. C., Donev, A. N., and Tobias, R. D. *Optimum Experimental Designs, with SAS*,
  Oxford University Press, 2007. Chapter 10 (nonlinear models). [6]

### Key insight

For linear models, the optimal design is trivially the maximum-amplitude input. The
interesting and practically relevant case arises when the optimal input depends on the
unknown parameter — local optimality then gives a computable but myopic policy.

### Hands-on

Consider y_t = θ · u_t + ε with u_t ∈ [-1, 1], T = 10, σ = 0.3, P_0 = 1.

1. Show analytically that u_t = ±1 at every step is D-optimal.
2. Now compare three strategies: u_t = 1 (constant), u_t = ±1 alternating, u_t ~ Uniform.
   Plot P_T for each.
3. What is the minimum achievable P_T given T = 10?



**Question:** If you have T = 10 measurements and must choose u_t ∈ [-1, 1], is the optimal
strategy unique? What if the sign of θ is unknown?

---

## 5. Why Greedy Design Fails: A Two-Step Counterexample

### Theory

The greedy locally optimal design maximizes Fisher information at each step using the current
estimate θ̂_t. This is **myopic** — it ignores how today's input affects tomorrow's estimation
quality.

Consider the nonlinear model:

$$y_t = \sin(\theta \cdot u_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

Fisher information for a single observation:

$$\mathcal{I}(\theta; u_t) = \frac{u_t^2 \cos^2(\theta \cdot u_t)}{\sigma^2}$$

The locally optimal input given current estimate θ̂ maximises $(x \cos x)^2$ where $x = \hat{\theta}\,u$.
The optimality condition df/dx = cos(x) - x·sin(x) = 0 gives cot(x) = x, with first positive
solution x* ≈ 0.860:

$$u_t^* = \frac{x^*}{\hat{\theta}_t} \approx \frac{0.860}{\hat{\theta}_t}$$

This places the operating point at maximum sensitivity of the observation map.
Note: the naively appealing choice u = π/(2θ̂) gives x = π/2 where cos(π/2) = 0 —
this is the point of **zero** Fisher information.

However, over two steps, a different first step can lead to a better posterior after step 2.
If θ̂_1 is biased due to a poor first observation, the greedy choice at step 2 compensates
poorly. A first step that is suboptimal in isolation can reduce the uncertainty about θ in a
way that makes step 2 far more informative [6].

The suboptimality of greedy design is related to the **dual effect** identified by
Feldbaum [7]: optimal control under parameter uncertainty must account for both current
performance and future information gain.

### Read

- Feldbaum, A. A. "Dual control theory I–IV," *Automation and Remote Control*, vols. 21–22,
  1960–1961. [7]
- Bar-Shalom, Y. and Tse, E. "Dual effect, certainty equivalence, and separation in
  stochastic control," *IEEE Transactions on Automatic Control*, vol. 19, no. 5,
  pp. 494–500, 1974. [8]

### Key insight

Greedy maximization of Fisher information is a one-step-ahead policy. It can be arbitrarily
suboptimal because it does not account for how the current input shapes the information
available at future steps.

### Hands-on

Use the sin(θu) model with θ = 1.5, σ = 0.3, u ∈ [0.1, 5], T = 2, P_0 = 2.

1. Compute the greedy optimal u_1 given θ̂_0 = 1.5 (prior mean = true value for simplicity).
2. For a grid of u_1 values, simulate both steps and compute E[P_2] by averaging over many
   noise realizations.
3. Plot E[P_2] as a function of u_1. Mark the greedy choice.


**Question:** Is the greedy u_1 the global minimizer of E[P_2]? Quantify the gap between
greedy and optimal. Under what conditions (σ, P_0) is the gap largest?


---

## 6. Bellman Equation and Dynamic Programming

### Theory

The sequential experiment design problem can be stated as a stochastic dynamic program [9].
The state at time t is the posterior (θ̂_t, P_t). The agent chooses u_t, observes y_t, and
updates the posterior. The objective is to minimize final uncertainty:

$$V_T(\hat{\theta}, P) = P$$

$$V_t(\hat{\theta}, P) = \min_{u} \mathbb{E}_{y} \left[ V_{t+1}\left( \hat{\theta}'(\hat{\theta}, P, u, y),\ P'(P, u) \right) \right]$$

Note that P' = f(P, u) does not depend on y (for Gaussian posteriors), but θ̂' does. The
**Bellman equation** decomposes the T-step problem into T one-step problems solved
backwards [9].

For the sin(θu) model, the posterior update is approximately Gaussian when P is small
(linearization around θ̂), enabling tractable DP by discretizing (θ̂, P) [9].

The optimal policy π*(θ̂, P) = argmin_u V(θ̂, P) can be computed by backward induction over
a grid of states. This gives the exact optimal probing signal for any posterior state.

### Read

- Bertsekas, D. P. *Dynamic Programming and Optimal Control*, Vol. I, 4th ed., Athena
  Scientific, 2017. Chapter 1 (introduction) and Chapter 6 (stochastic DP). [9]
- Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT
  Press, 2018. Chapter 3 (finite MDPs and Bellman equations). [10]

### Key insight

Dynamic programming converts a T-step sequential problem into T one-step problems via
backward induction. The value function V(θ̂, P) captures the expected final uncertainty
achievable from any posterior state — it is the optimal "price" of uncertainty.

### Hands-on

Solve the 5-step sin(θu) problem via DP.

1. Discretize θ̂ ∈ [0.5, 3.0] and P ∈ [0.01, 2.0] on a grid (e.g., 50×50 points).
2. Implement backward induction to compute V_t(θ̂, P) and the optimal policy π*(θ̂, P).
3. Plot π*(θ̂, P) as a heatmap. Compare with the corrected local-FIM policy
   $u = 0.860 / \hat{\theta}$.



**Question:** Does π*(θ̂, P) agree with the greedy policy when P is large (high uncertainty)?
When P is small (low uncertainty)? Explain the difference intuitively.


---

## 7. From Dynamic Programming to Reinforcement Learning

### Theory

DP requires a known transition model — in our case, how the posterior (θ̂, P) evolves with
each input u and observation y. For the Gaussian approximation this is available, but in
general it may not be.

**Reinforcement learning (RL)** approximates the optimal policy or value function from
experience, without requiring a model [10]. The key connection is:

- DP solves the Bellman equation exactly by backward induction on a known model.
- RL solves (or approximates) the same Bellman equation from sampled transitions.

The **Q-function** (action-value function) satisfies:

$$Q_t(s, u) = \mathbb{E}\left[ r_{t+1} + \gamma \min_{u'} Q_{t+1}(s', u') \mid s_t = s, u_t = u \right]$$

where s = (θ̂, P), r is the reward (negative P at final step), and s' is the next posterior
state. **Q-learning** updates this estimate from observed transitions [10]:

$$Q(s, u) \leftarrow Q(s, u) + \alpha \left[ r + \gamma \min_{u'} Q(s', u') - Q(s, u) \right]$$

For continuous state spaces, the Q-function is approximated by a neural network (**deep RL**,
e.g., DQN or policy gradient methods such as PPO) [11].

Note the conceptual parallel: the exploration-exploitation tradeoff in RL mirrors the dual
control problem. An RL agent probing its environment to improve its policy is solving the
same fundamental problem as a controller probing a system to identify its parameters [8].

### Read

- Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT
  Press, 2018. Chapters 6 (TD learning) and 13 (policy gradient). [10]
- Mnih, V. et al. "Human-level control through deep reinforcement learning," *Nature*,
  vol. 518, pp. 529–533, 2015. [11]

### Key insight

RL is a family of algorithms for solving the Bellman equation from data rather than from a
known model. When the posterior state is available and low-dimensional, RL should recover the
DP solution; its advantage appears when the model is unknown or the state space is too large
for exact DP.

### Hands-on

Implement Q-learning (tabular) on the 5-step sin(θu) problem.

1. Use the same discretized state space as in Section 6.
2. Train for 10,000 episodes. In each episode, sample θ ~ Uniform(0.5, 3.0) and run 5 steps.
3. Plot the learned Q-policy π_Q(θ̂, P) and overlay with the DP solution π*(θ̂, P) from
   Section 6.
4. Plot the learning curve: mean E[P_5] per 500 episodes.


**Question:** How many episodes does Q-learning need to approximately recover the DP policy?
What happens if you remove the control cost (allow any u freely) — does convergence improve
or degrade? Why?


---

## 8. Full Toy Problem: Learning the Optimal Probing Signal

### Theory

We now combine all elements into the complete toy problem. The model is:

$$y_t = \sin(\theta \cdot u_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

with unknown θ, and the objective is to minimize a cost that trades estimation accuracy
against excitation energy [6]:

$$J = \mathbb{E}\left[ P_T + \lambda \sum_{t=1}^T u_t^2 \right]$$

The state is the posterior (θ̂_t, P_t). The agent must learn a policy π: (θ̂, P) → u that
minimizes J.

Three baselines are available for comparison:
1. **Greedy local FIM** — $u_t \approx 0.860 / \hat{\theta}_t$, ignores energy cost and future steps.
2. **DP** — exact optimal solution via backward induction (Section 6), serves as ground truth.
3. **RL (learned)** — policy trained from episodes, no model assumed beyond the simulator.

For well-trained RL, the policy should approach the DP solution. Differences reveal the
cost of model-free learning (sample efficiency) and any approximation errors.

### Read

- Mehra, R. K. "Optimal input signals for parameter estimation in dynamic systems — survey
  and new results," *IEEE Transactions on Automatic Control*, vol. 19, no. 6, pp. 753–768,
  1974. [12]
- RL algorithm of your choice — PPO: Schulman, J. et al. "Proximal policy optimization
  algorithms," arXiv:1707.06347, 2017. [13]

### Key insight

The optimal probing signal is a function of the current posterior state (θ̂, P), not a fixed
waveform. It front-loads excitation when uncertainty is high and backs off when the estimate
is already precise — a behavior that emerges naturally from the DP/RL solution but is absent
from greedy or fixed-signal approaches.

### Hands-on

Train an RL agent (use Stable-Baselines3 or a minimal custom implementation) on the full
toy problem with λ = 0.1, T = 10, σ = 0.3.

1. State: (θ̂_t, P_t). Action: u_t ∈ [0.1, 5.0]. Reward: -P_T - λ Σ u_t² at final step.
2. Compare J achieved by: greedy local-FIM policy, DP solution, trained RL agent.
3. Plot the learned probing signal u_t over a typical episode. Does the agent front-load
   excitation?
4. Vary λ ∈ {0, 0.05, 0.1, 0.5} and plot the Pareto curve of final P_T vs total energy Σ u_t².


**Question:** Does the RL agent front-load excitation early and exploit later, or does it
spread excitation uniformly? How does this behavior change with λ? Quantify the energy-
accuracy tradeoff compared to the DP baseline.

---

## References

[1] L. Ljung, *System Identification: Theory for the User*, 2nd ed. Upper Saddle River, NJ:
Prentice Hall, 1999.

[2] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York: Springer, 2006.

[3] R. E. Kalman, "A new approach to linear filtering and prediction problems," *ASME Journal
of Basic Engineering*, vol. 82, pp. 35–45, 1960.

[4] S. M. Kay, *Fundamentals of Statistical Signal Processing, Vol. I: Estimation Theory*.
Englewood Cliffs, NJ: Prentice Hall, 1993.

[5] F. Pukelsheim, *Optimal Design of Experiments*. Philadelphia: SIAM, 2006.

[6] A. C. Atkinson, A. N. Donev, and R. D. Tobias, *Optimum Experimental Designs, with SAS*.
Oxford: Oxford University Press, 2007.

[7] A. A. Feldbaum, "Dual control theory I–IV," *Automation and Remote Control*, vols. 21–22,
1960–1961.

[8] Y. Bar-Shalom and E. Tse, "Dual effect, certainty equivalence, and separation in
stochastic control," *IEEE Transactions on Automatic Control*, vol. 19, no. 5,
pp. 494–500, 1974.

[9] D. P. Bertsekas, *Dynamic Programming and Optimal Control*, Vol. I, 4th ed. Belmont, MA:
Athena Scientific, 2017.

[10] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
Cambridge, MA: MIT Press, 2018.

[11] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*,
vol. 518, pp. 529–533, 2015.

[12] R. K. Mehra, "Optimal input signals for parameter estimation in dynamic systems — survey
and new results," *IEEE Transactions on Automatic Control*, vol. 19, no. 6,
pp. 753–768, 1974.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy
optimization algorithms," arXiv:1707.06347, 2017.
