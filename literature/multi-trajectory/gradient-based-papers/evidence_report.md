## PDF Evidence Report

Here are the exact quoted passages and equations supporting each claim, with page numbers.

---

### 1. *Trajectory-based actuator identification via differentiable simulation*

**Claim:** Diagonal weight matrix, velocity excluded from loss, Adam optimizer.

**SUPPORTED — all three elements confirmed.**

**Weighted loss & velocity exclusion (Page 5, Eq. 2):**

> "We minimize a weighted output mismatch without any parameter regularization to avoid biasing physically meaningful parameters (e.g., PD gains)."
>
> **Equation (2):**
> $$L_\text{batch}(z) = \frac{1}{MN}\sum_{j=0}^{M-1}\sum_{i=1}^{N}\left\|W\left(s'_{i,j}-s_{i,j}\right)\right\|_2^2$$
>
> "where $W = \text{diag}(w_q, w_{\dot q})$ is a diagonal weight matrix penalizing angle and velocity residuals. Thus, **state components may be present in the rollout state and model inputs even when their residuals are not penalized in the loss**."

**Explicit weight values & velocity exclusion (Page 12 / Appendix B):**

> "In Eq. (2), we set $W = \text{diag}(1, 0)$, i.e., $w_q = 1$, $w_{\dot q} = 0$, so the objective penalizes position error only. **Velocity remains part of the observed state and simulator rollout, but velocity residuals are not penalized** because the measured velocity signal is noticeably noisier than position."

**Adam optimizer (Page 12 / Appendix B):**

> "All identified models minimize the segmented trajectory-matching objective in Eq. (2) using **Adam (Optax) with learning rate $10^{-2}$**."

---

### 2. *Dynamic Modeling of Robotic Manipulator via an Augmented Deep Lagrangian Network*

**Claim:** Mahalanobis/diagonal covariance loss, because residuals differ in magnitude between joints/channels.

**SUPPORTED.**

**Equation and explanation (Page 4, Eq. 8):**

> $$(\theta^*, \phi^*) = \arg\min_{\theta,\phi} \left\|\hat{f}^{-1}(q, \dot q, \ddot q;\theta,\phi) - \tau_R\right\|^2_{W_{\tau_R}}$$
>
> "where $\tau_R$ represents the real torque collected from the physical manipulator, $\|\cdot\|_W$ represents the **Mahalanobis norm**, and $W_{\tau_R}$ represents the **diagonal covariance matrix of the generalized forces**. **It is necessary to normalize the loss function using covariance matrix since the torque magnitude may vary greatly from joint to joint.**"

*(Page 4 of the PDF, surrounding Eq. 8)*

---

### 3. *Constrained Gray-Box Identification of Electromechanical Systems Under Unfiltered Step-Response Data*

**Claim:** Normalized composite residual from heterogeneous signals (current, velocity, steady-state); signal normalization applied.

**SUPPORTED.**

**Procedure and equation (Pages 6–7, Eq. 3 and surrounding text):**

The abstract states: "The method estimates all electromechanical parameters by minimizing a **normalized residual that combines current, velocity, and steady-state algebraic constraints**."

The optimization problem (Eq. 3, Page 6):
$$\hat\theta = \arg\min_{\theta>0} \|r(\theta)\|_2^2$$

The residual composition (Page 7, Procedure step 3):

> "**Residual composition:** construct $r(\theta)$ as the concatenation of
> - **normalized trajectory errors:** $\alpha_\omega(\omega_\text{sim}-\omega)/\text{RMS}(\omega)$ and $\alpha_i(i_\text{sim}-i)/\text{RMS}(i)$;
> - **steady-state penalties:** $\lambda_\text{ss}[\omega_\text{ss}(\theta)-\bar\omega]$ and $\lambda_\text{ss}[i_\text{ss}(\theta)-\bar i]$ based on (4);
> - **current-limit penalty:** $\lambda_\text{pk}\max(0, \max_t i_\text{sim}(t)-V/R)$."

Explanation of normalization (Page 7):

> "The trajectory terms $e_\omega(t)$ and $e_i(t)$ are **normalized by the RMS of their corresponding measured signals, which naturally balances the relative contribution of current and velocity**; thus $\alpha_\omega = \alpha_i = 1$ is sufficient and avoids additional manual scaling."

---

### 4. *Real-time Model Predictive Control and System Identification Using Differentiable Physics Simulation*

**Claim:** Gradient-based physical parameter identification is performed in closed loop using real observations.

**SUPPORTED.**

**SysID method with gradient-based optimization (Page 3):**

> "Given a sequence of recently observed states containing position and velocity in generalized coordinate, $x_{0:H} = [q_{0:H}, \dot q_{0:H}]$, in the history buffer $H$, a standard SysID routine finds the optimal system parameters that best fit the observations:
> $$\mu^* = \arg\min_\mu \|\text{sim}(q_0, \dot q_0, u_{0:H}; \mu, H) - q_{0:H}\|,$$
> where $\text{sim}(q_0, \dot q_0, u_{0:H}; \mu, H)$ is the forward simulation … Utilizing a **differential physics simulator (e.g. NimblePhysics), we can compute the gradients of the objective function efficiently to optimize $\mu$ in real time.**"

**Real observations / closed-loop (Page 3):**

> "The modeling thread takes the **state sequence in the history buffer $H$, which stores the most recent observed states from the target environment**, optimizes $\mu$ to match $H$ via the gradients provided by the differentiable physics engine."

> **Note on optimizer name:** The specific optimizer name (e.g., Adam, SGD) is **NOT FOUND** in the text. The paper describes gradient-based minimization via the differentiable physics engine but does not name the optimizer explicitly.

---

### 5. *Combining Physics and Deep Learning to learn Continuous-Time Dynamics Models*

**Claim:** Deep Lagrangian Networks — loss = squared residual of the Euler-Lagrange equation, with diagonal covariance/Mahalanobis norm.

**SUPPORTED.**

**Loss equation (Page 7 of PDF, Eq. 12):**

> "One approach to learn the network parameters is to minimize the residual of the Euler-Lagrange differential equation. This optimization problem is described by
> $$\psi^*, \phi^* = \arg\min_{\psi,\phi}\left\|\frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot q} - \frac{\partial\mathcal{L}}{\partial q_i} - \tau\right\|^2_{W_\tau},\quad(12)$$
> with the **Mahalanobis norm** $\|\cdot\|_W$ and the **diagonal covariance matrix of the generalized forces** $W_\tau$. **It is beneficial to normalize the loss using the covariance matrix because magnitude of the residual might vary between different joints.** This optimization can be solved using any gradient-based optimization technique."

---

### 6. *Differentiable Simulation for Physical System Identification*

**Claim:** Differentiable physics framework estimates friction coefficients and masses via backpropagation; loss equation.

**SUPPORTED.**

**Parameters estimated (Page 1, Abstract and Fig. 1 caption):**

> "In particular, using our approach we demonstrate **accurate estimation of friction coefficients and object masses** both in synthetic and real experiments."

Fig. 1 caption:

> "The differentiability of the simulator allows to integrate it into a larger learning architecture and **infer physical parameters such as friction coefficients $\mu$ and mass $M$ of the objects**, from real trajectories of these objects."

**Loss equation and Adam optimizer (Page 6, Section IV-B):**

> "We note the 'simulator function' $g_\mu : x_t \mapsto \hat x_{t+1}$ whose computational graph corresponds to the Algorithm 1 and whose only unknown parameter is $\mu$. Then, we can define the MSE loss:
> $$\mathcal{L}(\mu) = \sum_{t=1}^{T}\|x_t - \hat x_t\|_2^2 = \sum_{t=1}^{T}\|x_t - g_\mu(x_{t-1})\|_2^2$$
> which is the sum of the errors made by the simulator at each time step. Using the **differentiability of the 'simulator function' $g_\mu$ with respect to $\mu$, it is possible to compute $\nabla_\mu\mathcal{L}$ by back-propagating the loss using the Automatic Differentiation tool of PyTorch**. Then, we minimize $\mathcal{L}$ with respect to $\mu$ using **Adam algorithm**."

---

### 7. *Physics-informed online learning of gray-box models by moving horizon estimation*

**Claim:** Physical submodel combined with trainable data-driven part; training uses BPTT; arrival cost covariance interpreted as adaptive learning rate. Published in *European Journal of Control*.

**SUPPORTED.**

**Journal confirmation (Page 1, header):** "European Journal of Control 74 (2023) 100861"

**Physical submodel + data-driven part (Abstract, Page 1):**

> "A potential solution is the use of a **physics-informed, or gray-box model that extends a physics-based model with a data-driven part**. Learning the latter might be challenging, due to noisy measurements and lack of full state information. This work presents a method based on Moving Horizon Estimation (MHE) for simultaneous state estimation and training of a black-box submodel, such as a neural network."

**BPTT (Page 3, Section 3):**

> "Such a reduced problem can be solved using different nonlinear programming solvers, but requires processing the entire data set in one shot, as the cost function of the problem is not separable due to the dynamic constraints in (7b). **The gradient of the cost function can be evaluated efficiently by backpropagation through time**, although this could lead to the problem of vanishing or exploding gradients."

*(The paper cites Reference [29]: P.J. Werbos, "Backpropagation through time: what it does and how to do it," Proc. IEEE 78(10) (1990).)*

**Arrival cost covariance = adaptive learning rate (Pages 3–4):**

> "In our setup (5), **the part of the arrival cost related to the neural network parameters $w$ can be seen as an adaptive learning-rate**. It can be tuned by setting the covariance on the neural network parameters, $Q_w$, in the arrival cost update."

---

### Summary Table

| # | Claim | Status |
|---|-------|--------|
| 1 | Diagonal W, velocity excluded from loss, Adam optimizer | ✅ All confirmed (Eq. 2, Appendix B, p. 5 & 12) |
| 2 | Mahalanobis/diagonal covariance, magnitude varies by joint | ✅ Confirmed (Eq. 8, p. 4) |
| 3 | Normalized composite residual (current, velocity, steady-state) | ✅ Confirmed (Eq. 3, Procedure, p. 6–7) |
| 4 | Gradient-based ID in closed loop from real observations | ✅ Confirmed (p. 3); specific optimizer name **NOT FOUND** |
| 5 | Loss = squared EL residual, diagonal covariance (DeLaN) | ✅ Confirmed (Eq. 12, p. 7) |
| 6 | Friction & mass via backprop; MSE loss + Adam | ✅ Confirmed (Section IV-B, p. 6) |
| 7 | BPTT, arrival cost = adaptive LR, European J. Control | ✅ All confirmed (p. 1, 3–4) |
