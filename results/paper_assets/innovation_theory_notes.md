# RetraSyn-PST Innovation Notes

## Innovation 1: Cognitive-Aware Sampling

### Definition

Let `H_t^m` be the motion-history context matched by the K-PST and `H_t^c` be the coarse-grid context matched by the multi-scale context stream. For a user `u` active at time `t`, define:

`U_m(u,t) = lambda_h * H_norm(P_m(. | H_t^m)) + lambda_r * (1 - r_m(H_t^m))`

`U_c(u,t) = lambda_h * H_norm(P_c(. | H_t^c)) + lambda_r * (1 - r_c(H_t^c))`

where:

- `H_norm` is normalized entropy
- `r_m`, `r_c` are reliability weights of the motion and coarse nodes

Let novelty be:

`N(u,t) = max(depth(B_t(u)) - depth(C_t(u)), 0)`

where `B_t(u)` is the longest broadcast match and `C_t(u)` is the longest confirmed match.

We define report and exploration weights:

`w_r(u,t) = eta_0 + eta_1 U_m(u,t) + eta_2 U_c(u,t) + eta_3 (1 - \bar r(u,t))`

`w_e(u,t) = zeta_0 + I_new(u,t) * (zeta_1 U_m^b(u,t) + zeta_2 U_c^b(u,t) + zeta_3 N(u,t) + zeta_4 G(u,t))`

where `\bar r(u,t)` is the average reliability of the matched nodes and `G(u,t)` is the normalized branch-gain signal.

The effective sampling rates are then dynamically adjusted by entropy and novelty:

`rho_r(t) = rho_r^0 * Phi_r(\bar U_m(t), \bar U_c(t), \bar w_r(t))`

`rho_e(t) = rho_e^0 * Phi_e(\bar U_m(t), \bar U_c(t), \bar N(t), \bar w_e(t))`

### Proposition 1

Under the same total reporting budget, if the sampling weights are positively correlated with conditional uncertainty and negatively correlated with node reliability, then the weighted estimator allocates more samples to high-variance contexts and reduces the aggregate mean squared error of the conditional distributions compared with uniform sampling.

### Proof Sketch

For OUE, the per-coordinate variance is `Var(eps, n) = 4 exp(eps) / (n (exp(eps)-1)^2)`. Since the variance is inversely proportional to the effective local sample size `n`, moving samples from low-uncertainty nodes to high-uncertainty nodes lowers the weighted sum `sum_x q_x Var_x` whenever the weights are aligned with the uncertainty profile.


## Innovation 2: Multi-Scale Spatio-Temporal Consistency

### State Definition

Let the fine-grid domain be `C_f` with size `|C_f| = g^2`, the coarse-grid domain be `C_c` with size `|C_c| = g_c^2`, and the motion token domain be `Delta`.

Define a surjective coarse mapping:

`psi : C_f -> C_c`

The two context streams are:

`H_t^c = (psi(c_{t-k_c}), ..., psi(c_{t-1}))`

`H_t^m = (delta_{t-k_m}, ..., delta_{t-1})`

where `delta_t = c_t - c_{t-1}` is encoded as a bounded relative motion token.

### Joint Generation Rule

The fine-grid next-step distribution is synthesized as:

`P(c_t | H_t^c, H_t^m) proportional to P_abs(c_t)^alpha * P_motion(delta_t | H_t^m)^beta * P_coarse(psi(c_t) | H_t^c)^gamma`

Here:

- `P_abs` is the fine-grid absolute transition prior
- `P_motion` is the K-PST motion distribution
- `P_coarse` is the coarse-grid context model

### Proposition 2

Compared with an absolute high-order PST over the fine-grid domain, the proposed multi-scale factorization reduces the effective context dimension from `O(|C_f|^k)` to `O(|C_c|^{k_c} + |Delta|^{k_m})`, while preserving high-resolution prediction in the target space.

### Proof Sketch

The original high-order absolute context uses a state space whose cardinality grows exponentially with the fine-grid resolution. In contrast, the proposed factorization replaces the context space with a coarse absolute chain and a bounded relative-motion chain. Since `|C_c| << |C_f|` and `|Delta|` is bounded by the receptive field radius rather than `g^2`, the context space is substantially compressed. Under OUE, the same total sample size is redistributed over fewer contexts, so the effective sample size per context increases and the estimation noise decreases.

### Corollary 2.1

If `|C_c| < |C_f|`, then the signal-to-noise ratio of the context estimator improves by a factor proportional to the ratio of the state-space sizes:

`SNR_gain approx (|C_f|^k) / (|C_c|^{k_c} * |Delta|^{k_m})`

This provides the resolution-privacy-utility tradeoff: lowering the context resolution improves privacy efficiency without sacrificing fine-grid prediction resolution.


## Innovation 3: Noise-Robust Physical Semantic Projection

### Physical Energy

For each candidate next motion `delta`, define the physical energy:

`E_t(delta) = lambda_v * | ||delta|| - ||delta_{t-1}|| | / R`

`+ lambda_theta * (1 - cos(theta(delta, delta_{t-1}))) / 2`

`+ lambda_a * (0.7 * ||delta - delta_{t-1}|| / A + 0.3 * ||(delta - delta_{t-1}) - (delta_{t-1} - delta_{t-2})|| / A )`

where:

- `R` is the physical receptive radius
- `A` is the acceleration scale
- the three terms measure speed deviation, turning cost, and acceleration/jerk cost

### Spatially-Adaptive Projection

Let `D_t(c)` be the hotspot density from the marginal calibrator. The projection strength for candidate grid `c` is:

`Lambda_t(c) = alpha * (1 - D_t(c) / max_x D_t(x))^gamma`

This makes projection strong on cold road cells and weak on hotspot cells.

We also define a hotspot-adaptive feasible-floor:

`F_t(c) = f_0 + kappa * (D_t(c) / max_x D_t(x)) * (1 - f_0)`

Then the projected conditional distribution is:

`P_phys(c_t = c | H_t^m, H_t^c) proportional to P_joint(c) * exp(-Lambda_t(c) E_t(c)) * I_feasible(c)`

`+ P_joint(c) * F_t(c) * (1 - I_feasible(c))`

followed by normalization.

### Proposition 3

For any fixed pre-projection distribution `P_joint`, the exponentially tilted projection decreases the expected physical energy:

`E_{P_phys}[E_t] <= E_{P_joint}[E_t]`

whenever `Lambda_t(c) >= 0`.

### Proof Sketch

The projection is an energy-based reweighting of the candidate distribution. Since each candidate is down-weighted by `exp(-Lambda_t(c) E_t(c))`, larger-energy candidates receive weaker mass after projection. Therefore the expectation of the energy under the reweighted distribution is no greater than that under the original distribution, up to the residual feasible-floor mass on infeasible states.

### Proposition 4

If `D_t(c)` is large, then `Lambda_t(c)` becomes small, so the projection becomes nearly identity on hotspot cells. Therefore hotspot-specific behaviors such as stopping, turning, and short-distance wandering are preserved instead of being globally over-smoothed.

### Practical Interpretation

The physical projection is not a global rigid constraint. It is a spatially adaptive manifold regularizer:

- on trunk roads, it suppresses LDP-induced teleportation and unstable sharp turns
- in hotspots, it relaxes the regularizer and preserves destination-area stop-and-turn behavior


## Implementation Mapping

- Cognitive-aware sampling: `cognitive_sampling_weights`, `adaptive_sampling_rates`
- Multi-scale context model: `CoarseGridMapper`, `CoarseContextModel`, `joint_motion_cdf`
- Physical projection: `physical_feasibility_projection`

These three components together form a unified privacy-efficient, multi-scale, and physically grounded trajectory synthesis framework.
