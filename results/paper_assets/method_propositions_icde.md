# Method and Propositions

## 1. Method Overview

We model stream trajectory synthesis under local differential privacy (LDP) as a three-factor conditional generation problem. Let `c_t` denote the fine-grid location at timestamp `t`, `psi(c_t)` denote its coarse-grid projection, and `delta_t` denote the bounded relative motion token. The proposed generator maintains:

- a fine-grid absolute prior `P_abs(c_t)`
- a motion-prefix tree `P_motion(delta_t | H_t^m)`
- a coarse-context model `P_coarse(psi(c_t) | H_t^c)`

where:

- `H_t^m = (delta_{t-k_m}, ..., delta_{t-1})`
- `H_t^c = (psi(c_{t-k_c}), ..., psi(c_{t-1}))`

The generation rule is:

`P(c_t | H_t^m, H_t^c) proportional to P_abs(c_t)^alpha P_motion(delta_t | H_t^m)^beta P_coarse(psi(c_t) | H_t^c)^gamma`

This factorization decouples macro-intent and micro-motion: the coarse stream preserves long-range destination semantics, while the motion stream preserves local kinematic regularity.


## 2. Cognitive-Aware LDP Sampling

For each active user, the center estimates uncertainty from the matched motion node and coarse node. Let `H_norm` be normalized entropy and `r(.)` be node reliability. We define:

`U_m = lambda_h H_norm(P_motion(. | H_t^m)) + lambda_r (1 - r(H_t^m))`

`U_c = lambda_h H_norm(P_coarse(. | H_t^c)) + lambda_r (1 - r(H_t^c))`

The report and explore weights are:

`w_r = eta_0 + eta_1 U_m + eta_2 U_c + eta_3 (1 - \bar r)`

`w_e = zeta_0 + I_new (zeta_1 U_m^b + zeta_2 U_c^b + zeta_3 N + zeta_4 G)`

where `N` is the broadcast-confirmation novelty and `G` is the normalized branch-gain signal. The effective sampling rates are adaptively scaled by the average uncertainty signals over active users.

### Proposition 1

Under OUE, if the reporting weights are monotone with respect to conditional uncertainty and inverse reliability, then the cognitive-aware allocation reduces the weighted aggregate estimation variance compared with uniform sampling under the same total reporting budget.

### Proof Sketch

For OUE, the per-coordinate variance is inversely proportional to the effective local sample size. Reallocating user reports from low-uncertainty contexts to high-uncertainty contexts increases the local sample size where the posterior variance is largest, thereby reducing the global weighted variance.


## 3. Multi-Scale Spatio-Temporal Consistency

The coarse-grid mapping `psi : C_f -> C_c` introduces an asymmetric-resolution Markov dependency: low-resolution contexts are used to predict high-resolution targets.

### Proposition 2

The proposed factorization reduces the effective context complexity from `O(|C_f|^k)` to `O(|C_c|^{k_c} + |Delta|^{k_m})`, while retaining fine-grid prediction in the target space.

### Proof Sketch

The absolute high-order PST grows exponentially with the fine-grid resolution. Replacing absolute high-order contexts with a coarse absolute chain and a bounded relative-motion chain compresses the state space without compressing the prediction space. Under LDP, this increases the effective sample size per context and improves the signal-to-noise ratio.

### Corollary 2.1

Let `S_f = |C_f|^k` be the fine absolute context size and `S_ms = |C_c|^{k_c} |Delta|^{k_m}` be the proposed multi-scale context size. Then the context-level signal-to-noise gain is approximately:

`SNR_gain approx S_f / S_ms`

This gives a direct resolution-privacy-utility tradeoff: lowering context resolution improves privacy efficiency without sacrificing the prediction resolution.


## 4. Spatially-Adaptive Physical Projection

To suppress LDP-induced teleportation while preserving hotspot behavior, we project the joint candidate distribution onto a soft physical manifold.

For each candidate motion `delta`, define the physical energy:

`E_t(delta) = lambda_v * | ||delta|| - ||delta_{t-1}|| | / R + lambda_theta * (1 - cos(theta))/2 + lambda_a * (0.7 ||delta-delta_{t-1}||/A + 0.3 ||(delta-delta_{t-1})-(delta_{t-1}-delta_{t-2})||/A )`

Let `D_t(c)` be the calibrated hotspot density. The adaptive projection strength is:

`Lambda_t(c) = alpha * (1 - D_t(c) / max_x D_t(x))^gamma`

Thus, hotspot cells receive a weaker physical constraint, while cold cells retain a stronger physical regularization. The projected distribution is:

`P_phys(c) proportional to P_joint(c) exp(-Lambda_t(c) E_t(c)) I_feasible(c) + P_joint(c) F_t(c) (1-I_feasible(c))`

where `F_t(c)` is the hotspot-adaptive feasible floor.

### Proposition 3

For any fixed pre-projection distribution `P_joint`, the expected physical energy under the projected distribution is no larger than that under `P_joint`, up to the residual mass preserved by the feasible-floor term.

### Proof Sketch

The projection is an energy-based exponential tilting. Candidates with larger energy are reweighted downward more strongly. Hence the expectation of the energy under the projected distribution is reduced unless the residual feasible-floor dominates.

### Proposition 4

If `D_t(c)` is close to the hotspot maximum, then `Lambda_t(c)` approaches zero and the projection approaches identity on candidate `c`. Therefore hotspot-specific behaviors, such as short-distance turning and local wandering, are not over-smoothed.


## 5. Practical Consequences

The complete framework jointly contributes:

- Cognitive-aware sampling improves budget efficiency in high-uncertainty contexts.
- Multi-scale consistency stabilizes fine-grid prediction under coarse contexts.
- Spatially-adaptive physical projection lowers kinematic violations without globally destroying hotspot semantics.

This combination is especially suitable for streaming LDP trajectory synthesis, where privacy noise, sparse high-order states, and physical infeasibility interact simultaneously.
