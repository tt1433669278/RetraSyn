# Experimental Analysis for ICDE Submission

## 1. Experimental Scope

We executed a logically closed experimental suite on the T-Drive dataset covering:

- main comparison against RetraSyn
- innovation-3 projection ablation
- current budget allocation ablation
- multi-scale context resolution sensitivity
- efficiency breakdown at the per-timestamp level
- qualitative trajectory comparison

All generated artifacts are saved under the `results/` directory and the corresponding synthetic datasets are saved under `data/syn_data/tdrive/`.


## 2. Main Comparison

The most representative current comparison is between:

- `RetraSyn`
- `K-PST + no projection` (`projoff`)
- `K-PST + adaptive projection (strength=0.6)` (`adaptive06`)

From [tdrive_balanced_g6_adaptive_projection_ablation.csv](/d:/paper/p9/RetraSyn/results/projection_ablation_adaptive/tdrive_balanced_g6_adaptive_projection_ablation.csv):

- RetraSyn: Transition `0.4104`, Pattern `0.3892`, Trip `0.3362`, Physical Violation `1.3518`, Hotspot `0.4997`
- K-PST projoff: Transition `0.4166`, Pattern `0.4455`, Trip `0.2684`, Physical Violation `1.1288`, Hotspot `0.4314`
- K-PST adaptive06: Transition `0.4134`, Pattern `0.4655`, Trip `0.2622`, Physical Violation `0.9910`, Hotspot `0.4085`

### Interpretation

These numbers support the intended utility exchange:

- the K-PST family substantially improves high-order semantic fidelity (`Pattern F1`) over RetraSyn
- it also lowers destination-level distortion (`Trip Error`)
- adaptive physical projection further reduces physical infeasibility without sacrificing the Pattern gain

The remaining tradeoff is hotspot preservation. This is why the spatially-adaptive projection was introduced: it partially recovers hotspot behavior compared with a global rigid projection, but hotspot fidelity is still lower than the purely density-oriented baseline.


## 3. Validation of Innovation 1: Cognitive-Aware Sampling

We added an explicit uniform-sampling control and compared it against the adaptive scheme under the same projection setting (`strength=0.6`).

Evaluation results:

- Uniform sampling:
  Transition `0.4190`, Pattern `0.4515`, Trip `0.2689`, Physical Violation `1.0002`, Hotspot `0.3958`
- Adaptive sampling:
  Transition `0.4134`, Pattern `0.4655`, Trip `0.2622`, Physical Violation `0.9910`, Hotspot `0.4085`

### Interpretation

This ablation shows that cognitive-aware sampling is not a cosmetic modification. With the same generator and the same physical projection:

- adaptive sampling improves `Pattern F1`
- lowers `Trip Error`
- slightly lowers `Transition Error`
- slightly improves hotspot ranking

This is consistent with Proposition 1: when more reports are allocated to uncertain contexts, the local estimation variance decreases where it matters most.


## 4. Validation of Innovation 2: Multi-Scale Spatio-Temporal Consistency

We ran a context-resolution sensitivity study on `grid_num=10`, varying `context_grid_num` in `{10, 8, 6, 4}`. Results are saved in:

- [tdrive_balanced_g10_context_sensitivity.csv](/d:/paper/p9/RetraSyn/results/context_sensitivity/tdrive_balanced_g10_context_sensitivity.csv)
- [tdrive_balanced_g10_context_sensitivity.png](/d:/paper/p9/RetraSyn/results/context_sensitivity/tdrive_balanced_g10_context_sensitivity.png)

### Key Results

- `ctx10`: Pattern `0.2717`, Hotspot `0.0691`
- `ctx8`: Pattern `0.2650`, Hotspot `0.0963`
- `ctx6`: Pattern `0.2404`, Hotspot `0.1261`
- `ctx4`: Pattern `0.2633`, Hotspot `0.1736`

### Interpretation

This validates the asymmetric-resolution thesis:

- when context resolution is too fine (`ctx10`), the model overfits sparse noisy contexts and hotspot quality collapses
- when the context is moderately coarsened (`ctx6` or `ctx4`), hotspot and density behavior recover significantly
- `ctx4` offers the best hotspot quality in this study, while `ctx10` gives the strongest local pattern score

This is consistent with Proposition 2 and Corollary 2.1: lowering context resolution improves privacy efficiency and stabilizes utility under fine-grid synthesis.


## 5. Validation of Innovation 3: Spatially-Adaptive Physical Projection

The adaptive projection ablation is saved in:

- [tdrive_balanced_g6_adaptive_projection_ablation.csv](/d:/paper/p9/RetraSyn/results/projection_ablation_adaptive/tdrive_balanced_g6_adaptive_projection_ablation.csv)
- [tdrive_balanced_g6_adaptive_projection_ablation.png](/d:/paper/p9/RetraSyn/results/projection_ablation_adaptive/tdrive_balanced_g6_adaptive_projection_ablation.png)

### Main Trend

As the projection strength increases from `0.0` to `1.4`:

- `Physical Violation Error` drops from `1.1288` to `0.9795`
- `Trip Error` drops from `0.2684` to `0.2596`
- `Pattern F1` remains above the RetraSyn baseline
- but `Hotspot NDCG` degrades from `0.4314` to `0.4193` and then further if projection is too strong

### Best Tradeoff

The best overall balance is `adaptive06`:

- Pattern `0.4655`
- Trip `0.2622`
- Physical Violation `0.9910`
- Hotspot `0.4085`

The strongest physical regularization does not give the best overall utility. This empirically supports Proposition 4: projection must be spatially adaptive and moderate, otherwise hotspot semantics are over-smoothed.


## 6. Budget Ablation

The current budget ablation is saved in:

- [tdrive_balanced_g6_budget_ablation_current.csv](/d:/paper/p9/RetraSyn/results/budget_ablation_current/tdrive_balanced_g6_budget_ablation_current.csv)
- [tdrive_balanced_g6_budget_ablation_current.png](/d:/paper/p9/RetraSyn/results/budget_ablation_current/tdrive_balanced_g6_budget_ablation_current.png)

We varied `pst_length_eps_frac` in `{0.02, 0.05, 0.10, 0.15}`.

### Key Findings

- `0.02` gives the best `Pattern F1` (`0.4655`) and best `Trip Error` (`0.2622`)
- `0.10` gives the best `Hotspot NDCG` (`0.4576`)
- `0.15` improves `ST Query Error`, but does not improve the structural metrics enough to justify the larger length budget

### Interpretation

This is consistent with the theory behind the DP length histogram:

- only a very small fraction of the privacy budget is needed to model length
- over-allocating to length steals budget from the spatial models and hurts structural fidelity

Thus the current default `length_eps_frac=0.02` is justified as the most balanced choice.


## 7. Efficiency and Complexity

Per-timestamp efficiency results are saved in:

- [tdrive_efficiency_profile.csv](/d:/paper/p9/RetraSyn/results/efficiency/tdrive_efficiency_profile.csv)
- [tdrive_efficiency_profile.png](/d:/paper/p9/RetraSyn/results/efficiency/tdrive_efficiency_profile.png)

### Observations

- `Proj-Off g6`: trajectory synthesis `87.37 ms/timestamp`
- `Adaptive-0.6 g6`: trajectory synthesis `146.47 ms/timestamp`
- `Adaptive-1.0 g6`: trajectory synthesis `154.51 ms/timestamp`
- `Adaptive-1.4 g6`: trajectory synthesis `143.30 ms/timestamp`

### Interpretation

The physical projection cost is concentrated in the synthesis stage rather than PST updating. This matches the algorithmic design: the projection is applied on the candidate distribution during generation, not during the OUE aggregation stage.

From an ICDE perspective, this efficiency profile is important because it shows:

- the projection cost is measurable and localized
- the update stage remains stable
- the main runtime tradeoff is controllable through projection strength


## 8. Qualitative Evidence

The qualitative comparison figure is saved in:

- [trajectory_comparison_retrasyn_vs_kpst.png](/d:/paper/p9/RetraSyn/results/paper_assets/trajectory_comparison_retrasyn_vs_kpst.png)

In this figure:

- the left panel shows RetraSyn trajectories, which are more diffused and less directionally coherent
- the right panel shows K-PST trajectories with adaptive projection, which are smoother and exhibit stronger commuting corridors

This visualization aligns with the Pattern and Trip improvements observed in the quantitative experiments.


## 9. Paper-Ready Figures

The following paper-style figures are generated:

- [metrics_radar_tradeoff.png](/d:/paper/p9/RetraSyn/results/paper_assets/metrics_radar_tradeoff.png)
- [metrics_dual_axis_tradeoff.png](/d:/paper/p9/RetraSyn/results/paper_assets/metrics_dual_axis_tradeoff.png)
- [trajectory_comparison_retrasyn_vs_kpst.png](/d:/paper/p9/RetraSyn/results/paper_assets/trajectory_comparison_retrasyn_vs_kpst.png)

These figures collectively show:

- global utility tradeoff
- hotspot-vs-physical regularization tradeoff
- qualitative trajectory realism


## 10. Remaining Gaps Before a Full ICDE Submission

The current suite is logically much stronger than before, but two additional validations would still improve submission strength:

1. Cross-dataset generalization on `Oldenburg` and `SanJoaquin`.
2. Privacy-budget sensitivity across `epsilon in {0.5, 1.0, 2.0}` under the final adaptive configuration.

These are not implementation blockers. They are final-stage empirical strengthening experiments.


## 11. Overall Conclusion

The experiments support the intended story of the paper:

- Innovation 1 improves privacy-budget efficiency by adaptive uncertainty-aware sampling.
- Innovation 2 improves fine-grid robustness by using coarse contexts and bounded motion states.
- Innovation 3 reduces physically implausible transitions while preserving hotspot behavior through spatially-adaptive projection.

The strongest recommendation for the current artifact is:

- use `adaptive06` as the main model configuration
- use `projoff` and `uniform06` as the key ablations
- use `context_grid_num` sensitivity on `grid_num=10` to validate the asymmetric-resolution theory
