import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epsilon', type=float, default=1.0,
                    help='Privacy budget')
parser.add_argument('--grid_num', type=int, default=6,
                    help='Number of grids is n x n')
parser.add_argument('--w', type=int, default=20,
                    help='Window size')
parser.add_argument('--method', type=str, default='retrasyn_pst')
parser.add_argument('--dataset', type=str, default='tdrive')
parser.add_argument('--multiprocessing', action='store_true')
parser.add_argument('--phi', type=int, default=20,
                    help='size of evaluation time range')
parser.add_argument('--compare_methods', type=str, default='',
                    help='Comma-separated synthetic methods to compare in evaluation')
parser.add_argument('--pst_profile', type=str, default='balanced',
                    choices=['balanced', 'aggressive'],
                    help='Preset hyperparameter profile for RetraSyn-PST')
parser.add_argument('--pst_sampling_mode', type=str, default='adaptive',
                    choices=['adaptive', 'uniform'],
                    help='Sampling policy for report/explore users')
parser.add_argument('--pst_report_rate', type=float, default=None,
                    help='Confirmed-context report rate multiplier; actual rate is pst_report_rate / w')
parser.add_argument('--pst_explore_rate', type=float, default=None,
                    help='Broadcast-context exploration rate multiplier; actual rate is pst_explore_rate / w')
parser.add_argument('--pst_split_sigma', type=float, default=None,
                    help='Sigma multiplier for PST split significance thresholds')
parser.add_argument('--pst_confirm_gain', type=float, default=None,
                    help='Minimum conditional-distribution gain required for higher-order PST confirmation')
parser.add_argument('--pst_length_eps_frac', type=float, default=None,
                    help='Fraction of epsilon reserved for DP length histogram')
parser.add_argument('--pst_marginal_eps_frac', type=float, default=None,
                    help='Fraction of epsilon reserved for per-timestep marginal calibration')
parser.add_argument('--pst_explore_budget_ratio', type=float, default=None,
                    help='Fraction of remaining PST budget allocated to exploration nodes')
parser.add_argument('--pst_momentum_min_candidates', type=int, default=None,
                    help='Minimum number of momentum-filtered spatial candidates to keep')
parser.add_argument('--pst_consistency_mix', type=float, default=None,
                    help='Mixing strength for hierarchical consistency post-processing')
parser.add_argument('--pst_length_max', type=int, default=None,
                    help='Maximum trajectory length bin used by the DP length histogram')
parser.add_argument('--pst_base_grid_num', type=int, default=6,
                    help='Reference grid resolution used to scale the physical receptive radius')
parser.add_argument('--pst_base_radius', type=int, default=1,
                    help='Reference receptive radius at pst_base_grid_num')
parser.add_argument('--pst_max_radius', type=int, default=3,
                    help='Maximum receptive radius allowed by the adaptive physical field')
parser.add_argument('--pst_context_grid_num', type=int, default=6,
                    help='Coarse-grid resolution used for multi-scale context modeling')
parser.add_argument('--pst_context_depth', type=int, default=3,
                    help='Maximum depth of the coarse-grid context chain')
parser.add_argument('--pst_context_budget_ratio', type=float, default=None,
                    help='Fraction of report/explore PST budget assigned to the coarse context stream')
parser.add_argument('--pst_coarse_mix', type=float, default=None,
                    help='Strength of the coarse-context gate in the joint fine-grid synthesis distribution')
parser.add_argument('--ablation_output', type=str, default='',
                    help='Optional JSONL file used to append resolution-privacy-utility ablation records')
parser.add_argument('--pst_projection_strength', type=float, default=None,
                    help='Soft projection strength used to project noisy transition distributions back to the physical manifold')
parser.add_argument('--pst_projection_feasible_floor', type=float, default=None,
                    help='Minimum residual mass kept for candidates outside the hard feasible set during soft projection')
parser.add_argument('--pst_speed_weight', type=float, default=None,
                    help='Energy weight for speed deviation in the physical feasibility projection')
parser.add_argument('--pst_turn_weight', type=float, default=None,
                    help='Energy weight for turning-angle penalties in the physical feasibility projection')
parser.add_argument('--pst_accel_weight', type=float, default=None,
                    help='Energy weight for acceleration smoothness penalties in the physical feasibility projection')
parser.add_argument('--pst_projection_density_gamma', type=float, default=None,
                    help='Exponent used by spatially-adaptive projection; hotspot cells get weaker projection as density increases')


args = parser.parse_args()

PST_PROFILES = {
    'balanced': {
        'pst_report_rate': 4.50,
        'pst_explore_rate': 1.60,
        'pst_split_sigma': 1.00,
        'pst_confirm_gain': 0.010,
        'pst_length_eps_frac': 0.02,
        'pst_marginal_eps_frac': 0.10,
        'pst_explore_budget_ratio': 0.45,
        'pst_momentum_min_candidates': 5,
        'pst_consistency_mix': 0.18,
        'pst_length_max': 96,
        'pst_context_budget_ratio': 0.24,
        'pst_coarse_mix': 0.34,
        'pst_projection_strength': 2.6,
        'pst_projection_feasible_floor': 0.10,
        'pst_speed_weight': 0.45,
        'pst_turn_weight': 0.75,
        'pst_accel_weight': 0.85,
        'pst_projection_density_gamma': 1.35,
    },
    'aggressive': {
        'pst_report_rate': 5.50,
        'pst_explore_rate': 3.00,
        'pst_split_sigma': 0.75,
        'pst_confirm_gain': 0.006,
        'pst_length_eps_frac': 0.02,
        'pst_marginal_eps_frac': 0.08,
        'pst_explore_budget_ratio': 0.65,
        'pst_momentum_min_candidates': 4,
        'pst_consistency_mix': 0.10,
        'pst_length_max': 128,
        'pst_context_budget_ratio': 0.30,
        'pst_coarse_mix': 0.42,
        'pst_projection_strength': 3.0,
        'pst_projection_feasible_floor': 0.06,
        'pst_speed_weight': 0.35,
        'pst_turn_weight': 0.85,
        'pst_accel_weight': 1.00,
        'pst_projection_density_gamma': 1.15,
    }
}

profile_defaults = PST_PROFILES[args.pst_profile]
for key, value in profile_defaults.items():
    if getattr(args, key) is None:
        setattr(args, key, value)
