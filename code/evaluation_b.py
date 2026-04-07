import pickle

import utils
from grid import Grid, GridMap
from parse import args
import multiprocessing
import random
import numpy as np
from typing import List, Tuple
import json
import experiment
from logger.logger import ConfigParser
import lzma
from pathlib import Path

config = ConfigParser(name='evaluation', save_dir='./')
logger = config.get_logger(config.exper_name)
CORES = multiprocessing.cpu_count() // 2
random.seed(2023)
np.random.seed(2023)


def synthetic_method_tag(method_name: str):
    if method_name == 'retrasyn_pst':
        return f'{method_name}_{args.pst_profile}'
    return method_name


def spatial_decomposition(db: List[List[Tuple[float, float, int]]], gm: GridMap, multi=False):
    if multi:
        def decomp_multi(xy_l: List[Tuple[float, float, int]]):
            return utils.xyt2grid(xy_l, gm)

        pool = multiprocessing.Pool(CORES)
        grid_db = pool.map(decomp_multi, db)
        pool.close()
    else:
        grid_db = [utils.xyt2grid(traj, gm) for traj in db]

    return grid_db


def split_traj_db(grid_db: List[List[Tuple[Grid, int]]], gm: GridMap):
    def split_traj(grid_t: List[Tuple[Grid, int]]):
        new_trajs = []
        split_id = []
        for i in range(len(grid_t) - 1):
            curr_grid = grid_t[i][0]
            next_grid = grid_t[i + 1][0]
            if not (curr_grid.equal(next_grid) or gm.is_adjacent_grids(curr_grid, next_grid)):
                split_id.append(i + 1)
        if not len(split_id):
            return [grid_t]

        start_id = 0
        for sid in split_id:
            new_trajs.append(grid_t[start_id:sid])
            start_id = sid
        new_trajs.append(grid_t[sid:])
        return new_trajs

    new_grid_db = []
    for traj in grid_db:
        new_grid_db.extend(split_traj(traj))

    return new_grid_db


def resolve_methods():
    if args.compare_methods.strip():
        methods = [method.strip() for method in args.compare_methods.split(',') if method.strip()]
    elif args.method == 'retrasyn_pst':
        methods = ['retrasyn', synthetic_method_tag('retrasyn_pst')]
    else:
        methods = [args.method]

    deduped = []
    seen = set()
    for method in methods:
        if method == 'retrasyn_pst':
            method = synthetic_method_tag(method)
        if method not in seen:
            deduped.append(method)
            seen.add(method)
    return deduped


def load_syn_db(method_name: str):
    syn_path = PROJECT_ROOT / 'data' / 'syn_data' / args.dataset / f'{method_name}_{args.epsilon}_g{args.grid_num}_w{args.w}.pkl'
    if not syn_path.exists():
        raise FileNotFoundError(f'Synthetic data file not found: {syn_path}')
    with open(syn_path, 'rb') as f:
        return pickle.load(f)


def load_syn_metadata(method_name: str):
    meta_path = PROJECT_ROOT / 'data' / 'syn_data' / args.dataset / f'{method_name}_{args.epsilon}_g{args.grid_num}_w{args.w}.meta.json'
    if not meta_path.exists():
        return {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_points_by_time(db: List[List[Tuple[float, float, int]]], max_time: int):
    buckets = [[] for _ in range(max_time)]
    for traj in db:
        for x, y, t in traj:
            if 0 <= t < max_time:
                buckets[t].append((x, y))
    return [np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float) for points in buckets]


def count_query_answers(points_by_time, queries):
    answers = np.zeros(len(queries), dtype=float)
    for q_idx, query in enumerate(queries):
        total = 0
        for t in range(query.min_t, query.max_t + 1):
            points = points_by_time[t]
            if points.size == 0:
                continue
            mask = (
                (points[:, 0] >= query.left_x)
                & (points[:, 0] <= query.right_x)
                & (points[:, 1] >= query.down_y)
                & (points[:, 1] <= query.up_y)
            )
            total += int(np.count_nonzero(mask))
        answers[q_idx] = total
    return answers


def evaluate_syn_dataset(method_name: str,
                         syn_db: List[List[Tuple[float, float, int]]],
                         syn_grid_db: List[List[Tuple[Grid, int]]],
                         actual_st_ans: np.ndarray,
                         st_queries,
                         average_total_points: float,
                         orig_pattern_cache,
                         pattern_windows,
                         orig_counts,
                         orig_prefix_counts,
                         grid_domain,
                         normal_transitions,
                         max_time,
                         upt):
    del upt

    logger.info(f'Evaluating method: {method_name}')

    syn_counts = experiment.get_grid_count(syn_grid_db, grid_domain, max_time=max_time)
    syn_density = syn_counts / (syn_counts.sum(axis=1, keepdims=True) + 1e-10)
    density_results = experiment.eval_jsd(orig_density, syn_density)

    syn_trans = experiment.get_transition_count(syn_grid_db, normal_transitions, max_time=max_time)
    syn_distribution = syn_trans / (syn_trans.sum(axis=1, keepdims=True) + 1e-10)
    transition_results = experiment.eval_jsd(orig_distribution, syn_distribution)

    syn_points_by_time = build_points_by_time(syn_db, max_time)
    syn_st_ans = count_query_answers(syn_points_by_time, st_queries)
    st_query_error = np.abs(actual_st_ans - syn_st_ans) / np.maximum(actual_st_ans, average_total_points * 0.01)

    pattern_scores = []
    for idx, (min_time, max_time_window) in enumerate(pattern_windows):
        syn_pattern = experiment.mine_patterns(syn_grid_db, min_time, max_time_window)
        pattern_scores.append(experiment.calculate_pattern_f1(orig_pattern_cache[idx], syn_pattern))

    kendall_tau = experiment.calculate_coverage_kendall_tau(orig_grid_db, syn_grid_db, grid_map)
    length_err = experiment.calculate_length_error(orig_db, syn_db)
    trip_err = experiment.calculate_trip_error(orig_grid_db, syn_grid_db, grid_map)
    syn_physical_violation = experiment.calculate_physical_violation(syn_grid_db, grid_map)
    physical_violation_error = abs(syn_physical_violation - orig_physical_violation) / max(orig_physical_violation, 1e-8)

    syn_prefix_counts = np.vstack([np.zeros((1, syn_counts.shape[1])), np.cumsum(syn_counts, axis=0)])
    hotspot_scores = []
    for min_time, max_time_window in hotspot_windows:
        orig_total_counts = orig_prefix_counts[max_time_window + 1] - orig_prefix_counts[min_time]
        syn_total_counts = syn_prefix_counts[max_time_window + 1] - syn_prefix_counts[min_time]
        hotspot_scores.append(experiment.eval_hotspot_ndcg(orig_total_counts, syn_total_counts))

    return {
        'Density Error': float(density_results),
        'Transition Error': float(transition_results),
        'Spatial-Temporal Query Error': float(np.mean(st_query_error)),
        'Pattern F1 Error': float(np.mean(pattern_scores)),
        'Kendall-tau Coefficient': float(kendall_tau),
        'Length Error': float(length_err),
        'Trip Error': float(trip_err),
        'Physical Violation Error': float(physical_violation_error),
        'Hotspot NDCG': float(np.mean(hotspot_scores)),
    }


logger.info(args)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
orig_file = PROJECT_ROOT / 'data' / f'{args.dataset}.xz'
methods_to_eval = resolve_methods()

orig_db: List[List[Tuple[float, float, int]]]

with lzma.open(orig_file, 'rb') as f:
    orig_db = pickle.load(f)

with open(PROJECT_ROOT / 'data' / f'{args.dataset}_stats.json', 'r') as f:
    stats = json.load(f)

grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])
logger.info('Spatial decomposition...')
if args.multiprocessing:
    def decomp_multi(xy_l: List[Tuple[float, float, int]]):
        return utils.xyt2grid(xy_l, grid_map)


    if args.dataset == 'sanjoaquin':
        # dataset is too large, use smaller cores to avoid memory error
        CORES = 5

    pool = multiprocessing.Pool(CORES)
    orig_grid_db = pool.map(decomp_multi, orig_db)
    pool.close()
else:
    orig_grid_db = [utils.xyt2grid(traj, grid_map) for traj in orig_db]

orig_grid_db = split_traj_db(orig_grid_db, grid_map)

if args.dataset == 'oldenburg':
    max_time = 500
    # average user per timestamp
    upt = 34000
elif args.dataset == 'tdrive':
    max_time = 886
    upt = 3821
elif args.dataset == 'sanjoaquin':
    max_time = 1000
    upt = 56749

logger.info('Experiment: Density')
grid_domain = grid_map.get_list_map()
normal_transitions = grid_map.get_normal_transition()
orig_counts = experiment.get_grid_count(orig_grid_db, grid_domain, max_time=max_time)
orig_density = orig_counts / (orig_counts.sum(axis=1, keepdims=True) + 1e-10)

logger.info('Experiment: Transition')
orig_trans = experiment.get_transition_count(orig_grid_db, normal_transitions, max_time=max_time)
orig_distribution = orig_trans / (orig_trans.sum(axis=1, keepdims=True) + 1e-10)

logger.info('Experiment: Spatial-Temporal Query Error...')
st_queries = [experiment.SquareQuery(grid_map.min_x, grid_map.min_y, grid_map.max_x, grid_map.max_y, max_time,
                                         time_range=args.phi) for _ in
                  range(100)]
average_total_points = upt * (st_queries[0].max_t - st_queries[0].min_t + 1)
orig_points_by_time = build_points_by_time(orig_db, max_time)
actual_st_ans = count_query_answers(orig_points_by_time, st_queries)

random.seed(2023)
np.random.seed(2023)
logger.info('Experiment: Pattern Errors')
min_times = [random.randint(0, max_time - args.phi) for _ in range(100)]
max_times = [m_t + args.phi - 1 for m_t in min_times]
pattern_windows = list(zip(min_times, max_times))
orig_pattern_cache = [
    experiment.mine_patterns(orig_grid_db, min_time, max_time_window)
    for min_time, max_time_window in pattern_windows
]

random.seed(2023)
np.random.seed(2023)
logger.info('Experiment: Hotspot NDCG')
min_times = [random.randint(0, max_time - args.phi) for _ in range(100)]
max_times = [m_t + args.phi - 1 for m_t in min_times]
hotspot_windows = list(zip(min_times, max_times))
orig_prefix_counts = np.vstack([np.zeros((1, orig_counts.shape[1])), np.cumsum(orig_counts, axis=0)])
orig_physical_violation = experiment.calculate_physical_violation(orig_grid_db, grid_map)

results_by_method = {}
metadata_by_method = {}
for method_name in methods_to_eval:
    syn_db = load_syn_db(method_name)
    metadata_by_method[method_name] = load_syn_metadata(method_name)

    if args.multiprocessing:
        if args.dataset == 'sanjoaquin':
            CORES = 5
        pool = multiprocessing.Pool(CORES)
        syn_grid_db = pool.map(decomp_multi, syn_db)
        pool.close()
    else:
        syn_grid_db = [utils.xyt2grid(traj, grid_map) for traj in syn_db]

    results_by_method[method_name] = evaluate_syn_dataset(
        method_name,
        syn_db,
        syn_grid_db,
        actual_st_ans,
        st_queries,
        average_total_points,
        orig_pattern_cache,
        pattern_windows,
        orig_counts,
        orig_prefix_counts,
        grid_domain,
        normal_transitions,
        max_time,
        upt
    )

metric_order = [
    'Density Error',
    'Transition Error',
    'Spatial-Temporal Query Error',
    'Pattern F1 Error',
    'Kendall-tau Coefficient',
    'Length Error',
    'Trip Error',
    'Physical Violation Error',
    'Hotspot NDCG',
]

for method_name, metrics in results_by_method.items():
    logger.info(f'===== {method_name} =====')
    for metric_name in metric_order:
        logger.info(f'{metric_name}: {metrics[metric_name]}')
    metadata = metadata_by_method.get(method_name, {})
    if metadata:
        logger.info(
            'Ablation metadata: '
            f"context_grid_num={metadata.get('context_grid_num')}, "
            f"context_ratio={metadata.get('context_ratio')}, "
            f"length_eps={metadata.get('length_eps')}, "
            f"marginal_eps={metadata.get('marginal_eps')}, "
            f"motion_report_eps={metadata.get('motion_report_eps')}, "
            f"motion_explore_eps={metadata.get('motion_explore_eps')}, "
            f"context_report_eps={metadata.get('context_report_eps')}, "
            f"context_explore_eps={metadata.get('context_explore_eps')}, "
            f"projection_rejection_mass={metadata.get('projection_rejection_mass')}, "
            f"feasible_mass_ratio={metadata.get('feasible_mass_ratio')}, "
            f"physical_violation_reduction={metadata.get('physical_violation_reduction')}, "
            f"adaptive_projection_strength={metadata.get('adaptive_projection_strength')}, "
            f"adaptive_projection_floor={metadata.get('adaptive_projection_floor')}, "
            f"candidate_hotspot_density={metadata.get('candidate_hotspot_density')}"
        )

if len(methods_to_eval) > 1:
    baseline_method = methods_to_eval[0]
    logger.info(f'===== Comparison vs {baseline_method} =====')
    baseline_metrics = results_by_method[baseline_method]
    for method_name in methods_to_eval[1:]:
        logger.info(f'----- {method_name} -----')
        for metric_name in metric_order:
            delta = results_by_method[method_name][metric_name] - baseline_metrics[metric_name]
            logger.info(f'{metric_name} delta: {delta}')

if args.ablation_output:
    ablation_path = Path(args.ablation_output)
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ablation_path, 'a', encoding='utf-8') as f:
        for method_name in methods_to_eval:
            record = {
                'ablation_type': 'resolution_privacy_utility',
                'dataset': args.dataset,
                'method': method_name,
                'epsilon': args.epsilon,
                'grid_num': args.grid_num,
                'window': args.w,
                'profile': args.pst_profile if 'retrasyn_pst' in method_name else '',
            }
            record.update(metadata_by_method.get(method_name, {}))
            record.update(results_by_method[method_name])
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f'Resolution-privacy-utility ablation records appended to {ablation_path}')
