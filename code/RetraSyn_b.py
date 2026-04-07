import pickle
#qwe
from ldp import OUE
from grid import Grid, GridMap, Transition
from typing import List, Tuple
import utils
import numpy as np
import math
import json
from parse import args
import multiprocessing
import random
from syndb import SynDB, Users
from logger.logger import ConfigParser
import lzma
from collections import defaultdict
from pathlib import Path
from time import perf_counter
try:
    import scipy.ndimage as ndimage
except ImportError:
    ndimage = None

config = ConfigParser(name='RetraSyn', save_dir='./')
logger = config.get_logger(config.exper_name)

CORES = multiprocessing.cpu_count() // 2
random.seed(2023)
np.random.seed(2023)

logger.info(args)


class RuntimeProfiler:
    def __init__(self):
        self.timings = defaultdict(float)

    def add(self, key: str, duration: float):
        self.timings[key] += duration

    def log_summary(self, logger_obj):
        if not self.timings:
            return
        logger_obj.info('Runtime breakdown:')
        for key in ['spatial_decomposition', 'pst_update', 'trajectory_synthesis', 'write_file']:
            logger_obj.info(f'  {key}: {self.timings.get(key, 0.0):.3f}s')


runtime_profiler = RuntimeProfiler()


def synthetic_method_tag(method_name: str):
    if method_name == 'retrasyn_pst':
        return f'{method_name}_{args.pst_profile}'
    return method_name


def adaptive_spatial_radius(grid_n: int):
    base_grid = max(1, int(args.pst_base_grid_num))
    base_radius = max(1, int(args.pst_base_radius))
    max_radius = max(base_radius, int(args.pst_max_radius))
    scaled_radius = math.ceil(grid_n * base_radius / base_grid)
    return max(base_radius, min(max_radius, int(scaled_radius)))


KINEMATIC_RADIUS = adaptive_spatial_radius(args.grid_num)
KINEMATIC_SIDE = 2 * KINEMATIC_RADIUS + 1
KINEMATIC_TOKEN_COUNT = KINEMATIC_SIDE * KINEMATIC_SIDE


def delta_to_token(delta_i: int, delta_j: int):
    delta_i = max(-KINEMATIC_RADIUS, min(KINEMATIC_RADIUS, int(delta_i)))
    delta_j = max(-KINEMATIC_RADIUS, min(KINEMATIC_RADIUS, int(delta_j)))
    return (delta_i + KINEMATIC_RADIUS) * KINEMATIC_SIDE + (delta_j + KINEMATIC_RADIUS)


def token_to_delta(token: int):
    return token // KINEMATIC_SIDE - KINEMATIC_RADIUS, token % KINEMATIC_SIDE - KINEMATIC_RADIUS


def get_kinematic_token(g1: Grid, g2: Grid):
    delta_i = max(-KINEMATIC_RADIUS, min(KINEMATIC_RADIUS, g2.index[0] - g1.index[0]))
    delta_j = max(-KINEMATIC_RADIUS, min(KINEMATIC_RADIUS, g2.index[1] - g1.index[1]))
    return delta_to_token(delta_i, delta_j)


class CoarseGridMapper:
    def __init__(self, gm: GridMap, coarse_n: int):
        self.grid_map = gm
        self.coarse_n = max(2, min(int(coarse_n), gm.n))
        self.size = self.coarse_n * self.coarse_n
        self.fine_to_coarse = np.empty(gm.size, dtype=np.int32)
        self.coarse_to_fine = [[] for _ in range(self.size)]
        for fine_idx in range(gm.size):
            grid = gm.get_grid_by_linear(fine_idx)
            coarse_i = min(self.coarse_n - 1, int(grid.index[0] * self.coarse_n / gm.n))
            coarse_j = min(self.coarse_n - 1, int(grid.index[1] * self.coarse_n / gm.n))
            coarse_idx = coarse_i * self.coarse_n + coarse_j
            self.fine_to_coarse[fine_idx] = coarse_idx
            self.coarse_to_fine[coarse_idx].append(fine_idx)

    def map_fine_to_coarse(self, fine_idx: int):
        return int(self.fine_to_coarse[int(fine_idx)])

    def aggregate_distribution(self, fine_distribution: np.ndarray):
        coarse = np.bincount(
            self.fine_to_coarse,
            weights=np.asarray(fine_distribution, dtype=float),
            minlength=self.size
        ).astype(float)
        total = coarse.sum()
        if total > 1e-8:
            coarse /= total
        return coarse

    @property
    def resolution_ratio(self):
        return self.coarse_n / max(self.grid_map.n, 1)


def ldp_kde_smooth(raw_freq: np.ndarray, grid_n: int, sigma_override: float = None):
    if sigma_override is not None and sigma_override <= 0.05:
        return raw_freq
    if sigma_override is None and grid_n < 8:
        return raw_freq
    matrix_2d = np.asarray(raw_freq, dtype=float).reshape((grid_n, grid_n))
    sigma = float(sigma_override) if sigma_override is not None else max(0.5, grid_n / 15.0)
    if ndimage is not None:
        smoothed_2d = ndimage.gaussian_filter(matrix_2d, sigma=sigma, mode='reflect')
    else:
        padded = np.pad(matrix_2d, 1, mode='edge')
        smoothed_2d = (
            4.0 * padded[1:-1, 1:-1]
            + 2.0 * (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            + (
                padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:]
            )
        ) / 16.0
    return smoothed_2d.reshape(-1)


class DPLengthDistribution:
    def __init__(self, avg_len: float, max_length: int, prior_strength: float = 8.0):
        self.avg_len = max(avg_len, 1.0)
        self.max_length = max(8, max_length)
        self.prior_strength = prior_strength
        self.bin_ends = self._build_bins()
        self.prior_distribution = self._prior_distribution()
        self.cumulative_adjusted = self.prior_strength * self.prior_distribution.copy()
        self.distribution = self.prior_distribution.copy()

    def _build_bins(self):
        anchors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 28, 36, 48, 64, self.max_length]
        bin_ends = sorted({min(max(anchor, 1), self.max_length) for anchor in anchors})
        if bin_ends[-1] != self.max_length:
            bin_ends.append(self.max_length)
        return np.asarray(bin_ends, dtype=np.int32)

    def _prior_distribution(self):
        lengths = []
        prev_end = 0
        for end in self.bin_ends.tolist():
            lengths.append((prev_end + 1 + end) / 2.0)
            prev_end = end
        lengths = np.asarray(lengths, dtype=float)
        scale = max(self.avg_len, 1.0)
        prior = np.exp(-lengths / scale)
        prior /= max(prior.sum(), 1e-8)
        return prior

    def clip_lengths(self, lengths):
        if len(lengths) == 0:
            return np.empty(0, dtype=np.int32)
        return np.clip(np.asarray(lengths, dtype=np.int32), 1, self.max_length)

    def update(self, quit_lengths, epsilon: float):
        clipped = self.clip_lengths(quit_lengths)
        if clipped.size == 0:
            return

        bin_ids = np.searchsorted(self.bin_ends, clipped, side='left')
        counts = np.bincount(bin_ids, minlength=len(self.bin_ends))
        oue = OUE(epsilon, len(self.bin_ends), lambda x: x)
        oue.aggregate_count_vector(counts)
        oue.adjust()

        freq = oue.non_negative_data / max(oue.n, 1)
        total = freq.sum()
        if total <= 1e-8:
            return

        self.cumulative_adjusted += oue.non_negative_data
        self.distribution = self.cumulative_adjusted / max(self.cumulative_adjusted.sum(), 1e-8)

    def sample_lengths(self, sample_size: int):
        if sample_size <= 0:
            return np.empty(0, dtype=np.int32)
        sampled_bins = np.random.choice(len(self.bin_ends), size=sample_size, p=self.distribution)
        sampled_lengths = np.empty(sample_size, dtype=np.int32)
        for idx, bin_id in enumerate(sampled_bins.tolist()):
            high = int(self.bin_ends[bin_id])
            low = 1 if bin_id == 0 else int(self.bin_ends[bin_id - 1]) + 1
            sampled_lengths[idx] = np.random.randint(low, high + 1)
        return sampled_lengths


class PSTSyntheticState:
    def __init__(self):
        self.t = -1
        self.tail_prev = []
        self.tail_grid_idx = []
        self.tail_time = []
        self.finished_tail_ids = []

    def advance_time(self):
        self.t += 1
        return self.t

    def append_points(self, prev_tail_ids, grid_indices, timestamp: int):
        new_tail_ids = np.empty(len(grid_indices), dtype=np.int64)
        for idx, (prev_tail_id, grid_idx) in enumerate(zip(prev_tail_ids, grid_indices)):
            self.tail_prev.append(int(prev_tail_id))
            self.tail_grid_idx.append(int(grid_idx))
            self.tail_time.append(timestamp)
            new_tail_ids[idx] = len(self.tail_grid_idx) - 1
        return new_tail_ids

    def start_points(self, grid_indices, timestamp: int):
        if len(grid_indices) == 0:
            return np.empty(0, dtype=np.int64)
        prev_tail_ids = np.full(len(grid_indices), -1, dtype=np.int64)
        return self.append_points(prev_tail_ids, grid_indices, timestamp)

    def finish(self, tail_ids):
        for tail_id in tail_ids:
            if int(tail_id) >= 0:
                self.finished_tail_ids.append(int(tail_id))

    def finish_trimmed(self, tail_ids):
        for tail_id in tail_ids:
            prev_tail_id = self.tail_prev[int(tail_id)]
            if prev_tail_id >= 0:
                self.finished_tail_ids.append(int(prev_tail_id))

    def _materialize_trajectory(self, tail_id: int, grid_lookup):
        traj = []
        current_tail = int(tail_id)
        while current_tail >= 0:
            traj.append((grid_lookup[self.tail_grid_idx[current_tail]], self.tail_time[current_tail]))
            current_tail = self.tail_prev[current_tail]
        traj.reverse()
        return traj

    def build_syndb(self, active_tail_ids, grid_lookup):
        syn_db = SynDB()
        syn_db.t = self.t
        all_tail_ids = self.finished_tail_ids + [int(tail_id) for tail_id in active_tail_ids]
        syn_db.history_data = [self._materialize_trajectory(tail_id, grid_lookup) for tail_id in all_tail_ids]
        syn_db.current_data = []
        return syn_db


class TimestepMarginalCalibrator:
    def __init__(self, gm: GridMap, grid_scale: float = 1.0, prior_mix: float = 0.25, ema: float = 0.35):
        self.grid_map = gm
        self.grid_size = gm.size
        self.grid_scale = max(1.0, float(grid_scale))
        self.prior_mix = prior_mix
        self.ema = ema
        self.current_distribution = np.full(self.grid_size, 1.0 / self.grid_size, dtype=float)
        self.hotspot_distribution = self.current_distribution.copy()
        self.neighbor_cache = [gm.get_candidate_linear(gm.get_grid_by_linear(i)) for i in range(self.grid_size)]

        coarse_n = max(3, int(round(gm.n / 2))) if gm.n > 6 else gm.n
        self.block_count = coarse_n * coarse_n
        self.block_ids = np.empty(self.grid_size, dtype=np.int32)
        for idx in range(self.grid_size):
            grid = gm.get_grid_by_linear(idx)
            block_i = min(coarse_n - 1, int(grid.index[0] * coarse_n / gm.n))
            block_j = min(coarse_n - 1, int(grid.index[1] * coarse_n / gm.n))
            self.block_ids[idx] = block_i * coarse_n + block_j
        self.block_sizes = np.bincount(self.block_ids, minlength=self.block_count).astype(float)

    def update(self, counts: np.ndarray, epsilon: float):
        total_count = int(counts.sum())
        if total_count <= 0:
            return

        oue = OUE(epsilon, self.grid_size, lambda x: x)
        oue.aggregate_count_vector(counts)
        oue.adjust()

        raw_freq = oue.adjusted_data.astype(float)
        transition_sigma = 0.2 if self.grid_map.n >= 10 else 0.0
        transition_freq = ldp_kde_smooth(raw_freq, self.grid_map.n, sigma_override=transition_sigma)
        transition_freq = np.maximum(transition_freq, 0.0) / max(oue.n, 1)
        total = transition_freq.sum()
        if total <= 1e-8:
            return

        current = transition_freq / total
        self.current_distribution = (
            self.ema * self.current_distribution
            + (1.0 - self.ema) * current
        )
        self.current_distribution /= max(self.current_distribution.sum(), 1e-8)
        hotspot_freq = ldp_kde_smooth(raw_freq, self.grid_map.n)
        hotspot_freq = np.maximum(hotspot_freq, 0.0) / max(oue.n, 1)
        hotspot_total = hotspot_freq.sum()
        if hotspot_total > 1e-8:
            self.hotspot_distribution = self._build_hotspot_distribution(hotspot_freq / hotspot_total)
        else:
            self.hotspot_distribution = self._build_hotspot_distribution()

    def _build_hotspot_distribution(self, base: np.ndarray = None):
        if base is None:
            base = self.current_distribution
        local_smooth = np.empty(self.grid_size, dtype=float)
        for idx, neighbors in enumerate(self.neighbor_cache):
            local_smooth[idx] = float(base[neighbors].sum())
        local_smooth /= max(local_smooth.sum(), 1e-8)

        if self.block_count > 0:
            coarse_mass = np.bincount(self.block_ids, weights=base, minlength=self.block_count).astype(float)
            coarse_backproj = coarse_mass[self.block_ids] / np.maximum(self.block_sizes[self.block_ids], 1.0)
            coarse_backproj /= max(coarse_backproj.sum(), 1e-8)
        else:
            coarse_backproj = base

        if self.grid_scale <= 1.0:
            return base.copy()

        local_weight = min(0.24, 0.10 + 0.08 * (self.grid_scale - 1.0))
        coarse_weight = min(0.30, 0.12 + 0.10 * (self.grid_scale - 1.0))
        base_weight = max(0.30, 1.0 - local_weight - coarse_weight)
        hotspot = base_weight * base + local_weight * local_smooth + coarse_weight * coarse_backproj
        hotspot /= max(hotspot.sum(), 1e-8)
        return hotspot

    def candidate_prior(self, candidates: np.ndarray, hotspot_bias: float = 0.0):
        hotspot_bias = float(np.clip(hotspot_bias, 0.0, 1.0))
        temporal_prior = self.current_distribution[candidates]
        temporal_total = temporal_prior.sum()
        if temporal_total <= 1e-8:
            temporal_prior = np.full(len(candidates), 1.0 / max(len(candidates), 1), dtype=float)
        else:
            temporal_prior = temporal_prior / temporal_total

        if hotspot_bias <= 1e-8:
            return temporal_prior

        hotspot_prior = self.hotspot_distribution[candidates]
        hotspot_total = hotspot_prior.sum()
        if hotspot_total <= 1e-8:
            return temporal_prior
        hotspot_prior = hotspot_prior / hotspot_total
        mixed_prior = (1.0 - hotspot_bias) * temporal_prior + hotspot_bias * hotspot_prior
        return mixed_prior / max(mixed_prior.sum(), 1e-8)

    def blend(self, candidates: np.ndarray, probs: np.ndarray, hotspot_bias: float = 0.0):
        if self.prior_mix <= 1e-8:
            return probs

        candidate_prior = self.candidate_prior(candidates, hotspot_bias=hotspot_bias)
        mixed = (1.0 - self.prior_mix) * probs + self.prior_mix * candidate_prior
        mixed_total = mixed.sum()
        if mixed_total <= 1e-8:
            return probs
        return mixed / mixed_total


class AbsoluteLocalNode:
    def __init__(self, grid_idx: int, candidates: np.ndarray, motion_tokens: np.ndarray):
        self.grid_idx = grid_idx
        self.candidates = np.asarray(candidates, dtype=np.int32)
        self.motion_tokens = np.asarray(motion_tokens, dtype=np.int32)
        self.candidate_to_pos = {int(candidate): idx for idx, candidate in enumerate(self.candidates.tolist())}
        self.adjusted_mass = np.ones(len(self.candidates), dtype=float)
        self.probs = np.full(len(self.candidates), 1.0 / max(len(self.candidates), 1), dtype=float)
        self.report_count = 0


class AbsoluteTransitionModel:
    def __init__(self, gm: GridMap, grid_scale: float = 1.0):
        self.grid_map = gm
        self.nodes = {}
        self.grid_scale = max(1.0, float(grid_scale))
        self.candidate_radius = adaptive_spatial_radius(gm.n)
        self.min_dynamic_candidates = 5

    def ensure_node(self, grid_idx: int):
        node = self.nodes.get(grid_idx)
        if node is None:
            grid = self.grid_map.get_grid_by_linear(grid_idx)
            candidates = self.grid_map.get_candidate_linear(grid, radius=self.candidate_radius)
            motion_tokens = np.asarray(
                [get_kinematic_token(grid, self.grid_map.get_grid_by_linear(int(candidate))) for candidate in candidates],
                dtype=np.int32
            )
            node = AbsoluteLocalNode(grid_idx, candidates, motion_tokens)
            self.nodes[grid_idx] = node
        return node

    def update_node_distribution(self, grid_idx: int, counts: np.ndarray, epsilon: float):
        total_count = int(counts.sum())
        if total_count <= 0:
            return
        node = self.ensure_node(grid_idx)
        oue = OUE(epsilon, len(node.candidates), lambda x: x)
        oue.aggregate_count_vector(counts)
        oue.adjust()
        freq = oue.non_negative_data / max(oue.n, 1)
        total = freq.sum()
        if total > 1e-8:
            node.adjusted_mass = 0.80 * node.adjusted_mass + 0.20 * oue.non_negative_data
            node.probs = node.adjusted_mass / max(node.adjusted_mass.sum(), 1e-8)
        node.report_count = oue.n

    def candidate_distribution(self, grid_idx: int, history: Tuple[int, ...], marginal_calibrator: TimestepMarginalCalibrator):
        node = self.ensure_node(grid_idx)
        probs = node.probs.copy()
        hotspot_bias = min(0.65, 0.18 + 0.20 * (self.grid_scale - 1.0))
        candidate_prior = marginal_calibrator.candidate_prior(node.candidates, hotspot_bias=hotspot_bias)
        support_weight = node.report_count / (node.report_count + 10.0 * self.grid_scale)
        probs = support_weight * probs + (1.0 - support_weight) * candidate_prior
        probs = marginal_calibrator.blend(node.candidates, probs, hotspot_bias=hotspot_bias)

        curr_grid = self.grid_map.get_grid_by_linear(grid_idx)
        if len(history) < 2:
            candidate_mask = np.fromiter(
                (
                    self.grid_map.chebyshev_distance(curr_grid, self.grid_map.get_grid_by_linear(int(candidate))) <= 1
                    for candidate in node.candidates.tolist()
                ),
                dtype=bool,
                count=len(node.candidates)
            )
        else:
            momentum = np.sum(np.asarray([token_to_delta(int(token)) for token in history[-2:]], dtype=np.int32), axis=0)
            speed = max(abs(int(momentum[0])), abs(int(momentum[1])))
            if speed <= 1:
                candidate_mask = np.fromiter(
                    (
                        self.grid_map.chebyshev_distance(curr_grid, self.grid_map.get_grid_by_linear(int(candidate))) <= 1
                        for candidate in node.candidates.tolist()
                    ),
                    dtype=bool,
                    count=len(node.candidates)
                )
            else:
                scores = np.empty(len(node.candidates), dtype=np.int32)
                for idx, candidate in enumerate(node.candidates.tolist()):
                    next_grid = self.grid_map.get_grid_by_linear(int(candidate))
                    step_i = next_grid.index[0] - curr_grid.index[0]
                    step_j = next_grid.index[1] - curr_grid.index[1]
                    scores[idx] = step_i * int(momentum[0]) + step_j * int(momentum[1])
                candidate_mask = scores >= 0
                candidate_mask[node.candidates == grid_idx] = True
                if int(np.count_nonzero(candidate_mask)) < self.min_dynamic_candidates:
                    top_idx = np.argsort(scores)[-self.min_dynamic_candidates:]
                    candidate_mask[top_idx] = True

        filtered_candidates = node.candidates[candidate_mask]
        filtered_tokens = node.motion_tokens[candidate_mask]
        filtered_probs = probs[candidate_mask]
        filtered_total = filtered_probs.sum()
        if filtered_total <= 1e-8:
            filtered_probs = np.full(len(filtered_candidates), 1.0 / max(len(filtered_candidates), 1), dtype=float)
        else:
            filtered_probs = filtered_probs / filtered_total
        return filtered_candidates, filtered_tokens, filtered_probs


def spatial_decomposition(xy_l: List[Tuple[float, float, float, float, int]], gm: GridMap):
    grid_list = []
    for (x0, y0, x1, y1, flag) in xy_l:
        if flag == 0:
            g0 = gm.point_to_grid((x0, y0))
            g1 = gm.point_to_grid((x1, y1))
            grid_list.append((g0, g1, flag))
        elif flag == 1:
            g1 = gm.point_to_grid((x1, y1))
            grid_list.append((g1, g1, flag))
        else:
            g0 = gm.point_to_grid((x0, y0))
            grid_list.append((g0, g0, flag))
    return grid_list


def spatial_decomposition_uid(xy_l, gm: GridMap):
    grid_list = []
    for (x0, y0, x1, y1, flag, uid) in xy_l:
        if flag == 0:
            g0 = gm.point_to_grid((x0, y0))
            g1 = gm.point_to_grid((x1, y1))
            grid_list.append((g0, g1, flag, uid))
        elif flag == 1:
            g1 = gm.point_to_grid((x1, y1))
            grid_list.append((g1, g1, flag, uid))
        else:
            g0 = gm.point_to_grid((x0, y0))
            grid_list.append((g0, g0, flag, uid))
    return grid_list


def split_traj(traj_stream: List[List[Tuple[Grid, Grid, int]]], gm: GridMap):
    """
    Deal with non-adjacent transitions;
    If (G1, G2, flag) is not adjacent, split it into (G1, end, 2) at t and (start, G2, 1) at t + 1
    """
    new_stream = []
    while len(new_stream) <= len(traj_stream):
        new_stream.append([])
    for t in range(len(traj_stream)):
        for g1, g2, flag in traj_stream[t]:
            if flag:
                new_stream[t].append((g1, g2, flag))
                continue
            if not g1.equal(g2) and not gm.is_adjacent_grids(g1, g2):
                new_stream[t].append((g1, g1, 2))
                new_stream[t + 1].append((g2, g2, 1))
            else:
                new_stream[t].append((g1, g2, flag))
    return new_stream


def split_traj_uid(traj_stream, gm: GridMap, max_step: int = 1):
    new_stream = []
    while len(new_stream) <= len(traj_stream):
        new_stream.append([])
    for t in range(len(traj_stream)):
        for g1, g2, flag, uid in traj_stream[t]:
            if flag:
                new_stream[t].append((g1, g2, flag, uid))
                continue
            if not g1.equal(g2) and gm.chebyshev_distance(g1, g2) > max_step:
                new_stream[t].append((g1, g1, 2, uid))
                new_stream[t + 1].append((g2, g2, 1, uid))
            else:
                new_stream[t].append((g1, g2, flag, uid))
    return new_stream


def generate_markov_matrix(markov_vec: np.ndarray, trans_domain: List[Transition]):
    n = grid_map.size + 1
    markov_mat = np.zeros((n, n), dtype=float)
    end_distribution = np.zeros(n - 1)
    for k in range(len(markov_vec)):
        if markov_vec[k] <= 0:
            continue

        # find index in matrix
        trans = trans_domain[k]
        if not trans.flag:
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = utils.grid_index_map_func(trans.g2, grid_map)
        elif trans.flag == 1:
            # entering transition, located in last row of the matrix
            i = -1
            j = utils.grid_index_map_func(trans.g2, grid_map)
        else:
            # quitting transition, located in last column of the matrix
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = -1
            end_distribution[i] = markov_vec[k]
        markov_mat[i][j] = markov_vec[k]

    # Normalize probabilities by each ROW
    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    end_distribution = end_distribution / (end_distribution.sum() + 1e-8)
    return markov_mat, end_distribution


def convert_grid_to_raw(grid_db: List[List[Tuple[Grid, int]]]):
    def traj_grid_to_raw(traj: List[Tuple[Grid, int]]):
        xy_traj = []
        for (g, t) in traj:
            x, y = g.sample_point()
            xy_traj.append((x, y, t))
        return xy_traj

    raw_db = [traj_grid_to_raw(traj) for traj in grid_db]

    return raw_db


def oue_variance(epsilon: float, n: int):
    if n <= 0:
        return float('inf')
    exp_eps = math.exp(epsilon)
    denom = n * (exp_eps - 1) ** 2
    if denom <= 0:
        return float('inf')
    return 4 * exp_eps / denom


def normalized_entropy(probs: np.ndarray):
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 1e-8 or probs.size <= 1:
        return 0.0
    probs = probs / total
    probs = probs[probs > 1e-12]
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.clip(entropy / math.log(len(probs)), 0.0, 1.0))


def vector_norm(step):
    step = np.asarray(step, dtype=float)
    return float(np.sqrt(np.sum(step * step)))


class PSTNode:
    def __init__(self, context: Tuple[int, ...], candidates: np.ndarray, token_space: int, quit_token: int = None):
        self.context = context
        self.candidates = np.asarray(candidates, dtype=np.int32)
        self.token_to_pos = np.full(token_space, -1, dtype=np.int32)
        self.token_to_pos[self.candidates] = np.arange(len(self.candidates), dtype=np.int32)
        self.quit_token = quit_token
        if quit_token is None or quit_token < 0 or quit_token >= token_space:
            self.quit_idx = -1
        else:
            self.quit_idx = int(self.token_to_pos[quit_token])
        self.probs = np.full(len(self.candidates), 1.0 / len(self.candidates), dtype=float)
        self.observed_probs = self.probs.copy()
        self.quit_prob = 0.0 if self.quit_idx < 0 else float(self.probs[self.quit_idx])
        self.freq_estimate = np.zeros(len(self.candidates), dtype=float)
        self.raw_adjusted = np.zeros(len(self.candidates), dtype=float)
        self.report_count = 0
        self.support = 0.0
        self.noise_tau = 1.0
        self.last_update = -1
        self.max_freq = 0.0
        self.reliable = False
        self.negative_mass_ratio = 1.0
        self.backoff_weight = 0.0
        self.confirmed = len(context) == 1
        self.broadcasted = len(context) == 1
        self.broadcast_at = 0 if self.broadcasted else math.inf
        self.gain = 0.0
        self.gain_tau = 1.0
        self.reference_context = context
        self.branch_support_ema = np.zeros(len(self.candidates), dtype=float)
        self.branch_gain_ema = np.zeros(len(self.candidates), dtype=float)
        self.branch_hits = np.zeros(len(self.candidates), dtype=np.int32)
        self.branch_ldp_mass = np.zeros(len(self.candidates), dtype=float)
        self.branch_observed_mass = np.zeros(len(self.candidates), dtype=float)
        self.branch_var_sum = np.zeros(len(self.candidates), dtype=float)
        self.confirm_support_mass = 0.0
        self.confirm_var_sum = 0.0
        self.confirm_gain_ema = 0.0


class CoarseContextModel:
    def __init__(self, mapper: CoarseGridMapper, max_depth=3, split_sigma=1.0, confirm_gain=0.01):
        self.mapper = mapper
        self.max_depth = max(1, int(max_depth))
        self.split_sigma = split_sigma
        self.confirm_gain = confirm_gain
        self.token_space = mapper.size
        self.nodes = {}
        self.global_mass = np.ones(self.token_space, dtype=float)
        self.global_probs = self.global_mass / self.global_mass.sum()
        self.context_cache = {}

        root = self.ensure_node(())
        root.confirmed = True
        root.broadcasted = True
        root.broadcast_at = 0
        for coarse_idx in range(self.token_space):
            self.ensure_node((coarse_idx,))

    def invalidate_cache(self):
        self.context_cache.clear()

    def ensure_node(self, context: Tuple[int, ...]):
        normalized_context = tuple(context[-self.max_depth:])
        node = self.nodes.get(normalized_context)
        if node is None:
            candidates = np.arange(self.token_space, dtype=np.int32)
            node = PSTNode(normalized_context, candidates, self.token_space, quit_token=None)
            node.confirmed = len(normalized_context) <= 1
            node.broadcasted = len(normalized_context) <= 1
            node.broadcast_at = 0 if node.broadcasted else math.inf
            self.nodes[normalized_context] = node
        return node

    def longest_confirmed_match(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed), 0, -1):
            node = self.nodes.get(trimmed[-order:])
            if node is not None and node.confirmed:
                return node
        return self.ensure_node((trimmed[-1],))

    def longest_broadcast_match(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed), 0, -1):
            node = self.nodes.get(trimmed[-order:])
            if node is not None and node.broadcasted:
                return node
        return self.ensure_node((trimmed[-1],))

    def get_reference_node(self, context: Tuple[int, ...]):
        trimmed = tuple(context[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed) - 1, 0, -1):
            candidate = self.nodes.get(trimmed[-order:])
            if candidate is not None and candidate.confirmed:
                return candidate
        return self.ensure_node((trimmed[-1],))

    def activate_broadcasts(self, timestamp: int):
        for node in self.nodes.values():
            if not node.broadcasted and node.broadcast_at <= timestamp:
                node.broadcasted = True

    def update_global_distribution(self, fine_distribution: np.ndarray):
        coarse_probs = self.mapper.aggregate_distribution(fine_distribution)
        if coarse_probs.sum() > 1e-8:
            self.global_probs = 0.70 * self.global_probs + 0.30 * coarse_probs
            self.global_probs /= max(self.global_probs.sum(), 1e-8)

    def full_probs(self, node: PSTNode):
        full = np.zeros(self.token_space, dtype=float)
        for token, prob in zip(node.candidates.tolist(), node.probs.tolist()):
            full[int(token)] = float(prob)
        total = full.sum()
        if total > 1e-8:
            full /= total
        return full

    def reliability_weight(self, node: PSTNode):
        if node.report_count <= 0 or not node.confirmed:
            return 0.0
        signal_margin = max(0.0, node.max_freq - node.noise_tau)
        signal_scale = signal_margin / (node.max_freq + 1e-8)
        negative_penalty = max(0.0, 1.0 - node.negative_mass_ratio)
        return float(np.clip(signal_scale * negative_penalty, 0.0, 1.0))

    def node_uncertainty(self, node: PSTNode):
        entropy = normalized_entropy(node.probs)
        confidence = self.reliability_weight(node)
        return float(np.clip(0.65 * entropy + 0.35 * (1.0 - confidence), 0.0, 1.0))

    def update_node_distribution(self, node: PSTNode, counts: np.ndarray, epsilon: float, timestamp: int):
        total_count = int(counts.sum())
        if total_count <= 0:
            return
        oue = OUE(epsilon, len(node.candidates), lambda x: x)
        oue.aggregate_count_vector(counts)
        oue.adjust()

        raw_adjusted = oue.adjusted_data / max(oue.n, 1)
        freq = oue.non_negative_data / max(oue.n, 1)
        total = freq.sum()
        if total > 1e-8:
            node.probs = freq / total
        node.observed_probs = counts / total_count
        node.raw_adjusted = raw_adjusted
        node.freq_estimate = freq
        node.report_count = oue.n
        node.support = float(total_count)
        node.last_update = timestamp
        node.max_freq = float(node.probs.max()) if node.probs.size else 0.0
        node.noise_tau = self.split_sigma * math.sqrt(oue_variance(epsilon, oue.n) / max(len(node.candidates), 1))
        positive_mass = float(np.maximum(raw_adjusted, 0.0).sum())
        negative_mass = float(np.maximum(-raw_adjusted, 0.0).sum())
        node.negative_mass_ratio = negative_mass / (positive_mass + 1e-8)
        node.reliable = (
            node.report_count > 0
            and node.max_freq > node.noise_tau
            and node.negative_mass_ratio < 0.6
        )
        node.confirm_support_mass += float(total_count)
        node.confirm_var_sum += oue_variance(epsilon, oue.n) * max(total_count, 1) ** 2
        node.branch_ldp_mass += oue.non_negative_data
        node.branch_var_sum += oue_variance(epsilon, oue.n) * max(total_count, 1) ** 2

        if len(node.context) > 1:
            ref_node = self.get_reference_node(node.context[1:])
            node.reference_context = ref_node.context
            node.gain = float(np.abs(self.full_probs(node) - self.full_probs(ref_node)).sum())
            node.gain_tau = max(self.confirm_gain, 0.35 * (node.noise_tau + ref_node.noise_tau))
            node.confirm_gain_ema = 0.6 * node.confirm_gain_ema + 0.4 * node.gain
            if (
                node.confirm_support_mass > max(6.0, self.split_sigma * math.sqrt(node.confirm_var_sum + 1e-8))
                and node.confirm_gain_ema > node.gain_tau
                and node.negative_mass_ratio < 0.6
            ):
                node.confirmed = True
                node.broadcasted = True
        else:
            node.reference_context = node.context
            node.gain = 0.0
            node.gain_tau = 0.0
            node.confirmed = True
            node.broadcasted = True
        node.backoff_weight = self.reliability_weight(node)

    def maybe_split_node(self, node: PSTNode, timestamp: int):
        if len(node.context) >= self.max_depth or node.report_count <= 0 or not node.confirmed:
            return 0

        suffix_node = self.get_reference_node(node.context[1:]) if len(node.context) > 1 else None
        suffix_probs = self.global_probs if suffix_node is None else self.full_probs(suffix_node)
        created = 0
        candidate_scores = []
        for coarse_idx, freq in zip(node.candidates.tolist(), node.freq_estimate.tolist()):
            coarse_pos = int(node.token_to_pos[int(coarse_idx)])
            support_prob = float(node.probs[coarse_pos])
            gain = abs(support_prob - float(suffix_probs[int(coarse_idx)]))
            gain_tau = max(self.confirm_gain, node.noise_tau)
            node.branch_support_ema[coarse_pos] = 0.6 * node.branch_support_ema[coarse_pos] + 0.4 * support_prob
            node.branch_gain_ema[coarse_pos] = 0.6 * node.branch_gain_ema[coarse_pos] + 0.4 * gain
            support_tau = self.split_sigma * math.sqrt(node.branch_var_sum[coarse_pos] + 1e-8)
            support_significant = node.branch_ldp_mass[coarse_pos] > max(4.0, support_tau)
            gain_significant = node.branch_gain_ema[coarse_pos] > gain_tau
            if support_significant and gain_significant:
                candidate_scores.append((gain, freq, int(coarse_idx)))

        candidate_scores.sort(reverse=True)
        for _, _, coarse_idx in candidate_scores[:2]:
            context = tuple((node.context + (coarse_idx,))[-self.max_depth:])
            if context not in self.nodes:
                child = self.ensure_node(context)
                child.reference_context = self.get_reference_node(context[1:]).context
                child.broadcast_at = timestamp + 1
                created += 1
            elif not self.nodes[context].broadcasted:
                self.nodes[context].broadcast_at = min(self.nodes[context].broadcast_at, timestamp + 1)
        return created

    def backoff_distribution(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        cached = self.context_cache.get(trimmed)
        if cached is not None:
            return cached

        mixed = self.global_probs.copy()
        if trimmed:
            base_node = self.ensure_node((trimmed[-1],))
            if base_node.report_count > 0:
                mixed = base_node.probs.copy()
        for order in range(2, len(trimmed) + 1):
            node = self.nodes.get(trimmed[-order:])
            if node is None or not node.confirmed or not node.reliable:
                continue
            weight = node.backoff_weight
            mixed = weight * node.probs + (1.0 - weight) * mixed
        total = mixed.sum()
        if total > 1e-8:
            mixed = mixed / total
        else:
            mixed = self.global_probs.copy()
        self.context_cache[trimmed] = mixed
        return mixed

class DynamicPST:
    def __init__(
        self,
        gm: GridMap,
        coarse_mapper: CoarseGridMapper,
        max_depth=4,
        split_top_k=2,
        split_sigma=1.5,
        confirm_gain=0.03,
        marginal_prior_mix=0.25,
        marginal_ema=0.35,
        momentum_min_candidates=4,
        consistency_mix=0.5,
        grid_scale=1.0
    ):
        self.grid_map = gm
        self.coarse_mapper = coarse_mapper
        self.max_depth = max_depth
        self.split_top_k = split_top_k
        self.split_sigma = split_sigma
        self.confirm_gain = confirm_gain
        self.quit_token = KINEMATIC_TOKEN_COUNT
        self.momentum_min_candidates = max(2, momentum_min_candidates)
        self.consistency_mix = consistency_mix
        self.grid_scale = max(1.0, float(grid_scale))
        self.absolute_power = min(1.9, 1.0 + 0.55 * (self.grid_scale - 1.0))
        self.motion_power = max(0.45, 1.0 - 0.22 * (self.grid_scale - 1.0))
        self.motion_stability = min(0.70, 0.12 + 0.22 * (self.grid_scale - 1.0))
        self.coarse_power = float(np.clip(args.pst_coarse_mix * (1.15 if self.coarse_mapper.coarse_n < gm.n else 0.85), 0.10, 1.25))
        self.projection_strength = max(0.0, float(args.pst_projection_strength))
        self.projection_feasible_floor = float(np.clip(args.pst_projection_feasible_floor, 0.0, 1.0))
        self.speed_weight = max(0.0, float(args.pst_speed_weight))
        self.turn_weight = max(0.0, float(args.pst_turn_weight))
        self.accel_weight = max(0.0, float(args.pst_accel_weight))
        self.projection_density_gamma = max(0.1, float(args.pst_projection_density_gamma))
        self.nodes = {}
        self.start_distribution = np.full(gm.size, 1.0 / gm.size, dtype=float)
        self.start_noise_tau = 1.0
        self.token_space = KINEMATIC_TOKEN_COUNT + 1
        self.grid_lookup = [gm.get_grid_by_linear(i) for i in range(gm.size)]
        self.synthesis_lut = {}
        self.joint_synthesis_lut = {}
        self.projection_totals = defaultdict(float)
        self.projection_interval = defaultdict(float)
        self.absolute_model = AbsoluteTransitionModel(gm, grid_scale=self.grid_scale)
        self.coarse_model = CoarseContextModel(
            coarse_mapper,
            max_depth=args.pst_context_depth,
            split_sigma=split_sigma,
            confirm_gain=max(0.5 * confirm_gain, 1e-3)
        )
        self.marginal_calibrator = TimestepMarginalCalibrator(
            gm,
            grid_scale=self.grid_scale,
            prior_mix=marginal_prior_mix,
            ema=marginal_ema
        )
        self.next_grid_lut = np.zeros((gm.size, KINEMATIC_TOKEN_COUNT), dtype=np.int32)
        self.next_motion_lut = np.zeros((gm.size, KINEMATIC_TOKEN_COUNT), dtype=np.int32)
        stay_token = delta_to_token(0, 0)
        for grid_idx, grid in enumerate(self.grid_lookup):
            for token in range(KINEMATIC_TOKEN_COUNT):
                delta_i, delta_j = token_to_delta(token)
                next_i = grid.index[0] + delta_i
                next_j = grid.index[1] + delta_j
                if 0 <= next_i < gm.n and 0 <= next_j < gm.n:
                    self.next_grid_lut[grid_idx, token] = gm.map[next_i][next_j].linear_index
                    self.next_motion_lut[grid_idx, token] = token
                else:
                    self.next_grid_lut[grid_idx, token] = grid_idx
                    self.next_motion_lut[grid_idx, token] = stay_token
        root = self.ensure_node(())
        root.confirmed = True
        root.broadcasted = True
        root.broadcast_at = 0
        for token in range(KINEMATIC_TOKEN_COUNT):
            self.ensure_node((token,))

    def invalidate_synthesis_lut(self):
        self.synthesis_lut.clear()
        self.joint_synthesis_lut.clear()
        self.coarse_model.invalidate_cache()

    def record_projection_stats(self, stats, weight: int):
        weight = max(int(weight), 0)
        if weight <= 0 or not stats:
            return
        for key, value in stats.items():
            self.projection_totals[key] += float(value) * weight
            self.projection_interval[key] += float(value) * weight
        self.projection_totals['samples'] += weight
        self.projection_interval['samples'] += weight

    def projection_summary(self, reset: bool = False):
        source = self.projection_interval if reset else self.projection_totals
        sample_count = max(source.get('samples', 0.0), 1e-8)
        summary = {
            'projection_rejection_mass': source.get('rejection_mass', 0.0) / sample_count,
            'feasible_mass_ratio': source.get('feasible_mass_ratio', 0.0) / sample_count,
            'physical_violation_reduction': source.get('violation_reduction', 0.0) / sample_count,
            'raw_physical_violation': source.get('raw_violation', 0.0) / sample_count,
            'projected_physical_violation': source.get('projected_violation', 0.0) / sample_count,
            'adaptive_projection_strength': source.get('adaptive_strength', 0.0) / sample_count,
            'adaptive_projection_floor': source.get('adaptive_floor', 0.0) / sample_count,
            'candidate_hotspot_density': source.get('hotspot_density', 0.0) / sample_count,
            'samples': source.get('samples', 0.0),
        }
        if reset:
            self.projection_interval = defaultdict(float)
        return summary

    def momentum_vector(self, context: Tuple[int, ...]):
        if len(context) < 2:
            return None
        deltas = np.asarray([token_to_delta(int(token)) for token in context[-2:]], dtype=np.int32)
        momentum = np.sum(deltas, axis=0)
        if not np.any(momentum):
            return None
        return momentum

    def candidate_space(self, context: Tuple[int, ...]):
        candidates = np.arange(KINEMATIC_TOKEN_COUNT, dtype=np.int32)
        local_mask = np.fromiter(
            (
                max(abs(token_to_delta(int(candidate))[0]), abs(token_to_delta(int(candidate))[1])) <= 1
                for candidate in candidates.tolist()
            ),
            dtype=bool,
            count=len(candidates)
        )
        if len(context) < 2:
            return candidates[local_mask]
        momentum = self.momentum_vector(context)
        if momentum is None:
            return candidates[local_mask]

        speed = max(abs(int(momentum[0])), abs(int(momentum[1])))
        if speed <= 1:
            return candidates[local_mask]

        scores = np.empty(len(candidates), dtype=np.int32)
        for idx, candidate in enumerate(candidates.tolist()):
            step_i, step_j = token_to_delta(int(candidate))
            scores[idx] = step_i * int(momentum[0]) + step_j * int(momentum[1])

        keep_mask = scores >= 0
        keep_mask[candidates == delta_to_token(0, 0)] = True
        if int(np.count_nonzero(keep_mask)) < self.momentum_min_candidates:
            top_idx = np.argsort(scores)[-self.momentum_min_candidates:]
            keep_mask[top_idx] = True
        return candidates[keep_mask]

    def ensure_node(self, context: Tuple[int, ...]):
        normalized_context = tuple(context[-self.max_depth:])
        node = self.nodes.get(normalized_context)
        if node is None:
            node = PSTNode(
                normalized_context,
                self.candidate_space(normalized_context),
                self.token_space,
                quit_token=self.quit_token
            )
            self.nodes[normalized_context] = node
        return node

    def longest_match(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed), 0, -1):
            node = self.nodes.get(trimmed[-order:])
            if node is not None:
                return node
        return self.ensure_node((trimmed[-1],))

    def longest_confirmed_match(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed), 0, -1):
            node = self.nodes.get(trimmed[-order:])
            if node is not None and node.confirmed:
                return node
        return self.ensure_node((trimmed[-1],))

    def longest_broadcast_match(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed), 0, -1):
            node = self.nodes.get(trimmed[-order:])
            if node is not None and node.broadcasted:
                return node
        return self.ensure_node((trimmed[-1],))

    def activate_broadcasts(self, timestamp: int):
        for node in self.nodes.values():
            if not node.broadcasted and node.broadcast_at <= timestamp:
                node.broadcasted = True

    def get_reference_node(self, context: Tuple[int, ...]):
        trimmed = tuple(context[-self.max_depth:])
        if not trimmed:
            return self.ensure_node(())
        for order in range(len(trimmed) - 1, 0, -1):
            candidate = self.nodes.get(trimmed[-order:])
            if candidate is not None and candidate.confirmed:
                return candidate
        return self.ensure_node((trimmed[-1],))

    def update_start_distribution(self, counts: np.ndarray, epsilon: float):
        total_count = int(counts.sum())
        if total_count <= 0:
            return

        oue = OUE(epsilon, self.grid_map.size, lambda x: x)
        oue.aggregate_count_vector(counts)
        oue.adjust()

        raw_freq = oue.adjusted_data.astype(float)
        raw_freq = ldp_kde_smooth(raw_freq, self.grid_map.n)
        freq = np.maximum(raw_freq, 0.0) / max(oue.n, 1)
        total = freq.sum()
        if total > 0:
            self.start_distribution = freq / total
        self.start_noise_tau = self.split_sigma * math.sqrt(oue_variance(epsilon, oue.n))

    def update_timestep_marginal(self, counts: np.ndarray, epsilon: float):
        self.marginal_calibrator.update(counts, epsilon)
        self.coarse_model.update_global_distribution(self.marginal_calibrator.current_distribution)

    def apply_hierarchical_consistency(self):
        if self.consistency_mix <= 1e-8:
            return

        child_supports = defaultdict(dict)
        sorted_contexts = sorted(self.nodes.keys(), key=len, reverse=True)

        for context in sorted_contexts:
            if len(context) <= 1:
                continue
            node = self.nodes[context]
            parent = self.nodes.get(context[:-1])
            if parent is None or parent.support <= 1e-8:
                continue
            token = context[-1]
            token_pos = int(parent.token_to_pos[token])
            if token_pos < 0:
                continue

            parent_mass = max(parent.support * parent.probs[token_pos], 0.0)
            child_mass = max(node.support, 0.0)
            parent_var = max(parent.noise_tau ** 2, 1e-8)
            child_var = max(node.noise_tau ** 2, 1e-8)
            inv_parent = 1.0 / parent_var
            inv_child = 1.0 / child_var
            consistent_mass = (parent_mass * inv_parent + child_mass * inv_child) / (inv_parent + inv_child)
            consistent_mass = max(0.0, consistent_mass)
            node.support = (
                (1.0 - self.consistency_mix) * child_mass
                + self.consistency_mix * consistent_mass
            )
            node.confirm_support_mass = max(node.confirm_support_mass, node.support)
            child_supports[parent.context][token] = node.support

        for parent_context, token_supports in child_supports.items():
            parent = self.nodes[parent_context]
            if parent.support <= 1e-8:
                continue
            adjusted = parent.probs.copy()
            child_positions = []
            child_probs = []
            child_total_prob = 0.0
            for token, support in token_supports.items():
                token_pos = int(parent.token_to_pos[token])
                if token_pos < 0:
                    continue
                prob = float(np.clip(support / max(parent.support, 1e-8), 0.0, 1.0))
                child_positions.append(token_pos)
                child_probs.append(prob)
                child_total_prob += prob
            if not child_positions:
                continue

            child_total_prob = min(child_total_prob, 0.999)
            for pos, prob in zip(child_positions, child_probs):
                adjusted[pos] = (1.0 - self.consistency_mix) * adjusted[pos] + self.consistency_mix * prob

            remaining_positions = [idx for idx in range(len(adjusted)) if idx not in child_positions]
            remaining_mass = max(1.0 - sum(adjusted[pos] for pos in child_positions), 1e-8)
            if remaining_positions:
                existing_mass = adjusted[remaining_positions].sum()
                if existing_mass <= 1e-8:
                    adjusted[remaining_positions] = remaining_mass / len(remaining_positions)
                else:
                    adjusted[remaining_positions] *= remaining_mass / existing_mass

            adjusted_total = adjusted.sum()
            if adjusted_total > 1e-8:
                parent.probs = adjusted / adjusted_total
                parent.max_freq = float(parent.probs.max())

    def reliability_weight(self, node: PSTNode):
        if node.report_count <= 0 or not node.confirmed:
            return 0.0

        signal_margin = max(0.0, node.max_freq - node.noise_tau)
        signal_scale = signal_margin / (node.max_freq + 1e-8)
        negative_penalty = max(0.0, 1.0 - node.negative_mass_ratio)
        return float(np.clip(signal_scale * negative_penalty, 0.0, 1.0))

    def node_uncertainty(self, node: PSTNode):
        entropy = normalized_entropy(node.probs)
        confidence = self.reliability_weight(node)
        uncertainty = 0.65 * entropy + 0.35 * (1.0 - confidence)
        return float(np.clip(uncertainty, 0.0, 1.0))

    def normalized_move_probs(self, node: PSTNode):
        probs = np.zeros(KINEMATIC_TOKEN_COUNT, dtype=float)
        for token, prob in zip(node.candidates.tolist(), node.probs.tolist()):
            token = int(token)
            if 0 <= token < KINEMATIC_TOKEN_COUNT:
                probs[token] = float(prob)
        total = probs.sum()
        if total <= 1e-8:
            return None
        return probs / total

    def full_token_probs(self, node: PSTNode, observed: bool = False):
        values = node.observed_probs if observed else node.probs
        full = np.zeros(KINEMATIC_TOKEN_COUNT, dtype=float)
        for token, prob in zip(node.candidates.tolist(), values.tolist()):
            if 0 <= int(token) < KINEMATIC_TOKEN_COUNT:
                full[int(token)] = float(prob)
        total = full.sum()
        if total > 1e-8:
            full /= total
        return full

    def update_node_distribution(self, node: PSTNode, counts: np.ndarray, epsilon: float, timestamp: int):
        total_count = int(counts.sum())
        if total_count <= 0:
            return
        observed_probs = counts / total_count
        node.observed_probs = observed_probs
        oue = OUE(epsilon, len(node.candidates), lambda x: x)
        oue.aggregate_count_vector(counts)
        oue.adjust()

        raw_adjusted = oue.adjusted_data / max(oue.n, 1)
        freq = oue.non_negative_data / max(oue.n, 1)
        total = freq.sum()
        adjusted_counts = oue.non_negative_data.copy()
        if total > 0:
            node.probs = freq / total
            node.quit_prob = 0.0 if node.quit_idx < 0 else float(node.probs[node.quit_idx])
        node.raw_adjusted = raw_adjusted
        node.freq_estimate = freq
        node.report_count = oue.n
        node.support = float(total_count)
        node_var = oue_variance(epsilon, oue.n)
        candidate_var = node_var / max(len(node.candidates), 1)
        node.noise_tau = self.split_sigma * math.sqrt(candidate_var)
        node.branch_ldp_mass += adjusted_counts
        node.branch_observed_mass += counts
        node.branch_var_sum += node_var * max(total_count, 1) ** 2
        node.confirm_support_mass += float(total_count)
        node.confirm_var_sum += node_var * max(total_count, 1) ** 2
        node.last_update = timestamp
        node.max_freq = float(node.probs.max()) if node.probs.size else 0.0
        positive_mass = float(np.maximum(raw_adjusted, 0.0).sum())
        negative_mass = float(np.maximum(-raw_adjusted, 0.0).sum())
        node.negative_mass_ratio = negative_mass / (positive_mass + 1e-8)
        node.reliable = (
            node.report_count > 0
            and node.max_freq > node.noise_tau
            and node.negative_mass_ratio < 0.6
        )
        if len(node.context) > 1:
            ref_node = self.get_reference_node(node.context[1:])
            node.reference_context = ref_node.context
            node.gain = float(np.abs(self.full_token_probs(node) - self.full_token_probs(ref_node)).sum())
            node.gain_tau = max(
                self.confirm_gain,
                0.35 * (node.noise_tau + ref_node.noise_tau)
            )
            node.confirm_gain_ema = 0.6 * node.confirm_gain_ema + 0.4 * node.gain
            if (
                node.confirm_support_mass > max(8.0, self.split_sigma * math.sqrt(node.confirm_var_sum + 1e-8))
                and node.confirm_gain_ema > node.gain_tau
                and node.negative_mass_ratio < 0.6
            ):
                node.confirmed = True
                node.broadcasted = True
        else:
            node.reference_context = node.context
            node.gain = 0.0
            node.gain_tau = 0.0
            node.confirmed = True
        node.backoff_weight = self.reliability_weight(node)

    def maybe_split_node(self, node: PSTNode, timestamp: int):
        if len(node.context) >= self.max_depth or node.report_count <= 0 or not node.confirmed:
            return 0

        suffix_node = self.get_reference_node(node.context[1:]) if len(node.context) > 1 else None
        suffix_probs = None if suffix_node is None else self.full_token_probs(suffix_node)
        significant_children = []
        for token, freq in zip(node.candidates, node.freq_estimate):
            token = int(token)
            if token == self.quit_token:
                continue
            token_pos = int(node.token_to_pos[token])
            support_prob = float(node.probs[token_pos])
            gain = support_prob if suffix_node is None else abs(support_prob - suffix_probs[token])
            gain_tau = max(self.confirm_gain, node.noise_tau)
            node.branch_support_ema[token_pos] = 0.6 * node.branch_support_ema[token_pos] + 0.4 * support_prob
            node.branch_gain_ema[token_pos] = 0.6 * node.branch_gain_ema[token_pos] + 0.4 * gain
            support_tau = self.split_sigma * math.sqrt(node.branch_var_sum[token_pos] + 1e-8)
            support_significant = node.branch_ldp_mass[token_pos] > max(6.0, support_tau)
            gain_significant = node.branch_gain_ema[token_pos] > gain_tau
            if support_significant and gain_significant:
                node.branch_hits[token_pos] += 1
            if (
                support_significant
                and gain_significant
                and node.branch_hits[token_pos] >= 1
            ):
                significant_children.append((token, freq, support_prob, gain))

        if not significant_children:
            return 0

        significant_children.sort(key=lambda item: (item[3], item[1]), reverse=True)

        created = 0
        for token, _, _, _ in significant_children[:self.split_top_k]:
            context = tuple((node.context + (token,))[-self.max_depth:])
            if context not in self.nodes:
                child_node = self.ensure_node(context)
                child_node.reference_context = self.get_reference_node(context[1:]).context
                child_node.broadcast_at = timestamp + 1
                created += 1
            elif not self.nodes[context].broadcasted:
                self.nodes[context].broadcast_at = min(self.nodes[context].broadcast_at, timestamp + 1)
        return created

    def node_is_reliable(self, node: PSTNode):
        return node.confirmed and node.reliable

    def sample_start_grid(self, sample_size: int):
        probs = self.start_distribution.copy()
        if self.grid_scale > 1.0:
            start_hotspot_mix = min(0.55, 0.12 + 0.16 * (self.grid_scale - 1.0))
            probs = (
                (1.0 - start_hotspot_mix) * probs
                + start_hotspot_mix * self.marginal_calibrator.hotspot_distribution
            )
        total = probs.sum()
        if total <= 1e-8:
            return np.random.choice(self.grid_map.size, size=sample_size)
        return np.random.choice(self.grid_map.size, size=sample_size, p=probs / total)

    def backoff_distribution(self, history: Tuple[int, ...], traj_len: int, avg_len: float):
        del traj_len, avg_len
        trimmed = tuple(history[-self.max_depth:])
        if not trimmed:
            base_node = self.ensure_node(())
        else:
            base_node = self.ensure_node((trimmed[-1],))
        base_probs = self.normalized_move_probs(base_node)
        if base_probs is None:
            base_probs = np.full(KINEMATIC_TOKEN_COUNT, 1.0 / KINEMATIC_TOKEN_COUNT, dtype=float)

        mixed_probs = base_probs
        for order in range(2, len(trimmed) + 1):
            node = self.nodes.get(trimmed[-order:])
            if node is None or not node.confirmed:
                continue

            node_probs = self.normalized_move_probs(node)
            if node_probs is None:
                continue

            if not self.node_is_reliable(node):
                continue

            weight = node.backoff_weight
            if weight <= 1e-8:
                continue
            mixed_probs = weight * node_probs + (1.0 - weight) * mixed_probs

        mixed_total = mixed_probs.sum()
        if mixed_total <= 1e-8:
            mixed_probs = base_probs
        else:
            mixed_probs = mixed_probs / mixed_total
        return np.arange(KINEMATIC_TOKEN_COUNT, dtype=np.int32), mixed_probs

    def synthesis_entry(self, history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        cached = self.synthesis_lut.get(trimmed)
        if cached is not None:
            return cached

        candidates, probs = self.backoff_distribution(trimmed, 0, 0.0)
        move_mask = candidates != self.quit_token
        move_candidates = candidates[move_mask].astype(np.int32, copy=False)
        move_probs = probs[move_mask]
        total = move_probs.sum()
        if total <= 1e-8:
            base_node = self.ensure_node(()) if not trimmed else self.ensure_node((trimmed[-1],))
            move_candidates = base_node.candidates[base_node.candidates != self.quit_token].astype(np.int32, copy=False)
            move_probs = np.full(len(move_candidates), 1.0 / max(len(move_candidates), 1), dtype=float)
        else:
            move_probs = move_probs / total

        move_cdf = np.cumsum(move_probs)
        if move_cdf.size:
            move_cdf[-1] = 1.0
        cached = (move_candidates, move_cdf)
        self.synthesis_lut[trimmed] = cached
        return cached

    def physical_feasibility_projection(
        self,
        history: Tuple[int, ...],
        candidate_tokens: np.ndarray,
        probs: np.ndarray,
        candidate_hotspot_density: np.ndarray
    ):
        candidate_tokens = np.asarray(candidate_tokens, dtype=np.int32)
        probs = np.asarray(probs, dtype=float)
        candidate_hotspot_density = np.asarray(candidate_hotspot_density, dtype=float)
        total = probs.sum()
        if total <= 1e-8 or candidate_tokens.size == 0 or self.projection_strength <= 1e-8:
            return probs, {
                'rejection_mass': 0.0,
                'feasible_mass_ratio': 1.0,
                'violation_reduction': 0.0,
                'raw_violation': 0.0,
                'projected_violation': 0.0,
                'adaptive_strength': 0.0,
                'adaptive_floor': self.projection_feasible_floor,
                'hotspot_density': 0.0,
            }

        probs = probs / total
        hotspot_max = float(np.max(self.marginal_calibrator.hotspot_distribution))
        if hotspot_max <= 1e-8:
            hotspot_max = float(np.max(self.marginal_calibrator.current_distribution))
        hotspot_max = max(hotspot_max, 1e-8)
        hotspot_norm = np.clip(candidate_hotspot_density / hotspot_max, 0.0, 1.0)
        adaptive_strength = self.projection_strength * np.power(1.0 - hotspot_norm, self.projection_density_gamma)
        adaptive_floor = self.projection_feasible_floor + 0.75 * hotspot_norm * (1.0 - self.projection_feasible_floor)
        adaptive_floor = np.clip(adaptive_floor, self.projection_feasible_floor, 0.98)

        curr_steps = np.asarray([token_to_delta(int(token)) for token in candidate_tokens.tolist()], dtype=float)
        curr_speed = np.sqrt(np.sum(curr_steps * curr_steps, axis=1))

        if history:
            prev_step = np.asarray(token_to_delta(int(history[-1])), dtype=float)
        else:
            prev_step = np.zeros(2, dtype=float)
        if len(history) >= 2:
            prev_prev_step = np.asarray(token_to_delta(int(history[-2])), dtype=float)
        else:
            prev_prev_step = prev_step.copy()

        prev_speed = vector_norm(prev_step)
        prev_accel = prev_step - prev_prev_step
        accel = curr_steps - prev_step
        jerk = accel - prev_accel
        accel_norm = np.sqrt(np.sum(accel * accel, axis=1))
        jerk_norm = np.sqrt(np.sum(jerk * jerk, axis=1))

        denom = np.maximum(curr_speed * max(prev_speed, 1e-8), 1e-8)
        cosine = np.ones_like(curr_speed)
        moving_mask = (curr_speed > 1e-8) & (prev_speed > 1e-8)
        cosine[moving_mask] = np.clip(np.sum(curr_steps[moving_mask] * prev_step, axis=1) / denom[moving_mask], -1.0, 1.0)

        speed_scale = max(float(KINEMATIC_RADIUS), 1.0)
        accel_scale = max(speed_scale, prev_speed + 1.0)
        speed_dev = np.abs(curr_speed - prev_speed) / speed_scale
        turn_penalty = 0.5 * (1.0 - cosine)
        accel_penalty = 0.7 * (accel_norm / accel_scale) + 0.3 * (jerk_norm / max(accel_scale, 1.0))

        energy = (
            self.speed_weight * speed_dev
            + self.turn_weight * turn_penalty
            + self.accel_weight * accel_penalty
        )

        feasible_mask = np.ones(candidate_tokens.size, dtype=bool)
        feasible_mask &= curr_speed <= max(speed_scale, prev_speed + 1.0) + 1e-8
        feasible_mask &= accel_norm <= max(accel_scale, 1.0) + 1e-8
        if prev_speed > 1.25:
            feasible_mask &= cosine >= -0.15
        if prev_speed > 0.0 and vector_norm(prev_accel) > 1.5:
            feasible_mask &= jerk_norm <= max(accel_scale, 1.0) + 1e-8

        soft_gate = np.exp(-adaptive_strength * energy)
        soft_gate = soft_gate * feasible_mask.astype(float) + adaptive_floor * (~feasible_mask).astype(float)

        projected = probs * np.maximum(soft_gate, 1e-8)
        retained_mass = float(projected.sum())
        if retained_mass <= 1e-8:
            projected = probs.copy()
            retained_mass = float(projected.sum())
        projected /= max(retained_mass, 1e-8)

        raw_violation = float(np.dot(probs, energy))
        projected_violation = float(np.dot(projected, energy))
        violation_reduction = 0.0
        if raw_violation > 1e-8:
            violation_reduction = max(0.0, (raw_violation - projected_violation) / raw_violation)

        stats = {
            'rejection_mass': float(np.clip(1.0 - retained_mass, 0.0, 1.0)),
            'feasible_mass_ratio': float(np.clip(probs[feasible_mask].sum(), 0.0, 1.0)),
            'violation_reduction': float(np.clip(violation_reduction, 0.0, 1.0)),
            'raw_violation': raw_violation,
            'projected_violation': projected_violation,
            'adaptive_strength': float(np.dot(probs, adaptive_strength)),
            'adaptive_floor': float(np.dot(probs, adaptive_floor)),
            'hotspot_density': float(np.dot(probs, hotspot_norm)),
        }
        return projected, stats

    def joint_motion_cdf(self, current_grid_idx: int, history: Tuple[int, ...], coarse_history: Tuple[int, ...]):
        trimmed = tuple(history[-self.max_depth:])
        coarse_trimmed = tuple(coarse_history[-self.coarse_model.max_depth:])
        cache_key = (int(current_grid_idx), trimmed, coarse_trimmed)
        cached = self.joint_synthesis_lut.get(cache_key)
        if cached is not None:
            return cached

        motion_candidates, motion_cdf = self.synthesis_entry(trimmed)
        motion_probs = np.zeros(KINEMATIC_TOKEN_COUNT, dtype=float)
        candidate_probs = np.diff(np.concatenate(([0.0], motion_cdf)))
        motion_probs[motion_candidates] = candidate_probs

        base_history = () if not trimmed else (trimmed[-1],)
        base_candidates, base_cdf = self.synthesis_entry(base_history)
        base_motion_probs = np.zeros(KINEMATIC_TOKEN_COUNT, dtype=float)
        base_candidate_probs = np.diff(np.concatenate(([0.0], base_cdf)))
        base_motion_probs[base_candidates] = base_candidate_probs

        matched_node = self.longest_confirmed_match(trimmed)
        context_reliability = 0.0 if matched_node is None else matched_node.backoff_weight
        stability_mix = np.clip(self.motion_stability + 0.45 * (1.0 - context_reliability), 0.0, 0.90)
        motion_probs = (1.0 - stability_mix) * motion_probs + stability_mix * base_motion_probs
        motion_probs /= max(motion_probs.sum(), 1e-8)

        abs_candidates, abs_motion_tokens, abs_candidate_probs = self.absolute_model.candidate_distribution(
            current_grid_idx,
            trimmed,
            self.marginal_calibrator
        )
        coarse_probs = self.coarse_model.backoff_distribution(coarse_trimmed)
        coarse_candidate_ids = self.coarse_mapper.fine_to_coarse[abs_candidates]
        coarse_gate = np.power(np.maximum(coarse_probs[coarse_candidate_ids], 1e-8), self.coarse_power)
        motion_gate = np.power(np.maximum(motion_probs[abs_motion_tokens], 1e-8), self.motion_power)
        # Multi-scale joint synthesis:
        # P(fine_t | Hc, Hm) is proportional to
        # P_abs(fine_t)^alpha * P_motion(delta_t | Hm)^beta * P_coarse(c(fine_t) | Hc)^gamma
        combined_probs = (
            np.power(np.maximum(abs_candidate_probs, 1e-8), self.absolute_power)
            * motion_gate
            * coarse_gate
        )
        total = combined_probs.sum()
        if total <= 1e-8:
            combined_probs = abs_candidate_probs
            total = combined_probs.sum()
        if total <= 1e-8:
            combined_probs = np.full(len(abs_candidates), 1.0 / max(len(abs_candidates), 1), dtype=float)
        else:
            combined_probs = combined_probs / total

        candidate_hotspot_density = self.marginal_calibrator.hotspot_distribution[abs_candidates]
        combined_probs, projection_stats = self.physical_feasibility_projection(
            trimmed,
            abs_motion_tokens,
            combined_probs,
            candidate_hotspot_density
        )

        combined_cdf = np.cumsum(combined_probs)
        combined_cdf[-1] = 1.0
        cached = (abs_candidates, abs_motion_tokens, combined_cdf, projection_stats)
        self.joint_synthesis_lut[cache_key] = cached
        return cached

def collect_pst_reports_uid(traj_t, uid_states, pst: DynamicPST, sampled_uids, explore_uids):
    report_node_counts = {}
    explore_node_counts = {}
    coarse_report_counts = {}
    coarse_explore_counts = {}
    abs_local_counts = {}
    start_counts = np.zeros(pst.grid_map.size, dtype=int)
    confirmed_match_cache = {}
    broadcast_match_cache = {}
    coarse_confirmed_cache = {}
    coarse_broadcast_cache = {}
    sampled_uid_set = set(sampled_uids)
    explore_uid_set = set(explore_uids)

    for g1, g2, flag, uid in traj_t:
        if flag == 1:
            start_grid = g2.linear_index
            if uid in sampled_uid_set:
                start_counts[start_grid] += 1
            start_coarse = pst.coarse_mapper.map_fine_to_coarse(start_grid)
            uid_states[uid] = (start_grid, (), (start_coarse,))
            continue

        start_coarse = pst.coarse_mapper.map_fine_to_coarse(g1.linear_index)
        state = uid_states.get(uid, (g1.linear_index, (), (start_coarse,)))
        curr_grid_idx, motion_history, coarse_history = state
        should_report = uid in sampled_uid_set
        should_explore = uid in explore_uid_set

        if should_report:
            matched_context = confirmed_match_cache.get(motion_history)
            if matched_context is None:
                matched_context = pst.longest_confirmed_match(motion_history).context
                confirmed_match_cache[motion_history] = matched_context

            matched_node = pst.nodes[matched_context]
            if flag != 0:
                counts = None
            else:
                token = get_kinematic_token(g1, g2)
                counts = report_node_counts.get(matched_node.context)
            if counts is None:
                if flag == 0:
                    counts = np.zeros(len(matched_node.candidates), dtype=int)
                    report_node_counts[matched_node.context] = counts
            if flag == 0:
                token_pos = matched_node.token_to_pos[token]
                if token_pos >= 0:
                    counts[int(token_pos)] += 1

                abs_counts = abs_local_counts.get(curr_grid_idx)
                if abs_counts is None:
                    abs_node = pst.absolute_model.ensure_node(curr_grid_idx)
                    abs_counts = np.zeros(len(abs_node.candidates), dtype=int)
                    abs_local_counts[curr_grid_idx] = abs_counts
                else:
                    abs_node = pst.absolute_model.ensure_node(curr_grid_idx)
                abs_token_pos = abs_node.candidate_to_pos.get(g2.linear_index, -1)
                if abs_token_pos >= 0:
                    abs_counts[int(abs_token_pos)] += 1

                coarse_context = coarse_confirmed_cache.get(coarse_history)
                if coarse_context is None:
                    coarse_context = pst.coarse_model.longest_confirmed_match(coarse_history).context
                    coarse_confirmed_cache[coarse_history] = coarse_context
                coarse_node = pst.coarse_model.nodes[coarse_context]
                coarse_counts = coarse_report_counts.get(coarse_node.context)
                if coarse_counts is None:
                    coarse_counts = np.zeros(len(coarse_node.candidates), dtype=int)
                    coarse_report_counts[coarse_node.context] = coarse_counts
                coarse_token = pst.coarse_mapper.map_fine_to_coarse(g2.linear_index)
                coarse_pos = coarse_node.token_to_pos[coarse_token]
                if coarse_pos >= 0:
                    coarse_counts[int(coarse_pos)] += 1

        if should_explore:
            confirmed_context = confirmed_match_cache.get(motion_history)
            if confirmed_context is None:
                confirmed_context = pst.longest_confirmed_match(motion_history).context
                confirmed_match_cache[motion_history] = confirmed_context

            broadcast_context = broadcast_match_cache.get(motion_history)
            if broadcast_context is None:
                broadcast_context = pst.longest_broadcast_match(motion_history).context
                broadcast_match_cache[motion_history] = broadcast_context

            if len(broadcast_context) > len(confirmed_context):
                matched_node = pst.nodes[broadcast_context]
                if flag == 0:
                    token = get_kinematic_token(g1, g2)
                    counts = explore_node_counts.get(matched_node.context)
                    if counts is None:
                        counts = np.zeros(len(matched_node.candidates), dtype=int)
                        explore_node_counts[matched_node.context] = counts
                    token_pos = matched_node.token_to_pos[token]
                    if token_pos >= 0:
                        counts[int(token_pos)] += 1

            coarse_confirmed = coarse_confirmed_cache.get(coarse_history)
            if coarse_confirmed is None:
                coarse_confirmed = pst.coarse_model.longest_confirmed_match(coarse_history).context
                coarse_confirmed_cache[coarse_history] = coarse_confirmed

            coarse_broadcast = coarse_broadcast_cache.get(coarse_history)
            if coarse_broadcast is None:
                coarse_broadcast = pst.coarse_model.longest_broadcast_match(coarse_history).context
                coarse_broadcast_cache[coarse_history] = coarse_broadcast

            if len(coarse_broadcast) > len(coarse_confirmed) and flag == 0:
                coarse_node = pst.coarse_model.nodes[coarse_broadcast]
                coarse_counts = coarse_explore_counts.get(coarse_node.context)
                if coarse_counts is None:
                    coarse_counts = np.zeros(len(coarse_node.candidates), dtype=int)
                    coarse_explore_counts[coarse_node.context] = coarse_counts
                coarse_token = pst.coarse_mapper.map_fine_to_coarse(g2.linear_index)
                coarse_pos = coarse_node.token_to_pos[coarse_token]
                if coarse_pos >= 0:
                    coarse_counts[int(coarse_pos)] += 1

        if flag == 0:
            move_token = get_kinematic_token(g1, g2)
            coarse_token = pst.coarse_mapper.map_fine_to_coarse(g2.linear_index)
            uid_states[uid] = (
                g2.linear_index,
                tuple((motion_history + (move_token,))[-pst.max_depth:]),
                tuple((coarse_history + (coarse_token,))[-pst.coarse_model.max_depth:])
            )
        elif flag == 2 and uid in uid_states:
            uid_states.pop(uid, None)

    return (
        report_node_counts,
        explore_node_counts,
        coarse_report_counts,
        coarse_explore_counts,
        abs_local_counts,
        start_counts
    )


def cognitive_sampling_weights(users: Users, uid_states, pst: DynamicPST):
    report_weights = {}
    explore_weights = {}
    confirmed_match_cache = {}
    broadcast_match_cache = {}
    coarse_confirmed_cache = {}
    coarse_broadcast_cache = {}
    entropy_values = []
    coarse_entropy_values = []
    novelty_values = []
    report_weight_values = []
    explore_weight_values = []

    for uid in users.available_users:
        _, motion_history, coarse_history = uid_states.get(uid, (-1, (), ()))

        confirmed_context = confirmed_match_cache.get(motion_history)
        if confirmed_context is None:
            confirmed_context = pst.longest_confirmed_match(motion_history).context
            confirmed_match_cache[motion_history] = confirmed_context
        confirmed_node = pst.nodes[confirmed_context]

        broadcast_context = broadcast_match_cache.get(motion_history)
        if broadcast_context is None:
            broadcast_context = pst.longest_broadcast_match(motion_history).context
            broadcast_match_cache[motion_history] = broadcast_context
        broadcast_node = pst.nodes[broadcast_context]

        coarse_confirmed = coarse_confirmed_cache.get(coarse_history)
        if coarse_confirmed is None:
            coarse_confirmed = pst.coarse_model.longest_confirmed_match(coarse_history).context
            coarse_confirmed_cache[coarse_history] = coarse_confirmed
        coarse_confirmed_node = pst.coarse_model.nodes[coarse_confirmed]

        coarse_broadcast = coarse_broadcast_cache.get(coarse_history)
        if coarse_broadcast is None:
            coarse_broadcast = pst.coarse_model.longest_broadcast_match(coarse_history).context
            coarse_broadcast_cache[coarse_history] = coarse_broadcast
        coarse_broadcast_node = pst.coarse_model.nodes[coarse_broadcast]

        entropy = normalized_entropy(confirmed_node.probs)
        coarse_entropy = normalized_entropy(coarse_confirmed_node.probs)
        report_uncertainty = pst.node_uncertainty(confirmed_node)
        coarse_uncertainty = pst.coarse_model.node_uncertainty(coarse_confirmed_node)
        novelty = max(
            0.0,
            max(len(broadcast_context) - len(confirmed_context), len(coarse_broadcast) - len(coarse_confirmed))
        )
        novelty_boost = min(1.0, 0.35 * novelty)
        reliability_penalty = 1.0 - 0.5 * (
            pst.reliability_weight(confirmed_node) + pst.coarse_model.reliability_weight(coarse_confirmed_node)
        )
        report_weights[uid] = (
            0.12
            + 0.42 * report_uncertainty
            + 0.26 * coarse_uncertainty
            + 0.20 * reliability_penalty
        )

        explore_uncertainty = pst.node_uncertainty(broadcast_node)
        coarse_explore_uncertainty = pst.coarse_model.node_uncertainty(coarse_broadcast_node)
        branch_gain = max(0.0, getattr(broadcast_node, 'confirm_gain_ema', 0.0))
        branch_gain = min(1.0, branch_gain / max(pst.confirm_gain, 1e-8))
        coarse_branch_gain = max(0.0, getattr(coarse_broadcast_node, 'confirm_gain_ema', 0.0))
        coarse_branch_gain = min(1.0, coarse_branch_gain / max(pst.coarse_model.confirm_gain, 1e-8))
        should_explore = float(
            len(broadcast_context) > len(confirmed_context)
            or len(coarse_broadcast) > len(coarse_confirmed)
            or not broadcast_node.confirmed
            or not coarse_broadcast_node.confirmed
        )
        explore_weights[uid] = 0.05 + should_explore * (
            0.38 * explore_uncertainty
            + 0.22 * coarse_explore_uncertainty
            + 0.20 * novelty_boost
            + 0.12 * branch_gain
            + 0.08 * coarse_branch_gain
        )

        entropy_values.append(entropy)
        coarse_entropy_values.append(coarse_entropy)
        novelty_values.append(novelty_boost)
        report_weight_values.append(report_weights[uid])
        explore_weight_values.append(explore_weights[uid])

    stats = {
        'avg_entropy': float(np.mean(entropy_values)) if entropy_values else 0.0,
        'avg_coarse_entropy': float(np.mean(coarse_entropy_values)) if coarse_entropy_values else 0.0,
        'avg_novelty': float(np.mean(novelty_values)) if novelty_values else 0.0,
        'avg_report_weight': float(np.mean(report_weight_values)) if report_weight_values else 0.0,
        'avg_explore_weight': float(np.mean(explore_weight_values)) if explore_weight_values else 0.0,
        'available_users': len(entropy_values),
    }

    return report_weights, explore_weights, stats


def adaptive_sampling_rates(base_report_rate: float, base_explore_rate: float, sampling_stats, grid_scale: float):
    entropy_signal = float(np.clip(sampling_stats.get('avg_entropy', 0.0), 0.0, 1.0))
    coarse_entropy_signal = float(np.clip(sampling_stats.get('avg_coarse_entropy', 0.0), 0.0, 1.0))
    novelty_signal = float(np.clip(sampling_stats.get('avg_novelty', 0.0), 0.0, 1.0))
    report_weight_signal = float(np.clip(sampling_stats.get('avg_report_weight', 0.0), 0.0, 1.5))
    explore_weight_signal = float(np.clip(sampling_stats.get('avg_explore_weight', 0.0), 0.0, 1.5))

    report_multiplier = (
        0.50
        + 0.45 * entropy_signal
        + 0.25 * coarse_entropy_signal
        + 0.25 * min(1.0, report_weight_signal)
    )
    explore_multiplier = (
        0.20
        + 0.55 * entropy_signal
        + 0.30 * coarse_entropy_signal
        + 0.85 * novelty_signal
        + 0.30 * min(1.0, explore_weight_signal)
    )
    if grid_scale > 1.0:
        explore_multiplier += 0.10 * (grid_scale - 1.0)

    effective_report_rate = float(np.clip(base_report_rate * report_multiplier, 0.0, 1.0))
    effective_explore_rate = float(np.clip(base_explore_rate * explore_multiplier, 0.0, 1.0))
    return effective_report_rate, effective_explore_rate


def update_length_model(traj_t, uid_lengths, length_model: DPLengthDistribution, epsilon: float):
    quit_lengths = []

    for g1, _, flag, uid in traj_t:
        del g1
        if flag == 1:
            uid_lengths[uid] = 1
            continue

        curr_len = uid_lengths.get(uid, 1)
        if flag == 0:
            uid_lengths[uid] = curr_len + 1
        elif flag == 2:
            quit_lengths.append(curr_len)
            uid_lengths.pop(uid, None)

    length_model.update(quit_lengths, epsilon)


def collect_timestep_marginal_counts(traj_t, grid_size: int):
    counts = np.zeros(grid_size, dtype=int)
    for _, g2, flag, _ in traj_t:
        if flag in (0, 1):
            counts[g2.linear_index] += 1
    return counts


def generate_pst_points(
    synthetic_state: PSTSyntheticState,
    synthetic_tail_ids,
    synthetic_histories,
    synthetic_coarse_histories,
    synthetic_grid_indices,
    synthetic_lengths,
    synthetic_target_lengths,
    pst: DynamicPST,
    target_n: int,
    avg_len: float,
    length_model: DPLengthDistribution
):
    del avg_len
    timestamp = synthetic_state.advance_time()
    new_tail_ids = []
    new_histories = []
    new_coarse_histories = []
    new_grid_indices = []
    new_lengths = []
    new_target_lengths = []
    grouped_indices = defaultdict(list)

    for idx, history in enumerate(synthetic_histories):
        target_len = int(synthetic_target_lengths[idx])
        curr_len = int(synthetic_lengths[idx])
        if curr_len >= target_len:
            synthetic_state.finish([synthetic_tail_ids[idx]])
            continue
        current_grid_idx = int(synthetic_grid_indices[idx])
        coarse_history = synthetic_coarse_histories[idx]
        grouped_indices[(current_grid_idx, history, coarse_history)].append(idx)

    for (current_grid_idx, history, coarse_history), indices in grouped_indices.items():
        candidate_grids, candidate_tokens, combined_cdf, projection_stats = pst.joint_motion_cdf(
            current_grid_idx,
            history,
            coarse_history
        )
        pst.record_projection_stats(projection_stats, len(indices))
        sampled_ids = np.searchsorted(combined_cdf, np.random.random(len(indices)), side='right')
        next_grid_indices = candidate_grids[sampled_ids]
        next_motion_tokens = candidate_tokens[sampled_ids]
        prev_tail_ids = np.asarray([synthetic_tail_ids[local_idx] for local_idx in indices], dtype=np.int64)
        updated_tail_ids = synthetic_state.append_points(prev_tail_ids, next_grid_indices, timestamp)
        for local_idx, updated_tail_id, next_grid_idx, next_token in zip(indices,
                                                                         updated_tail_ids.tolist(),
                                                                         next_grid_indices.tolist(),
                                                                         next_motion_tokens.tolist()):
            new_tail_ids.append(int(updated_tail_id))
            new_histories.append(tuple((history + (int(next_token),))[-pst.max_depth:]))
            next_coarse = pst.coarse_mapper.map_fine_to_coarse(next_grid_idx)
            new_coarse_histories.append(
                tuple((coarse_history + (next_coarse,))[-pst.coarse_model.max_depth:])
            )
            new_grid_indices.append(int(next_grid_idx))
            new_lengths.append(int(synthetic_lengths[local_idx]) + 1)
            new_target_lengths.append(int(synthetic_target_lengths[local_idx]))

    active_n = len(new_tail_ids)
    if active_n < target_n:
        sampled = pst.sample_start_grid(target_n - active_n)
        sampled_lengths = length_model.sample_lengths(target_n - active_n)
        started_tail_ids = synthetic_state.start_points(sampled, timestamp)
        new_tail_ids.extend(int(tail_id) for tail_id in started_tail_ids.tolist())
        new_histories.extend(() for _ in sampled.tolist())
        new_coarse_histories.extend(
            (pst.coarse_mapper.map_fine_to_coarse(int(grid_idx)),) for grid_idx in sampled.tolist()
        )
        new_grid_indices.extend(int(grid_idx) for grid_idx in sampled.tolist())
        new_lengths.extend(1 for _ in sampled.tolist())
        new_target_lengths.extend(int(length) for length in sampled_lengths.tolist())
    elif active_n > target_n:
        overflow = active_n - target_n
        traj_lens = np.asarray(new_lengths, dtype=np.int32)
        target_lens = np.asarray(new_target_lengths, dtype=np.int32)
        quit_scores = np.fromiter(
            (
                1.0 / max(int(target_len) - int(traj_len) + 1, 1)
                for traj_len, target_len in zip(traj_lens.tolist(), target_lens.tolist())
            ),
            dtype=float,
            count=active_n
        )

        if quit_scores.sum() <= 1e-8:
            removed_ids = np.random.choice(active_n, size=overflow, replace=False)
        else:
            quit_scores += 1e-8
            removed_ids = np.random.choice(
                active_n,
                size=overflow,
                replace=False,
                p=quit_scores / quit_scores.sum()
            )

        keep_mask = np.ones(active_n, dtype=bool)
        keep_mask[removed_ids] = False
        synthetic_state.finish_trimmed([new_tail_ids[int(idx)] for idx in removed_ids.tolist()])
        new_tail_ids = [tail_id for idx, tail_id in enumerate(new_tail_ids) if keep_mask[idx]]
        new_histories = [history for idx, history in enumerate(new_histories) if keep_mask[idx]]
        new_coarse_histories = [history for idx, history in enumerate(new_coarse_histories) if keep_mask[idx]]
        new_grid_indices = [grid_idx for idx, grid_idx in enumerate(new_grid_indices) if keep_mask[idx]]
        new_lengths = [traj_len for idx, traj_len in enumerate(new_lengths) if keep_mask[idx]]
        new_target_lengths = [target_len for idx, target_len in enumerate(new_target_lengths) if keep_mask[idx]]

    return new_tail_ids, new_histories, new_coarse_histories, new_grid_indices, new_lengths, new_target_lengths

def RetraSyn(traj_stream, w: int, eps, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)

    synthetic_db = SynDB()
    trans_distribution = []
    used_budget = []
    release = []
    used_budget_in_curr_w = 0
    # number of significant transitions
    N_st = []

    for t in range(2):
        # warm-up stage
        oue = OUE(eps / w, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        est_counts = oue.non_negative_data / oue.n

        # generate Markov matrix
        markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                              trans_domain)
        trans_distribution.append(markov_mat)
        release.append(est_counts / est_counts.sum())

        # generate new points in synthetic data based on current distribution
        synthetic_db.generate_new_points(markov_mat, grid_map, avg_lens[args.dataset])

        # adjust size of synthetic database
        synthetic_db.adjust_data_size(markov_mat, len(traj_stream[t]), grid_map, end_distribution)

        used_budget.append(eps / w)
        used_budget_in_curr_w += eps / w
        N_st.append(0)

    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue

        # budget recycling
        if t >= w:
            used_budget_in_curr_w -= used_budget[t - w]
        # calculate remaining budget
        eps_rm = eps - used_budget_in_curr_w

        # calculate deviation
        dev = np.abs(np.array(release[-1]) - np.average(release[max(0, t - 5):t], axis=0)).sum()

        # calculate allocation portion
        cr = max(0.5, 1 - np.average(N_st[max(0, t - 5):t]) / len(trans_domain))
        p = utils.allocation_p(dev, w, alpha=8)
        p = min(p * cr, 0.6)
        eps_t = p * eps_rm

        oue = OUE(eps_t, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()
        f_hat = oue.non_negative_data / oue.n
        f_tilde = release[-1]

        # select significant patterns
        variance = 4 * math.exp(eps_t) / (oue.n * (math.exp(eps_t) - 1) ** 2)
        select = (f_tilde - f_hat) ** 2 > variance

        # merge significant patterns and other patterns
        counts = np.zeros(len(trans_domain))
        sig_counts = oue.non_negative_data / oue.n

        for i in range(len(select)):
            if select[i]:
                counts[i] = sig_counts[i]
            else:
                counts[i] = f_tilde[i] * sig_counts.sum()

        used_budget.append(eps_t)
        used_budget_in_curr_w += eps_t
        # generate Markov matrix
        markov_mat, end_distribution = generate_markov_matrix(counts, trans_domain)

        # generate new points in synthetic data based on current distribution
        synthetic_db.generate_new_points(markov_mat, grid_map, avg_lens[args.dataset])

        # check entering distribution
        if markov_mat[-1].sum() == 0:
            for i in range(t):
                if not trans_distribution[t - i - 1][-1].sum() == 0:
                    markov_mat[-1] = trans_distribution[t - i - 1][-1]
                    break

        # adjust size of synthetic database
        synthetic_db.adjust_data_size(markov_mat, len(traj_stream[t]), grid_map, end_distribution)

        trans_distribution.append(markov_mat)
        release.append(counts / counts.sum())

        N_st.append(np.sum(select))

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db


def RetraSynPST(traj_stream, w: int, eps, trans_domain: List[Transition]):
    del trans_domain
    grid_scale = max(1.0, grid_map.n / 6.0)
    context_grid_n = min(grid_map.n, max(2, int(args.pst_context_grid_num)))
    coarse_mapper = CoarseGridMapper(grid_map, context_grid_n)
    marginal_prior_mix = 0.42 if args.pst_profile == 'balanced' else 0.25
    marginal_ema = 0.22 if args.pst_profile == 'balanced' else 0.15
    marginal_prior_mix = min(0.68, marginal_prior_mix * math.sqrt(grid_scale))
    marginal_ema = max(0.10, marginal_ema / math.sqrt(grid_scale))
    length_eps = max(1e-3, args.pst_length_eps_frac * eps)
    marginal_eps = max(1e-3, min(0.24 * eps, args.pst_marginal_eps_frac * eps * math.sqrt(grid_scale)))
    remaining_eps = max(eps - length_eps - marginal_eps, 1e-3)
    explore_budget_ratio = max(0.30, args.pst_explore_budget_ratio - 0.10 * (grid_scale - 1.0))
    explore_eps = max(1e-3, remaining_eps * explore_budget_ratio)
    report_eps = max(1e-3, remaining_eps - explore_eps)
    context_budget_ratio = float(np.clip(args.pst_context_budget_ratio, 0.05, 0.60))
    context_report_eps = max(1e-3, report_eps * context_budget_ratio)
    motion_report_eps = max(1e-3, report_eps - context_report_eps)
    context_explore_eps = max(1e-3, explore_eps * context_budget_ratio)
    motion_explore_eps = max(1e-3, explore_eps - context_explore_eps)

    pst = DynamicPST(
        grid_map,
        coarse_mapper,
        max_depth=min(3, max(2, w // 6 + 1)),
        split_sigma=args.pst_split_sigma,
        confirm_gain=args.pst_confirm_gain,
        marginal_prior_mix=marginal_prior_mix,
        marginal_ema=marginal_ema,
        momentum_min_candidates=args.pst_momentum_min_candidates,
        consistency_mix=args.pst_consistency_mix,
        grid_scale=grid_scale
    )
    synthetic_state = PSTSyntheticState()
    synthetic_tail_ids = []
    uid_states = {}
    uid_lengths = {}
    synthetic_histories = []
    synthetic_coarse_histories = []
    synthetic_grid_indices = []
    synthetic_lengths = []
    synthetic_target_lengths = []
    users = Users()
    sampled_user_history = []
    quitted_users = []
    length_model = DPLengthDistribution(avg_lens[args.dataset], args.pst_length_max)
    base_report_sample_rate = min(1.0, max(0.0, args.pst_report_rate / max(w, 1)))
    base_explore_sample_rate = min(1.0, max(0.0, args.pst_explore_rate / max(w, 1)))
    sampling_monitor = defaultdict(list)

    for t in range(len(traj_stream)):
        pst.activate_broadcasts(t)
        pst.coarse_model.activate_broadcasts(t)

        if not traj_stream[t]:
            synthetic_state.advance_time()
            sampled_user_history.append(([], []))
            continue

        if t >= w:
            old_confirmed, old_explore = sampled_user_history[t - w]
            for uid in old_confirmed:
                users.recycle(uid)
            for uid in old_explore:
                users.recycle(uid)

        for uid in quitted_users:
            users.remove(uid)
            uid_states.pop(uid, None)
        quitted_users = []

        for _, _, flag, uid in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quitted_users.append(uid)

        report_weight_map, explore_weight_map, sampling_stats = cognitive_sampling_weights(users, uid_states, pst)
        if args.pst_sampling_mode == 'uniform':
            report_sample_rate = base_report_sample_rate
            explore_sample_rate = base_explore_sample_rate
            sampled_users = users.sample(report_sample_rate)
            explore_users = users.sample(explore_sample_rate)
        else:
            report_sample_rate, explore_sample_rate = adaptive_sampling_rates(
                base_report_sample_rate,
                base_explore_sample_rate,
                sampling_stats,
                grid_scale
            )
            sampled_users = users.weighted_sample(report_sample_rate, report_weight_map)
            explore_users = users.weighted_sample(explore_sample_rate, explore_weight_map)
        sampling_monitor['entropy'].append(sampling_stats['avg_entropy'])
        sampling_monitor['coarse_entropy'].append(sampling_stats['avg_coarse_entropy'])
        sampling_monitor['novelty'].append(sampling_stats['avg_novelty'])
        sampling_monitor['report_weight'].append(sampling_stats['avg_report_weight'])
        sampling_monitor['explore_weight'].append(sampling_stats['avg_explore_weight'])
        sampling_monitor['report_rate'].append(report_sample_rate)
        sampling_monitor['explore_rate'].append(explore_sample_rate)
        sampling_monitor['report_users'].append(len(sampled_users))
        sampling_monitor['explore_users'].append(len(explore_users))

        phase_start = perf_counter()
        (
            report_node_counts,
            explore_node_counts,
            coarse_report_counts,
            coarse_explore_counts,
            abs_local_counts,
            start_counts
        ) = collect_pst_reports_uid(
            traj_stream[t],
            uid_states,
            pst,
            sampled_users,
            explore_users
        )
        marginal_counts = collect_timestep_marginal_counts(traj_stream[t], grid_map.size)
        update_length_model(traj_stream[t], uid_lengths, length_model, length_eps)

        if start_counts.sum() > 0:
            pst.update_start_distribution(start_counts, motion_report_eps)
        pst.update_timestep_marginal(marginal_counts, marginal_eps)

        created_nodes = 0
        for context, counts in report_node_counts.items():
            node = pst.ensure_node(context)
            pst.update_node_distribution(node, counts, motion_report_eps, t)
            created_nodes += pst.maybe_split_node(node, t)
        for grid_idx, counts in abs_local_counts.items():
            pst.absolute_model.update_node_distribution(grid_idx, counts, motion_report_eps)
        for context, counts in coarse_report_counts.items():
            node = pst.coarse_model.ensure_node(context)
            pst.coarse_model.update_node_distribution(node, counts, context_report_eps, t)
            created_nodes += pst.coarse_model.maybe_split_node(node, t)
        for context, counts in explore_node_counts.items():
            node = pst.ensure_node(context)
            pst.update_node_distribution(node, counts, motion_explore_eps, t)
            created_nodes += pst.maybe_split_node(node, t)
        for context, counts in coarse_explore_counts.items():
            node = pst.coarse_model.ensure_node(context)
            pst.coarse_model.update_node_distribution(node, counts, context_explore_eps, t)
            created_nodes += pst.coarse_model.maybe_split_node(node, t)
        pst.apply_hierarchical_consistency()
        pst.invalidate_synthesis_lut()
        runtime_profiler.add('pst_update', perf_counter() - phase_start)

        phase_start = perf_counter()
        synthetic_tail_ids, synthetic_histories, synthetic_coarse_histories, synthetic_grid_indices, synthetic_lengths, synthetic_target_lengths = generate_pst_points(
            synthetic_state,
            synthetic_tail_ids,
            synthetic_histories,
            synthetic_coarse_histories,
            synthetic_grid_indices,
            synthetic_lengths,
            synthetic_target_lengths,
            pst,
            len(traj_stream[t]),
            avg_lens[args.dataset],
            length_model
        )
        runtime_profiler.add('trajectory_synthesis', perf_counter() - phase_start)

        sampled_user_history.append((sampled_users, explore_users))
        for uid in sampled_users:
            users.deactivate(uid)
        for uid in explore_users:
            users.deactivate(uid)

        if (t + 1) % 100 == 0:
            confirmed_nodes = sum(1 for node in pst.nodes.values() if node.confirmed)
            broadcasted_nodes = sum(1 for node in pst.nodes.values() if node.broadcasted)
            projection_stats = pst.projection_summary(reset=True)
            logger.info(
                f'{t + 1} timestamps processed, PST nodes={len(pst.nodes)}, '
                f'confirmed_nodes={confirmed_nodes}, broadcasted_nodes={broadcasted_nodes}, new_nodes={created_nodes}'
            )
            logger.info(
                'Sampling stats: '
                f"entropy={np.mean(sampling_monitor['entropy']):.4f}, "
                f"coarse_entropy={np.mean(sampling_monitor['coarse_entropy']):.4f}, "
                f"novelty={np.mean(sampling_monitor['novelty']):.4f}, "
                f"report_weight={np.mean(sampling_monitor['report_weight']):.4f}, "
                f"explore_weight={np.mean(sampling_monitor['explore_weight']):.4f}, "
                f"report_rate={np.mean(sampling_monitor['report_rate']):.4f}, "
                f"explore_rate={np.mean(sampling_monitor['explore_rate']):.4f}, "
                f"report_users={np.mean(sampling_monitor['report_users']):.1f}, "
                f"explore_users={np.mean(sampling_monitor['explore_users']):.1f}"
            )
            logger.info(
                'Projection stats: '
                f"rejection_mass={projection_stats['projection_rejection_mass']:.4f}, "
                f"feasible_mass_ratio={projection_stats['feasible_mass_ratio']:.4f}, "
                f"violation_reduction={projection_stats['physical_violation_reduction']:.4f}, "
                f"adaptive_strength={projection_stats['adaptive_projection_strength']:.4f}, "
                f"adaptive_floor={projection_stats['adaptive_projection_floor']:.4f}, "
                f"hotspot_density={projection_stats['candidate_hotspot_density']:.4f}, "
                f"raw_violation={projection_stats['raw_physical_violation']:.4f}, "
                f"projected_violation={projection_stats['projected_physical_violation']:.4f}, "
                f"samples={projection_stats['samples']:.0f}"
            )
            sampling_monitor = defaultdict(list)

    syn_db = synthetic_state.build_syndb(synthetic_tail_ids, pst.grid_lookup)
    projection_totals = pst.projection_summary(reset=False)
    syn_db.metadata = {
        'method': synthetic_method_tag('retrasyn_pst'),
        'dataset': args.dataset,
        'epsilon': eps,
        'grid_num': grid_map.n,
        'context_grid_num': coarse_mapper.coarse_n,
        'context_ratio': coarse_mapper.resolution_ratio,
        'max_depth': pst.max_depth,
        'context_depth': pst.coarse_model.max_depth,
        'length_eps': length_eps,
        'marginal_eps': marginal_eps,
        'motion_report_eps': motion_report_eps,
        'motion_explore_eps': motion_explore_eps,
        'context_report_eps': context_report_eps,
        'context_explore_eps': context_explore_eps,
        'report_rate_base': base_report_sample_rate,
        'explore_rate_base': base_explore_sample_rate,
        'sampling_mode': args.pst_sampling_mode,
        'coarse_mix': pst.coarse_power,
        'projection_strength': pst.projection_strength,
        'projection_feasible_floor': pst.projection_feasible_floor,
        'projection_density_gamma': pst.projection_density_gamma,
        'projection_speed_weight': pst.speed_weight,
        'projection_turn_weight': pst.turn_weight,
        'projection_accel_weight': pst.accel_weight,
        'projection_rejection_mass': projection_totals['projection_rejection_mass'],
        'feasible_mass_ratio': projection_totals['feasible_mass_ratio'],
        'physical_violation_reduction': projection_totals['physical_violation_reduction'],
        'raw_physical_violation': projection_totals['raw_physical_violation'],
        'projected_physical_violation': projection_totals['projected_physical_violation'],
        'adaptive_projection_strength': projection_totals['adaptive_projection_strength'],
        'adaptive_projection_floor': projection_totals['adaptive_projection_floor'],
        'candidate_hotspot_density': projection_totals['candidate_hotspot_density'],
        'runtime': dict(runtime_profiler.timings),
        'innovation_2_formula': 'P(fine_t|Hc,Hm) proportional to P_abs(fine_t)^alpha * P_motion(delta_t|Hm)^beta * P_coarse(c(fine_t)|Hc)^gamma'
    }
    return syn_db


def lbd(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    release = []
    used_budget = []

    synthetic_db = SynDB()
    trans_distribution = []

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    eps_rm = eps / 2
    oue = OUE(eps_rm / 2, len(trans_domain), lambda x: trans_domain_map[x])

    for (g1, g2, flag) in traj_stream[1]:
        trans = Transition(g1, g2, flag)
        oue.privatise(trans)

    oue.adjust()
    est_counts = oue.non_negative_data / oue.n
    release.append(est_counts / est_counts.sum())

    # generate Markov matrix
    markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                          trans_domain)
    trans_distribution.append(markov_mat)

    # generate new points in synthetic data based on current distribution
    synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

    used_budget.append(eps_rm / 2)
    used_budget_in_curr_w = eps_rm / 2

    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue
        # set dissimilarity budget
        eps_1 = eps / (2 * w)
        # estimate c_t
        oue = OUE(eps_1, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()
        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity
        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps_1) / (oue.n * (math.exp(eps_1) - 1) ** 2)

        if t >= w:
            # budget recovery
            used_budget_in_curr_w -= used_budget[t - w]
        eps_rm = eps / 2 - used_budget_in_curr_w

        err = 4 * math.exp(eps_rm / 2) / (oue.n * (math.exp(eps_rm / 2) - 1) ** 2)

        if dis > err:
            # perturbation
            oue = OUE(eps_rm/2, len(trans_domain), lambda x: trans_domain_map[x])

            for (g1, g2, flag) in traj_stream[t]:
                trans = Transition(g1, g2, flag)
                oue.privatise(trans)
            oue.adjust()
            est_counts = oue.non_negative_data / oue.n
            release.append(est_counts / est_counts.sum())
            used_budget.append(eps_rm / 2)
            used_budget_in_curr_w += eps_rm / 2
        else:
            # approximation
            release.append(release[-1])
            used_budget.append(0)

        markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')
    return synthetic_db


def lba(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    release = []
    l: int = 0
    eps_l2 = 0

    synthetic_db = SynDB()
    trans_distribution = []

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    eps_2 = eps / (2 * w)
    oue = OUE(eps_2, len(trans_domain), lambda x: trans_domain_map[x])

    for (g1, g2, flag) in traj_stream[1]:
        trans = Transition(g1, g2, flag)
        oue.privatise(trans)
    oue.adjust()
    est_counts = oue.non_negative_data / oue.n
    release.append(est_counts/est_counts.sum())

    # generate Markov matrix
    markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                          trans_domain)
    trans_distribution.append(markov_mat)

    # generate new points in synthetic data based on current distribution
    synthetic_db.generate_new_points_baseline(markov_mat, grid_map)
    l = 1
    eps_l2 = eps_2

    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue
        # set dissimilarity budget
        eps_1 = eps / (2 * w)
        # estimate c_t
        oue = OUE(eps_1, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag) in traj_stream[t]:
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()
        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity
        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps_1) / (oue.n * (math.exp(eps_1) - 1) ** 2)

        # calculate nullified timestamps
        t_N = eps_l2 / (eps / (2 * w)) - 1

        if t - l <= t_N:
            # nullified timestamp
            release.append(release[-1])
        else:
            # calculate absorbed timestamps
            t_A = t - (l + t_N)
            eps_2 = eps / (2 * w) * min(t_A, w)
            err = 4 * math.exp(eps_2) / (oue.n * (math.exp(eps_2) - 1) ** 2)

            if dis > err:
                # perturbation
                oue = OUE(eps_2, len(trans_domain), lambda x: trans_domain_map[x])

                for (g1, g2, flag) in traj_stream[t]:
                    trans = Transition(g1, g2, flag)
                    oue.privatise(trans)
                oue.adjust()
                est_counts = oue.non_negative_data / oue.n
                release.append(est_counts/est_counts.sum())
                l = t
                eps_l2 = eps_2
            else:
                # approximation
                release.append(release[-1])

        markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db

avg_lens = {
    'tdrive': 13.61,
    'oldenburg': 59.98,
    'sanjoaquin': 55.3
}

timestamps = {
    'tdrive': 886,
    'oldenburg': 500,
    'sanjoaquin': 1000
}

logger.info('Reading dataset...')
use_uid_stream = args.method == 'retrasyn_pst'
dataset_suffix = '_transition_id.xz' if use_uid_stream else '_transition.xz'
stats_func = utils.tid_dataset_stats if use_uid_stream else utils.t_dataset_stats
with lzma.open(f'./data/{args.dataset}{dataset_suffix}', 'rb') as f:
    dataset = pickle.load(f)[:timestamps[args.dataset]]

stats = stats_func(dataset, f'./data/{args.dataset}_stats.json')
grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])

logger.info('Spatial decomposition...')
phase_start = perf_counter()
if args.multiprocessing:
    def decomp_multi(xy_l):
        if use_uid_stream:
            return spatial_decomposition_uid(xy_l, grid_map)
        return spatial_decomposition(xy_l, grid_map)


    pool = multiprocessing.Pool(CORES)
    grid_db = pool.map(decomp_multi, dataset)
    pool.close()
else:
    if use_uid_stream:
        grid_db = [spatial_decomposition_uid(xy_l, grid_map) for xy_l in dataset]
    else:
        grid_db = [spatial_decomposition(xy_l, grid_map) for xy_l in dataset]

if use_uid_stream:
    grid_db = split_traj_uid(grid_db, grid_map, max_step=adaptive_spatial_radius(grid_map.n))
else:
    grid_db = split_traj(grid_db, grid_map)
runtime_profiler.add('spatial_decomposition', perf_counter() - phase_start)


if args.method == 'retrasyn':
    logger.info('RetraSyn ...')
    syn_grid_db = RetraSyn(grid_db, args.w, args.epsilon,
                           grid_map.get_all_transition())
elif args.method == 'retrasyn_pst':
    logger.info('RetraSyn-PST...')
    syn_grid_db = RetraSynPST(grid_db, args.w, args.epsilon,
                              grid_map.get_all_transition())
elif args.method == 'lbd':
    logger.info('LBD...')
    syn_grid_db = lbd(grid_db, args.w, args.epsilon, grid_map.get_all_transition())
elif args.method == 'lba':
    logger.info('LBA...')
    syn_grid_db = lba(grid_db, args.w, args.epsilon, grid_map.get_all_transition())
else:
    logger.info('Invalid method name!')
    exit()

phase_start = perf_counter()
syn_xy_db = convert_grid_to_raw(syn_grid_db.all_data)
syn_path = Path(f'./data/syn_data/{args.dataset}/{synthetic_method_tag(args.method)}_{args.epsilon}_g{args.grid_num}_w{args.w}.pkl')
with open(syn_path, 'wb') as f:
    pickle.dump(syn_xy_db, f)
if args.method == 'retrasyn_pst' and hasattr(syn_grid_db, 'metadata'):
    meta_path = syn_path.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(syn_grid_db.metadata, f, indent=2)
runtime_profiler.add('write_file', perf_counter() - phase_start)
if args.method == 'retrasyn_pst':
    runtime_profiler.log_summary(logger)
