import numpy as np
from grid import Grid, GridMap, Transition
from typing import List, Tuple
import utils
import random
import math
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


class Query:
    def __init__(self):
        pass

    def point_query(self, db):
        raise NotImplementedError

    def point_query_t(self, db):
        print('Temporal query is not supported!')


class SquareQuery(Query):
    def __init__(self,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float,
                 max_time: int,
                 time_range=5,
                 size_factor=9.0):
        super().__init__()
        self.edge = math.sqrt((max_x - min_x) * (max_y - min_y) / size_factor)
        # Randomly select center
        center_x = random.random() * (max_x - min_x - self.edge) + min_x + self.edge / 2
        center_y = random.random() * (max_y - min_y - self.edge) + min_y + self.edge / 2
        self.center = (center_x, center_y)

        self.left_x = center_x - self.edge / 2
        self.up_y = center_y + self.edge / 2
        self.right_x = center_x + self.edge / 2
        self.down_y = center_y - self.edge / 2

        self.min_t = random.randint(0, max_time - time_range)
        self.max_t = self.min_t + time_range - 1

    def in_square(self, point: Tuple[float, float]):
        return self.left_x <= point[0] <= self.right_x and self.down_y <= point[1] <= self.up_y

    def in_square_t(self, point: Tuple[float, float, int]):
        return self.min_t <= point[2] <= self.max_t and self.in_square((point[0], point[1]))

    def point_query(self, db: List[List[Tuple[float, float]]]):
        count = 0
        for t in db:
            for p in t:
                if self.in_square(p):
                    count += 1

        return count

    def point_query_t(self, db: List[List[Tuple[float, float, int]]]):
        count = 0
        for t in db:
            for p in t:
                if self.in_square_t(p):
                    count += 1
        return count


class Pattern:
    def __init__(self, grids: List[Grid]):
        self.grids = grids

    @property
    def size(self):
        return len(self.grids)

    def __eq__(self, other):
        if other is None:
            return False
        if not type(other) == Pattern:
            return False
        if not other.size == self.size:
            return False

        for i in range(self.size):
            if not self.grids[i].index == other.grids[i].index:
                return False

        return True

    def __hash__(self):
        prime = 31
        result = 1
        for g in self.grids:
            result = result * prime + g.__hash__()

        return result

    def check_pattern(self):
        if self.size <= 2:
            return True
        for i in range(self.size - 1):
            if self.grids[i].equal(self.grids[i + 1]):
                return False
        return True


def eval_st_query_error(orig_db, syn_db, queries: List[SquareQuery], sanity_bound=0.01, upt=34000):
    def build_points_by_time(db, max_time):
        buckets = [[] for _ in range(max_time)]
        for traj in db:
            for x, y, t in traj:
                if 0 <= t < max_time:
                    buckets[t].append((x, y))
        return [np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float) for points in buckets]

    def count_query(points_by_time, query):
        count = 0
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
            count += int(np.count_nonzero(mask))
        return count

    max_time = max(query.max_t for query in queries) + 1
    orig_points_by_time = build_points_by_time(orig_db, max_time)
    syn_points_by_time = build_points_by_time(syn_db, max_time)

    actual_ans = list()
    syn_ans = list()

    average_total_points = upt * (queries[0].max_t - queries[0].min_t + 1)

    for q in queries:
        actual_ans.append(count_query(orig_points_by_time, q))
        syn_ans.append(count_query(syn_points_by_time, q))

    actual_ans = np.asarray(actual_ans)
    syn_ans = np.asarray(syn_ans)
    numerator = np.abs(actual_ans - syn_ans)
    # use sanity bound to mitigate the effect of extremely small actual_ans
    denominator = np.maximum(actual_ans, average_total_points * sanity_bound)

    error = numerator / denominator

    return np.mean(error)


def eval_jsd(true, release):
    true_arr = np.asarray(true, dtype=float)
    release_arr = np.asarray(release, dtype=float)
    avg_prob = (true_arr + release_arr) / 2
    kl_true = np.log((true_arr + 1e-8) / (avg_prob + 1e-8)) * true_arr
    kl_release = np.log((release_arr + 1e-8) / (avg_prob + 1e-8)) * release_arr
    return np.mean(0.5 * np.sum(kl_true, axis=-1) + 0.5 * np.sum(kl_release, axis=-1))


def mine_patterns(db: List[List[Tuple[Grid, int]]], min_time, max_time, min_size=2, max_size=5):
    pattern_dict = {}
    max_pattern_size = min(max_size, max_time - min_time + 1)
    for traj in db:
        if len(traj) < min_size:
            continue

        start = 0
        while start < len(traj) and traj[start][1] < min_time:
            start += 1
        end = len(traj)
        while end > start and traj[end - 1][1] > max_time:
            end -= 1
        if end - start < min_size:
            continue

        indices = [g.linear_index for g, _ in traj[start:end]]
        for curr_size in range(min_size, min(max_pattern_size, len(indices)) + 1):
            limit = len(indices) - curr_size + 1
            for i in range(limit):
                pattern = indices[i:i + curr_size]
                if curr_size > 2 and any(pattern[j] == pattern[j + 1] for j in range(curr_size - 1)):
                    continue
                pattern_key = tuple(pattern)
                pattern_dict[pattern_key] = pattern_dict.get(pattern_key, 0) + 1

    return pattern_dict


def calculate_pattern_f1(orig_pattern,
                         syn_pattern,
                         k=100):
    sorted_orig = sorted(orig_pattern.items(), key=lambda x: x[1], reverse=True)
    sorted_syn = sorted(syn_pattern.items(), key=lambda x: x[1], reverse=True)

    orig_top_k = [x[0] for x in sorted_orig][:k]
    syn_top_k = [x[0] for x in sorted_syn][:k]
    if not orig_top_k or not syn_top_k:
        return 0

    orig_top_k_set = set(orig_top_k)
    count = 0
    for p1 in syn_top_k:
        if p1 in orig_top_k_set:
            count += 1

    precision = count / len(syn_top_k)
    recall = count / len(orig_top_k)

    return 2 * precision * recall / (precision + recall) if precision else 0


def get_grid_count(grid_db: List[List[Tuple[Grid, int]]], domain: List[Grid], max_time, min_time=0):
    """
    Return a list of grid counts for each timestamp
    """
    grid_counts = np.zeros((max_time, len(domain)), dtype=float)
    for traj in grid_db:
        for (g, t) in traj:
            if t < min_time or t >= max_time:
                continue
            grid_counts[t][g.linear_index] += 1

    return grid_counts


def get_transition_count(grid_db: List[List[Tuple[Grid, int]]], domain: List[Transition], max_time, min_time=0):
    grid_size = max(max(trans.g1.linear_index, trans.g2.linear_index) for trans in domain) + 1
    transition_lookup = -np.ones((grid_size, grid_size), dtype=int)
    for idx, trans in enumerate(domain):
        transition_lookup[trans.g1.linear_index, trans.g2.linear_index] = idx

    trans_counts = np.zeros((max_time - 1, len(domain)), dtype=float)
    for traj in grid_db:
        for i in range(len(traj) - 1):
            t = traj[i][1]
            if t < min_time or t >= max_time - 1:
                continue
            curr_index = traj[i][0].linear_index
            next_index = traj[i + 1][0].linear_index
            trans_index = transition_lookup[curr_index, next_index]
            if trans_index >= 0:
                trans_counts[t][trans_index] += 1

    return trans_counts


def eval_hotspot_ndcg(orig_counts, syn_counts, k=10):
    orig_sum = orig_counts.sum()
    syn_sum = syn_counts.sum()
    if orig_sum <= 0 or syn_sum <= 0:
        return 0

    orig_density = orig_counts / orig_sum
    syn_density = syn_counts / syn_sum
    orig_top_k = np.argsort(-orig_density)[:k]
    syn_top_k = np.argsort(-syn_density)[:k]
    orig_rank = {index: rank + 1 for rank, index in enumerate(orig_top_k)}

    r = np.zeros(k)

    for i, p1 in enumerate(syn_top_k):
        rank = orig_rank.get(int(p1))
        if rank is not None:
            r[i] = 1 / rank

    idcg = np.sum((np.ones(k) / np.arange(1, k + 1)) * 1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(r * 1. / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg else 0


def calculate_coverage_kendall_tau(orig_db: List[List[Tuple[Grid, int]]],
                                   syn_db: List[List[Tuple[Grid, int]]],
                                   grid_map: GridMap):
    actual_counts = np.zeros(grid_map.size)
    syn_counts = np.zeros(grid_map.size)

    # For each grid, find how many trajectories pass through it
    for traj in orig_db:
        passed = {utils.grid_index_map_func(g, grid_map) for g, _ in traj}
        for index in passed:
            actual_counts[index] += 1

    for traj in syn_db:
        passed = {utils.grid_index_map_func(g, grid_map) for g, _ in traj}
        for index in passed:
            syn_counts[index] += 1

    actual_diff = actual_counts[:, None] - actual_counts[None, :]
    syn_diff = syn_counts[:, None] - syn_counts[None, :]
    upper_idx = np.triu_indices(grid_map.size, k=1)
    pair_products = actual_diff[upper_idx] * syn_diff[upper_idx]
    concordant_pairs = np.count_nonzero(pair_products > 0)
    reversed_pairs = np.count_nonzero(pair_products < 0)

    denominator = grid_map.size * (grid_map.size - 1) / 2
    return (concordant_pairs - reversed_pairs) / denominator


def calculate_length_error(orig_db: List[List[Tuple[float, float, int]]],
                           syn_db: List[List[Tuple[float, float, int]]],
                           bucket_num=20):
    orig_length = [utils.get_travel_distance(t) for t in orig_db]
    syn_length = [utils.get_travel_distance(t) for t in syn_db]
    if not orig_length or not syn_length:
        return 0

    min_length = min(min(orig_length), min(syn_length))
    max_length = max(max(orig_length), max(syn_length))
    if np.isclose(min_length, max_length):
        return 0

    bins = np.linspace(min_length, max_length, bucket_num + 1)
    orig_count, _ = np.histogram(orig_length, bins=bins)
    syn_count, _ = np.histogram(syn_length, bins=bins)
    orig_count = orig_count.astype(float)
    syn_count = syn_count.astype(float)

    # Normalization
    orig_count /= (np.sum(orig_count) + 1e-10)
    syn_count /= (np.sum(syn_count) + 1e-10)

    return utils.js_divergence(orig_count, syn_count)


def get_trip_distribution(grid_db: List[List[Tuple[Grid, int]]], grid_map: GridMap):
    dist = np.zeros(grid_map.size * grid_map.size)

    for g_t in grid_db:
        if not g_t:
            continue
        start = g_t[0][0].linear_index
        end = g_t[-1][0].linear_index
        index = start * grid_map.size + end
        dist[index] += 1

    return dist


def calculate_trip_error(orig_db: List[List[Tuple[Grid, int]]],
                         syn_db: List[List[Tuple[Grid, int]]],
                         grid_map: GridMap):
    orig_trip = get_trip_distribution(orig_db, grid_map)
    syn_trip = get_trip_distribution(syn_db, grid_map)
    if orig_trip.sum() <= 0 or syn_trip.sum() <= 0:
        return 0

    orig_trip /= orig_trip.sum()
    syn_trip /= syn_trip.sum()

    return utils.js_divergence(orig_trip, syn_trip)


def adaptive_radius_from_grid(grid_n: int, base_grid_num=6, base_radius=1, max_radius=3):
    scaled_radius = math.ceil(grid_n * base_radius / max(base_grid_num, 1))
    return max(base_radius, min(max_radius, int(scaled_radius)))


def calculate_physical_violation(grid_db: List[List[Tuple[Grid, int]]],
                                 grid_map: GridMap,
                                 base_grid_num=6,
                                 base_radius=1,
                                 max_radius=3):
    radius = max(1.0, float(adaptive_radius_from_grid(grid_map.n, base_grid_num, base_radius, max_radius)))
    violation_sum = 0.0
    step_count = 0

    for traj in grid_db:
        if len(traj) < 2:
            continue

        steps = []
        for i in range(len(traj) - 1):
            g1 = traj[i][0]
            g2 = traj[i + 1][0]
            steps.append(np.asarray([g2.index[0] - g1.index[0], g2.index[1] - g1.index[1]], dtype=float))

        prev_step = np.zeros(2, dtype=float)
        prev_prev_step = np.zeros(2, dtype=float)
        for step in steps:
            speed = float(np.sqrt(np.sum(step * step)))
            prev_speed = float(np.sqrt(np.sum(prev_step * prev_step)))
            prev_accel = prev_step - prev_prev_step
            accel = step - prev_step
            jerk = accel - prev_accel

            speed_dev = abs(speed - prev_speed) / radius
            if speed > 1e-8 and prev_speed > 1e-8:
                cosine = np.clip(float(np.dot(step, prev_step) / max(speed * prev_speed, 1e-8)), -1.0, 1.0)
                turn_penalty = 0.5 * (1.0 - cosine)
            else:
                turn_penalty = 0.0
            accel_penalty = 0.7 * (float(np.sqrt(np.sum(accel * accel))) / max(radius, prev_speed + 1.0))
            accel_penalty += 0.3 * (float(np.sqrt(np.sum(jerk * jerk))) / max(radius, prev_speed + 1.0))

            violation_sum += 0.45 * speed_dev + 0.75 * turn_penalty + 0.85 * accel_penalty
            step_count += 1
            prev_prev_step = prev_step
            prev_step = step

    if step_count <= 0:
        return 0.0
    return violation_sum / step_count
