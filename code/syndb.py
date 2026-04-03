import random

import numpy as np
from grid import Grid, GridMap
import utils
from typing import List, Tuple


class SynDB:
    def __init__(self):
        # All history data
        self.history_data: List[List[Tuple[Grid, int]]] = []
        # Trajectories that haven't terminated
        self.current_data: List[List[Tuple[Grid, int]]] = []
        # Current timestamp
        self.t = -1

    def generate_new_points(self,
                            markov_mat: np.ndarray,
                            grid_map: GridMap,
                            avg_len: float):
        self.t += 1
        for traj in self.current_data:
            prev_grid = traj[-1][0]
            row = prev_grid.linear_index
            candidates = grid_map.get_candidate_linear(prev_grid)
            candidate_prob = np.nan_to_num(markov_mat[row, candidates].copy(), nan=0.0)

            # Quit probability
            quit_prob = markov_mat[row, -1] * min(1.0, len(traj) / avg_len)
            candidate_prob = np.append(candidate_prob, quit_prob)

            if candidate_prob.sum() < 0.00001:
                traj.append((prev_grid, self.t))
            else:
                candidate_prob = candidate_prob / candidate_prob.sum()
                sample_id = np.random.choice(len(candidate_prob), p=candidate_prob)

                if sample_id == len(candidate_prob) - 1:
                    # Quitting
                    continue
                traj.append((grid_map.get_grid_by_linear(int(candidates[sample_id])), self.t))

        # Move terminated trajectories to history data
        new_curr_data = []
        for traj in self.current_data:
            if traj[-1][1] == self.t:
                new_curr_data.append(traj)
            else:
                self.history_data.append(traj)
        self.current_data = new_curr_data

    def generate_new_points_baseline(self,
                                     markov_mat: np.ndarray,
                                     grid_map: GridMap):
        """
        For baseline, without considering quitting events
        """
        self.t += 1
        for traj in self.current_data:
            prev_grid = traj[-1][0]
            row = prev_grid.linear_index
            candidates = grid_map.get_candidate_linear(prev_grid)
            candidate_prob = np.nan_to_num(markov_mat[row, candidates].copy(), nan=0.0)

            if candidate_prob.sum() < 0.00001:
                sample_id = np.random.choice(len(candidates))
                traj.append((grid_map.get_grid_by_linear(int(candidates[sample_id])), self.t))
            else:
                candidate_prob = candidate_prob / candidate_prob.sum()
                sample_id = np.random.choice(len(candidate_prob), p=candidate_prob)
                traj.append((grid_map.get_grid_by_linear(int(candidates[sample_id])), self.t))

    def adjust_data_size(self,
                         markov_mat: np.ndarray,
                         target_n: int,
                         grid_map: GridMap,
                         quit_distribution: np.ndarray):
        if self.n < target_n:
            missing = target_n - self.n
            enter_prob = markov_mat[-1, :-1]
            total_prob = enter_prob.sum()
            if total_prob < 1e-8:
                sampled = np.random.choice(grid_map.size, size=missing)
            else:
                sampled = np.random.choice(grid_map.size, size=missing, p=enter_prob / total_prob)
            self.current_data.extend(
                [[(grid_map.get_grid_by_linear(int(sample_id)), self.t)] for sample_id in sampled]
            )

        if self.n > target_n:
            if np.sum(quit_distribution) < 1e-5:
                random.shuffle(self.current_data)
                sample_data = self.current_data[target_n:]

                for idx, traj in enumerate(sample_data):
                    sample_data[idx] = traj[:-1]
                self.history_data.extend(sample_data)
                self.current_data = self.current_data[:target_n]
            else:
                # Sampling based on quitting distribution
                prob = np.zeros(self.n)
                for i in range(self.n):
                    row = self.current_data[i][-2][0].linear_index
                    prob[i] = quit_distribution[row]
                prob += 1e-8
                prob = prob / prob.sum()
                sample_id = np.random.choice(self.n, size=self.n - target_n, replace=False, p=prob)
                keep_mask = np.ones(self.n, dtype=bool)
                keep_mask[sample_id] = False
                new_history_add = [self.current_data[i] for i in sample_id]

                for idx, traj in enumerate(new_history_add):
                    new_history_add[idx] = traj[:-1]
                self.history_data.extend(new_history_add)
                new_curr_data = [self.current_data[i] for i in np.flatnonzero(keep_mask)]
                self.current_data = new_curr_data

    def random_initialize(self,
                          target_n: int,
                          grid_map: GridMap):
        self.t = 0
        sampled = np.random.choice(grid_map.size, size=target_n)
        self.current_data = [[(grid_map.get_grid_by_linear(int(sample_id)), self.t)] for sample_id in sampled]

    @property
    def n(self):
        return len(self.current_data)

    @property
    def all_data(self):
        d = self.history_data.copy()
        d.extend(self.current_data)
        return d


class Users:
    """
    User status:
    1: active(available), 0: inactive(not recycled), 2: sampled for reporting, -1: quitted
    """

    def __init__(self):
        self.users = {}

    def register(self, uid):
        try:
            self.users[uid]
        except KeyError:
            self.users[uid] = 1

    def sample(self, p):
        available_users = self.available_users
        sampled_users = random.sample(available_users, int(p * len(available_users)))
        for uid in sampled_users:
            self.users[uid] = 2
        return sampled_users

    def deactivate(self, uid):
        self.users[uid] = 0

    def remove(self, uid):
        self.users[uid] = -1

    def recycle(self, uid):
        if self.users[uid] != -1:
            self.users[uid] = 1

    @property
    def available_users(self):
        a_u = []
        for (uid, state) in self.users.items():
            if state == 1:
                a_u.append(uid)
        return a_u
