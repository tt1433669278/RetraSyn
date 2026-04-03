import numpy as np
import math


class OUE:
    def __init__(self, epsilon, d, map_func=None):
        self.epsilon = epsilon
        self.d = d
        self.map_func = map_func

        # Sum of updated data (real counts)
        self.aggregated_data = np.zeros(self.d, dtype=int)
        # Unbiased adjustment from aggregated data
        self.adjusted_data = np.zeros(self.d, dtype=int)

        # Number of users
        self.n = 0

        # Probability of 1=>1
        self.p = 0.5
        # Probability of 0=>1
        self.q = 1 / (math.pow(math.e, self.epsilon) + 1)

    def _aggregate(self, index):
        self.aggregated_data[index] += 1
        self.n += 1

    def privatise(self, data):
        index = self.map_func(data)
        self._aggregate(index)

    def privatise_index(self, index):
        self._aggregate(index)

    def aggregate_indices(self, indices):
        if len(indices) == 0:
            return
        self.aggregated_data += np.bincount(indices, minlength=self.d)
        self.n += len(indices)

    def adjust(self):
        # If y=0, Prob(y'=1)=q, Prob(y'=0)=1-q
        tmp_perturbed_1 = np.copy(self.aggregated_data)
        est_count = np.random.binomial(tmp_perturbed_1, self.p)

        # If y=1, Prob(y'=0)=p
        tmp_perturbed_0 = self.n - np.copy(self.aggregated_data)
        est_count += np.random.binomial(tmp_perturbed_0, self.q)

        # Unbiased adjustment
        self.adjusted_data = (est_count - self.n * self.q)/(self.p-self.q)

    @property
    def non_negative_data(self):
        return np.maximum(self.adjusted_data, 0)
