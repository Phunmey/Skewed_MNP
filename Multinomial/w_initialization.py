import numpy as np


def _initialize_W(self):
    W = np.random.randn(self.N, self.J_minus_1)

    for i in range(self.N):
        if not (1 <= self.y[i] <= self.J):
            raise ValueError(f"Invalid y[{i}] = {self.y[i]}, must be in [1, {self.J}]")

        if self.y[i] == 1:
            W[i] = -np.abs(W[i]) - self.sep
        else:
            chosen = self.y[i] - 2
            W[i, chosen] = np.abs(W[i, chosen]) + 0.5
            for j in range(self.J_minus_1):
                if j != chosen:
                    W[i, j] = W[i, chosen] - 0.5 - np.abs(np.random.randn() * 0.3)

    return W
