"""Temperature scaling (Guo et al., 2017): p' = sigmoid(logit(p) / T), with T fit on validation NLL.
Dividing the logit keeps the symmetry f(1 - p) = 1 - f(p), so it never adds a position preference."""
import numpy as np
from scipy.optimize import minimize_scalar


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, p, y):
        z, y = _logit(p), np.asarray(y, float)

        def nll(log_t):
            q = np.clip(1 / (1 + np.exp(-z / np.exp(log_t))), 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def transform(self, p):
        return 1 / (1 + np.exp(-_logit(p) / self.T))
