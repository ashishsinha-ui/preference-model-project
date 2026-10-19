"""Isotonic regression calibration, fit symmetrically: both (p, y) and (1 - p, 1 - y) are used,
so the learned map keeps f(1 - p) = 1 - f(p) and does not introduce a position preference."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y, float)
        self.iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
        self.iso.fit(np.r_[p, 1 - p], np.r_[y, 1 - y])
        return self

    def transform(self, p):
        return self.iso.predict(np.asarray(p, float))
