"""Exact center row of the rolling LQR filter, cached for each window shape."""
from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@lru_cache(maxsize=128)
def smoothing_row(frames, index, data_cost, velocity_cost, acceleration_cost,
                  jerk_cost, anchor_start, anchor_end):
    data_cost = max(float(data_cost), 1e-12)
    costs = [max(float(v), 0.) for v in (velocity_cost, acceleration_cost, jerk_cost)]
    system = data_cost * np.eye(frames)
    for order, cost in enumerate(costs, start=1):
        if cost > 0. and frames > order:
            difference = np.diff(np.eye(frames), n=order, axis=0)
            system += cost * (difference.T @ difference)
    fixed = sorted(set(([0] if anchor_start else []) + ([frames - 1] if anchor_end else [])))
    row = np.zeros(frames)
    if index in fixed:
        row[index] = 1.
    else:
        free = np.asarray([i for i in range(frames) if i not in fixed])
        unit = np.zeros(len(free))
        unit[np.flatnonzero(free == index)[0]] = 1.
        inverse_row = cho_solve(cho_factor(system[np.ix_(free, free)], lower=True), unit)
        row[free] = data_cost * inverse_row
        if fixed:
            row[fixed] = -inverse_row @ system[np.ix_(free, fixed)]
    row.flags.writeable = False
    return row
