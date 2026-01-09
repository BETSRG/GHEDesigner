import os
import pickle

import numpy as np


class ResponseInterpolator:
    def __init__(self, filename="interpolator_table_huge.pkl"):
        """
        Loads the interpolation table and prepares axes for fast lookup.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Could not find interpolation table: {filename}")

        print(f"Loading Physics Table: {filename}...")
        with open(filename, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self.table = data['table']  # Dictionary: {(depth, spacing, beta): function}
        self.axes = data['axes']  # List: [depths, spacings, betas]

        # Ensure axes are sorted for binary search
        self.depths = np.sort(self.axes[0])
        self.spacings = np.sort(self.axes[1])
        self.betas = np.sort(self.axes[2])

        print("  > Table Loaded Successfully.")

    def _get_neighbors(self, value, axis):
        """
        Finds the two bounding grid points (lower, upper) and the relative weight.
        Returns: (val_lower, val_upper, fraction)
        """
        # 1. Handle Out of Bounds (Clamp to edges)
        if value <= axis[0]:
            return axis[0], axis[0], 0.0
        if value >= axis[-1]:
            return axis[-1], axis[-1], 0.0

        # 2. Find insertion index (Binary Search)
        idx = np.searchsorted(axis, value)

        # 3. Get Neighbors
        val_lower = axis[idx - 1]
        val_upper = axis[idx]

        # 4. Calculate Fraction (Distance from lower neighbor)
        fraction = (value - val_lower) / (val_upper - val_lower)

        return val_lower, val_upper, fraction

    def interpolate(self, depth, spacing, beta, time_values):
        """
        Generates a custom q' response curve for the specific geometry
        using 3D (Trilinear) interpolation.
        """
        # 1. Find Neighbors & Weights for all 3 Dimensions
        d_low, d_high, fd = self._get_neighbors(depth, self.depths)
        s_low, s_high, fs = self._get_neighbors(spacing, self.spacings)
        b_low, b_high, fb = self._get_neighbors(beta, self.betas)

        # 2. Retrieve the 8 Corner Curves (The "Cube")
        corners = {}

        try:
            for d in [d_low, d_high]:
                for s in [s_low, s_high]:
                    for b in [b_low, b_high]:
                        key = (d, s, b)
                        if key not in self.table:
                            raise ValueError(f"Grid point missing in table: {key}")
                        corners[key] = self.table[key](time_values)

        except KeyError as e:
            print(f"Interpolation Error: The table is sparse or missing the grid point {e}")
            return np.zeros_like(time_values)

        # 3. Trilinear Interpolation Math
        # Step A: Interpolate along Beta (Collapse 8 points -> 4 points)
        c00 = corners[(d_low, s_low, b_low)] * (1 - fb) + corners[(d_low, s_low, b_high)] * fb
        c01 = corners[(d_low, s_high, b_low)] * (1 - fb) + corners[(d_low, s_high, b_high)] * fb
        c10 = corners[(d_high, s_low, b_low)] * (1 - fb) + corners[(d_high, s_low, b_high)] * fb
        c11 = corners[(d_high, s_high, b_low)] * (1 - fb) + corners[(d_high, s_high, b_high)] * fb

        # Step B: Interpolate along Spacing (Collapse 4 points -> 2 points)
        c0 = c00 * (1 - fs) + c01 * fs
        c1 = c10 * (1 - fs) + c11 * fs

        # Step C: Interpolate along Depth (Collapse 2 points -> 1 Final Curve)
        final_curve = c0 * (1 - fd) + c1 * fd

        return final_curve
