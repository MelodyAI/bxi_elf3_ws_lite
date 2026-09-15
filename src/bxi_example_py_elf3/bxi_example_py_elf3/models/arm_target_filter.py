"""Continuous arm targets with velocity and acceleration bounds (radians)."""

import numpy as np
from copy import copy


class ArmTargetFilter:
    def __init__(self, indices, velocity=3.0, acceleration=20.0, frequency=12.0):
        self.indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray([velocity, acceleration, frequency], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Arm filter velocity/acceleration/frequency must be positive")
        self.max_velocity, self.max_acceleration, self.frequency = values
        self.reset()

    def reset(self):
        self.position = None
        self.velocity = np.zeros(len(self.indices), dtype=np.float32)
        self.limited_steps = 0

    def apply(self, target, measured_position, dt):
        if not np.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Arm filter dt must be in (0, 0.1]")
        result = np.asarray(target, dtype=np.float32).copy()
        if self.position is None:
            self.position = np.asarray(measured_position, dtype=np.float32)[self.indices].copy()
        error = result[self.indices] - self.position
        # Critically damped second-order response. Do not reset velocity on a
        # reversal: doing so creates precisely the acceleration impulse we avoid.
        acceleration = self.frequency**2 * error - 2 * self.frequency * self.velocity
        acceleration = np.clip(acceleration, -self.max_acceleration, self.max_acceleration)
        velocity = np.clip(self.velocity + acceleration * dt, -self.max_velocity, self.max_velocity)
        self.position += velocity * dt
        self.velocity = velocity
        self.limited_steps += int(np.any(np.abs(result[self.indices] - self.position) > 1e-3))
        result[self.indices] = self.position
        return result


class ArmReferenceFilter(ArmTargetFilter):
    """Filter source-time centers; roll out futures without advancing state."""

    def window(self, positions, velocities, source_dt):
        p = np.asarray(positions, dtype=np.float32).copy()
        v = np.asarray(velocities, dtype=np.float32).copy()
        if p.shape != (11, 29) or v.shape != p.shape:
            raise ValueError("Expected [11,29] reference position and velocity")
        # Initialize from the first reference. Robot entry is blended separately.
        if self.position is None:
            self.position = p[0, self.indices].copy()
        dt = min(max(float(source_dt), 1e-6), 0.25)
        steps = max(1, int(np.ceil(dt / 0.02)))
        current = p[0].copy()
        for _ in range(steps):
            current = self.apply(p[0], p[0], dt / steps)
        p[0] = current
        v[0, self.indices] = self.velocity
        future = copy(self)
        future.position = self.position.copy()
        future.velocity = self.velocity.copy()
        for i in range(1, len(p)):
            p[i] = future.apply(p[i], p[i], 0.02)
            v[i, self.indices] = future.velocity
        return p, v
