"""Reference trajectory utilities for the NMPC controller."""

from __future__ import annotations

import threading
from typing import Dict

import numpy as np


class ReferenceGenerator:
    """Generates constant references with optional smoothing."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.frame = config.get('frame', 'world')
        self._lock = threading.Lock()

        self._position = np.asarray(config.get('default_position', [0.0, 0.0, 1.0]), dtype=float)
        self._velocity = np.asarray(config.get('default_velocity', [0.0, 0.0, 0.0]), dtype=float)
        self._acceleration = np.asarray(config.get('default_acceleration', [0.0, 0.0, 0.0]),
                                        dtype=float)
        self._yaw = float(config.get('default_yaw', 0.0))

    def set_target(self,
                   position: np.ndarray,
                   velocity: np.ndarray | None = None,
                   acceleration: np.ndarray | None = None,
                   yaw: float | None = None) -> None:
        with self._lock:
            self._position = np.asarray(position, dtype=float).reshape(3)
            if velocity is not None:
                self._velocity = np.asarray(velocity, dtype=float).reshape(3)
            if acceleration is not None:
                self._acceleration = np.asarray(acceleration, dtype=float).reshape(3)
            if yaw is not None:
                self._yaw = float(yaw)

    def build_horizon(self, horizon: int, dt: float) -> Dict[str, np.ndarray]:
        with self._lock:
            pos = np.tile(self._position, (horizon + 1, 1))
            vel = np.tile(self._velocity, (horizon + 1, 1))
            acc = np.tile(self._acceleration, (horizon + 1, 1))
            yaw = np.full((horizon + 1,), self._yaw, dtype=float)

        return {
            'positions': pos,
            'velocities': vel,
            'accelerations': acc,
            'yaws': yaw,
        }
