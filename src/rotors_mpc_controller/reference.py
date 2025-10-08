"""Reference trajectory utilities for the NMPC controller."""

from __future__ import annotations

import threading
from typing import Dict

import numpy as np


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


class ReferenceGenerator:
    """Generates constant references with optional smoothing."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.frame = config.get('frame', 'world')
        self._lock = threading.Lock()

        self._position = np.asarray(config.get('default_position', [0.0, 0.0, 1.0]), dtype=float)
        self._velocity = np.asarray(config.get('default_velocity', [0.0, 0.0, 0.0]), dtype=float)
        self._yaw = float(config.get('default_yaw', 0.0))
        self._quaternion = _quat_from_yaw(self._yaw)
        self._body_rates = np.zeros(3, dtype=float)
        self._thrust = np.zeros(4, dtype=float)

    def set_target(self,
                   position: np.ndarray,
                   velocity: np.ndarray | None = None,
                   yaw: float | None = None,
                   quaternion: np.ndarray | None = None,
                   body_rates: np.ndarray | None = None,
                   thrust: np.ndarray | None = None) -> None:
        with self._lock:
            self._position = np.asarray(position, dtype=float).reshape(3)
            if velocity is not None:
                self._velocity = np.asarray(velocity, dtype=float).reshape(3)
            if quaternion is not None:
                self._quaternion = np.asarray(quaternion, dtype=float).reshape(4)
                norm = np.linalg.norm(self._quaternion)
                if norm != 0.0:
                    self._quaternion /= norm
                self._yaw = yaw if yaw is not None else self._yaw
            elif yaw is not None:
                self._yaw = float(yaw)
                self._quaternion = _quat_from_yaw(self._yaw)
            if body_rates is not None:
                self._body_rates = np.asarray(body_rates, dtype=float).reshape(3)
            if thrust is not None:
                self._thrust = np.asarray(thrust, dtype=float).reshape(-1)
                if self._thrust.shape[0] != 4:
                    raise ValueError('Thrust reference must have four components.')

    def update_defaults(self,
                        position: np.ndarray,
                        velocity: np.ndarray,
                        yaw: float,
                        frame: str | None = None) -> None:
        with self._lock:
            self._position = np.asarray(position, dtype=float).reshape(3)
            self._velocity = np.asarray(velocity, dtype=float).reshape(3)
            self._yaw = float(yaw)
            self._quaternion = _quat_from_yaw(self._yaw)
            self._body_rates = np.zeros(3, dtype=float)
            if frame is not None:
                self.frame = frame

    def update_hover_thrust(self, thrust_per_motor: float) -> None:
        with self._lock:
            self._thrust = np.full(4, float(thrust_per_motor), dtype=float)

    def build_horizon(self, horizon: int, dt: float) -> Dict[str, np.ndarray]:
        with self._lock:
            pos = np.tile(self._position, (horizon + 1, 1))
            vel = np.tile(self._velocity, (horizon + 1, 1))
            yaw = np.full((horizon + 1,), self._yaw, dtype=float)
            quat = np.tile(self._quaternion, (horizon + 1, 1))
            rates = np.tile(self._body_rates, (horizon + 1, 1))
            thrusts = np.tile(self._thrust, (horizon, 1))

        return {
            'positions': pos,
            'velocities': vel,
            'quaternions': quat,
            'body_rates': rates,
            'thrusts': thrusts,
            'yaws': yaw,
        }
