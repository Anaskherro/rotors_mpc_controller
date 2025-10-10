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
        self._trajectory: Dict[str, np.ndarray] | None = None
        self._traj_loop = False
        self._traj_index = 0

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
            self._trajectory = None
            self._traj_index = 0
            self._traj_loop = False

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
            self._trajectory = None
            self._traj_index = 0
            self._traj_loop = False

    def update_hover_thrust(self, thrust_per_motor: float) -> None:
        with self._lock:
            self._thrust = np.full(4, float(thrust_per_motor), dtype=float)

    def clear_trajectory(self) -> None:
        with self._lock:
            self._trajectory = None
            self._traj_index = 0
            self._traj_loop = False

    def set_trajectory(self,
                       positions: np.ndarray,
                       dt: float,
                       velocities: np.ndarray | None = None,
                       yaw: np.ndarray | float | None = None,
                       quaternions: np.ndarray | None = None,
                       body_rates: np.ndarray | None = None,
                       thrusts: np.ndarray | None = None,
                       loop: bool = False) -> None:
        if dt <= 0.0:
            raise ValueError('Trajectory dt must be positive.')

        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError('positions must have shape (N, 3).')
        if pos.shape[0] < 1:
            raise ValueError('positions must contain at least one sample.')
        steps = pos.shape[0]

        vel: np.ndarray
        if velocities is not None:
            vel = np.asarray(velocities, dtype=float)
            if vel.shape != pos.shape:
                raise ValueError('velocities must match positions shape.')
        else:
            vel = np.zeros_like(pos)
            if steps > 1:
                vel[:-1] = (pos[1:] - pos[:-1]) / dt
                vel[-1] = vel[-2]

        yaw_array: np.ndarray
        quat: np.ndarray
        if quaternions is not None:
            quat = np.asarray(quaternions, dtype=float)
            if quat.shape != (steps, 4):
                raise ValueError('quaternions must have shape (N, 4).')
            norms = np.linalg.norm(quat, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            quat = quat / norms
            yaw_array = np.unwrap(2.0 * np.arctan2(quat[:, 3], quat[:, 0]))
        else:
            if yaw is None:
                yaw_array = np.zeros(steps, dtype=float)
                if steps > 1:
                    yaw_array[:-1] = np.arctan2(vel[:-1, 1], vel[:-1, 0])
                    yaw_array[-1] = yaw_array[-2]
            else:
                yaw_array = np.asarray(yaw, dtype=float).reshape(-1)
                if yaw_array.size == 1:
                    yaw_array = np.full(steps, yaw_array[0], dtype=float)
                elif yaw_array.size != steps:
                    raise ValueError('yaw must be scalar or length N.')
            quat = np.vstack([_quat_from_yaw(val) for val in yaw_array])

        if body_rates is not None:
            rates = np.asarray(body_rates, dtype=float)
            if rates.shape != (steps, 3):
                raise ValueError('body_rates must have shape (N, 3).')
        else:
            rates = np.zeros((steps, 3), dtype=float)
            if steps > 1:
                yaw_rate = np.zeros(steps, dtype=float)
                yaw_rate[:-1] = (yaw_array[1:] - yaw_array[:-1]) / dt
                yaw_rate[-1] = yaw_rate[-2]
                rates[:, 2] = yaw_rate

        if thrusts is not None:
            thrust_arr = np.asarray(thrusts, dtype=float)
            if thrust_arr.shape not in {(steps, 4), (steps - 1, 4)}:
                raise ValueError('thrusts must have shape (N, 4) or (N-1, 4).')
            if thrust_arr.shape[0] == steps - 1:
                if steps == 1:
                    thrust_arr = np.tile(self._thrust, (1, 1))
                else:
                    thrust_arr = np.vstack([thrust_arr, thrust_arr[-1]])
        else:
            thrust_arr = np.tile(self._thrust, (steps, 1))

        with self._lock:
            self._trajectory = {
                'positions': pos,
                'velocities': vel,
                'quaternions': quat,
                'body_rates': rates,
                'thrusts': thrust_arr,
                'yaws': yaw_array,
            }
            self._traj_loop = bool(loop)
            self._traj_index = 0
            self._position = pos[0]
            self._velocity = vel[0]
            self._quaternion = quat[0]
            self._body_rates = rates[0]
            self._yaw = yaw_array[0]

    def build_horizon(self, horizon: int, dt: float) -> Dict[str, np.ndarray]:
        with self._lock:
            if self._trajectory is not None:
                traj = self._trajectory
                length = traj['positions'].shape[0]
                if self._traj_loop and length > 0:
                    indices = (self._traj_index + np.arange(horizon + 1)) % length
                    thrust_idx = (self._traj_index + np.arange(horizon)) % length
                else:
                    indices = np.clip(self._traj_index + np.arange(horizon + 1), 0, length - 1)
                    thrust_idx = np.clip(self._traj_index + np.arange(horizon), 0, length - 1)

                positions = traj['positions'][indices]
                velocities = traj['velocities'][indices]
                quaternions = traj['quaternions'][indices]
                rates = traj['body_rates'][indices]
                thrusts = traj['thrusts'][thrust_idx] if horizon > 0 else np.zeros((0, 4), dtype=float)
                yaws = traj['yaws'][indices]

                if length > 1:
                    next_index = self._traj_index + 1
                    if self._traj_loop:
                        self._traj_index = next_index % length
                    else:
                        self._traj_index = min(next_index, length - 1)

                return {
                    'positions': positions,
                    'velocities': velocities,
                    'quaternions': quaternions,
                    'body_rates': rates,
                    'thrusts': thrusts,
                    'yaws': yaws,
                }

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
