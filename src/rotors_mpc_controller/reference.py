"""Reference trajectory utilities for the NMPC controller."""

from __future__ import annotations

import math
import threading
from typing import Dict

import numpy as np


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _yaw_from_quat(quaternion: np.ndarray) -> float:
    quat = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return 0.0
    quat = quat / norm
    w, x, y, z = quat
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


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

        transition_cfg = config.get('transition', {})
        self._transition_enabled = bool(transition_cfg.get('enabled', True))
        self._transition_dt = float(transition_cfg.get('dt', 0.05))
        self._transition_max_speed = max(float(transition_cfg.get('max_speed', 1.0)), 1e-3)
        self._transition_max_acc = max(float(transition_cfg.get('max_acceleration', 0.75)), 1e-3)
        self._transition_distance_eps = float(transition_cfg.get('distance_epsilon', 0.02))
        self._transition_yaw_mode = str(transition_cfg.get('yaw_mode', 'interpolate')).lower()
        self._transition_max_yaw_rate = max(float(transition_cfg.get('max_yaw_rate', 1.0)), 1e-3)
        self._transition_min_duration = max(float(transition_cfg.get('min_duration', 0.0)), 0.0)

    def sync_state(self,
                   position: np.ndarray,
                   velocity: np.ndarray | None = None,
                   yaw: float | None = None,
                   quaternion: np.ndarray | None = None,
                   body_rates: np.ndarray | None = None,
                   thrust: np.ndarray | None = None,
                   *,
                   reset_trajectory: bool = False) -> None:
        """Synchronise the generator with an observed vehicle state without triggering new transitions."""
        pos_arr = np.asarray(position, dtype=float).reshape(3)
        vel_arr = np.asarray(velocity, dtype=float).reshape(3) if velocity is not None else None
        quat_arr = np.asarray(quaternion, dtype=float).reshape(4) if quaternion is not None else None
        rate_arr = np.asarray(body_rates, dtype=float).reshape(3) if body_rates is not None else None
        thrust_arr = np.asarray(thrust, dtype=float).reshape(-1) if thrust is not None else None

        with self._lock:
            self._position = pos_arr
            if vel_arr is not None:
                self._velocity = vel_arr
            if quat_arr is not None:
                norm = np.linalg.norm(quat_arr)
                self._quaternion = quat_arr / norm if norm > 0.0 else _quat_from_yaw(0.0)
                self._yaw = float(yaw) if yaw is not None else _yaw_from_quat(self._quaternion)
            elif yaw is not None:
                self._yaw = float(yaw)
                self._quaternion = _quat_from_yaw(self._yaw)
            if rate_arr is not None:
                self._body_rates = rate_arr
            if thrust_arr is not None and thrust_arr.size == 4:
                self._thrust = thrust_arr.astype(float, copy=True)
            if reset_trajectory:
                self._trajectory = None
                self._traj_index = 0
                self._traj_loop = False

    def _reference_snapshot_locked(self) -> Dict[str, np.ndarray]:
        if self._trajectory is not None:
            traj = self._trajectory
            if isinstance(traj, dict) and traj.get('positions') is not None:
                positions = np.asarray(traj['positions'], dtype=float)
                if positions.size > 0:
                    idx = int(np.clip(self._traj_index, 0, positions.shape[0] - 1))
                    pos = positions[idx].copy()
                    vel = np.asarray(traj['velocities'], dtype=float)[idx].copy()
                    quat = np.asarray(traj['quaternions'], dtype=float)[idx].copy()
                    rates = np.asarray(traj['body_rates'], dtype=float)[idx].copy()
                    thrusts_arr = traj.get('thrusts')
                    thrust: np.ndarray
                    if isinstance(thrusts_arr, np.ndarray) and thrusts_arr.size > 0:
                        thrust_idx = int(np.clip(idx, 0, thrusts_arr.shape[0] - 1))
                        thrust = thrusts_arr[thrust_idx].copy()
                    elif thrusts_arr is not None:
                        thrusts_np = np.asarray(thrusts_arr, dtype=float)
                        if thrusts_np.size > 0:
                            thrust_idx = int(np.clip(idx, 0, thrusts_np.shape[0] - 1))
                            thrust = thrusts_np[thrust_idx].copy()
                        else:
                            thrust = self._thrust.copy()
                    else:
                        thrust = self._thrust.copy()
                    yaws_arr = traj.get('yaws')
                    if isinstance(yaws_arr, np.ndarray) and yaws_arr.size > 0:
                        yaw_val = float(yaws_arr[idx])
                    elif yaws_arr is not None:
                        yaw_val = float(np.asarray(yaws_arr, dtype=float)[idx])
                    else:
                        yaw_val = _yaw_from_quat(quat)
                    return {
                        'position': pos,
                        'velocity': vel,
                        'quaternion': quat,
                        'body_rates': rates,
                        'thrust': thrust,
                        'yaw': yaw_val,
                    }
        return {
            'position': self._position.copy(),
            'velocity': self._velocity.copy(),
            'quaternion': self._quaternion.copy(),
            'body_rates': self._body_rates.copy(),
            'thrust': self._thrust.copy(),
            'yaw': float(self._yaw),
        }

    def _plan_transition(self,
                         start: Dict[str, np.ndarray],
                         target: Dict[str, np.ndarray]) -> Dict[str, object] | None:
        if not self._transition_enabled:
            return None

        dt = max(self._transition_dt, 1e-3)
        delta = target['position'] - start['position']
        distance = float(np.linalg.norm(delta))
        direction = delta / distance if distance > 1e-6 else np.zeros(3, dtype=float)

        yaw_start = float(start['yaw'])
        yaw_goal = float(target['yaw'])
        yaw_diff = math.atan2(math.sin(yaw_goal - yaw_start), math.cos(yaw_goal - yaw_start))

        if distance <= self._transition_distance_eps and abs(yaw_diff) <= 1e-3:
            return None

        v_max = self._transition_max_speed
        a_max = self._transition_max_acc
        distance_acc = v_max * v_max / (2.0 * a_max)

        if distance <= self._transition_distance_eps:
            t_acc = 0.0
            t_cruise = 0.0
            t_total_pos = 0.0
            v_peak = 0.0
        elif distance <= 2.0 * distance_acc:
            v_peak = math.sqrt(max(distance * a_max, 0.0))
            t_acc = v_peak / a_max
            t_cruise = 0.0
            t_total_pos = 2.0 * t_acc
        else:
            v_peak = v_max
            t_acc = v_peak / a_max
            t_cruise = max((distance - 2.0 * distance_acc) / v_peak, 0.0)
            t_total_pos = 2.0 * t_acc + t_cruise

        t_total_yaw = abs(yaw_diff) / self._transition_max_yaw_rate if self._transition_max_yaw_rate > 0.0 else 0.0
        t_total = max(t_total_pos, t_total_yaw, self._transition_min_duration)
        if t_total <= 0.0:
            t_total = max(dt, t_total_pos, t_total_yaw)

        steps = max(2, int(math.ceil(t_total / dt)) + 1)
        times = np.linspace(0.0, t_total, steps, dtype=float)

        positions = np.zeros((steps, 3), dtype=float)
        velocities = np.zeros((steps, 3), dtype=float)
        d_acc = 0.5 * a_max * t_acc * t_acc
        t_total_pos = max(t_total_pos, 0.0)

        for idx, t_val in enumerate(times):
            motion_t = min(t_val, t_total_pos)
            if distance <= self._transition_distance_eps:
                s = 0.0
                v_mag = 0.0
            elif motion_t <= t_acc:
                s = 0.5 * a_max * motion_t * motion_t
                v_mag = a_max * motion_t
            elif motion_t <= t_acc + t_cruise:
                s = d_acc + v_peak * (motion_t - t_acc)
                v_mag = v_peak
            else:
                remaining = max(t_total_pos - motion_t, 0.0)
                s = distance - 0.5 * a_max * remaining * remaining
                v_mag = a_max * remaining
            s = min(s, distance)
            positions[idx] = start['position'] + direction * s
            if distance > 1e-6:
                velocities[idx] = direction * v_mag
            else:
                velocities[idx] = start['velocity']

        positions[0] = start['position']
        positions[-1] = target['position']
        velocities[0] = start['velocity']
        velocities[-1] = target['velocity']

        yaw_values = np.full(steps, yaw_start, dtype=float)
        if abs(yaw_diff) > 1e-6:
            if self._transition_yaw_mode == 'interpolate':
                duration = max(t_total, 1e-6)
                ratios = np.clip(times / duration, 0.0, 1.0)
                yaw_values = yaw_start + ratios * yaw_diff
            else:
                yaw_values[-1] = yaw_start + yaw_diff
        yaw_values[-1] = yaw_goal

        quaternions = np.vstack([_quat_from_yaw(val) for val in yaw_values])
        body_rates = np.zeros((steps, 3), dtype=float)
        if steps > 1 and t_total > 0.0:
            dt_segments = np.diff(times)
            yaw_deltas = np.diff(yaw_values)
            rates = np.zeros(steps - 1, dtype=float)
            valid = dt_segments > 1e-6
            rates[valid] = yaw_deltas[valid] / dt_segments[valid]
            if np.any(~valid):
                rates[~valid] = 0.0
            body_rates[:-1, 2] = rates
            body_rates[-1, 2] = rates[-1] if rates.size else 0.0
        body_rates[0] = start['body_rates']
        body_rates[-1] = target['body_rates']

        thrusts = np.tile(target['thrust'], (steps, 1))
        thrusts[0] = start['thrust']
        thrusts[-1] = target['thrust']

        return {
            'positions': positions,
            'dt': dt,
            'velocities': velocities,
            'yaw': yaw_values,
            'quaternions': quaternions,
            'body_rates': body_rates,
            'thrusts': thrusts,
            'loop': False,
        }

    def get_active_trajectory(self) -> Dict[str, np.ndarray] | None:
        with self._lock:
            if self._trajectory is None:
                return None
            return {key: np.asarray(value, dtype=float).copy()
                    for key, value in self._trajectory.items()}

    def set_target(self,
                   position: np.ndarray,
                   velocity: np.ndarray | None = None,
                   yaw: float | None = None,
                   quaternion: np.ndarray | None = None,
                   body_rates: np.ndarray | None = None,
                   thrust: np.ndarray | None = None) -> None:
        position_arr = np.asarray(position, dtype=float).reshape(3)
        velocity_arr = np.asarray(velocity, dtype=float).reshape(3) if velocity is not None else None
        body_rates_arr = np.asarray(body_rates, dtype=float).reshape(3) if body_rates is not None else None
        thrust_arr = np.asarray(thrust, dtype=float).reshape(-1) if thrust is not None else None
        quaternion_arr = np.asarray(quaternion, dtype=float).reshape(4) if quaternion is not None else None

        with self._lock:
            start_state = self._reference_snapshot_locked()

        target_velocity = velocity_arr if velocity_arr is not None else np.zeros(3, dtype=float)
        target_body_rates = body_rates_arr if body_rates_arr is not None else np.zeros(3, dtype=float)
        if thrust_arr is not None:
            if thrust_arr.shape[0] != 4:
                raise ValueError('Thrust reference must have four components.')
            target_thrust = thrust_arr.astype(float, copy=True)
        else:
            target_thrust = start_state['thrust'].copy()

        if quaternion_arr is not None:
            norm = np.linalg.norm(quaternion_arr)
            if norm > 0.0:
                quaternion_arr = quaternion_arr / norm
            target_quat = quaternion_arr
            target_yaw = float(yaw) if yaw is not None else _yaw_from_quat(target_quat)
        else:
            target_yaw = float(yaw) if yaw is not None else float(start_state['yaw'])
            target_quat = _quat_from_yaw(target_yaw)

        target_state = {
            'position': position_arr,
            'velocity': target_velocity,
            'quaternion': target_quat,
            'body_rates': target_body_rates,
            'thrust': target_thrust,
            'yaw': target_yaw,
        }

        plan = self._plan_transition(start_state, target_state)
        if plan is not None:
            self.set_trajectory(**plan)
            return

        with self._lock:
            self._position = position_arr
            self._velocity = target_velocity
            self._yaw = target_yaw
            self._quaternion = target_quat
            self._body_rates = target_body_rates
            self._thrust = target_thrust
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
