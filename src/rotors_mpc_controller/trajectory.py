"""Trajectory generation utilities inspired by data_driven_mpc."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .controller import PositionNMPC


@dataclass(frozen=True)
class _QuadModel:
    mass: float
    inertia: np.ndarray  # shape (3,)
    x_offsets: np.ndarray  # shape (4,)
    y_offsets: np.ndarray  # shape (4,)
    z_torque: np.ndarray  # shape (4,)
    max_thrust: float

    @property
    def allocation_matrix(self) -> np.ndarray:
        return np.vstack((
            self.y_offsets,
            -self.x_offsets,
            self.z_torque,
            np.ones_like(self.z_torque),
        ))


def generate_loop_reference(controller: PositionNMPC,
                            traj_cfg: Dict[str, object],
                            dt: float) -> Dict[str, np.ndarray]:
    """Generate a minimum-snap loop trajectory compatible with PositionNMPC."""

    quad = _QuadModel(
        mass=controller.mass,
        inertia=np.asarray(controller.config.inertia, dtype=float).reshape(3),
        x_offsets=np.asarray(controller.config.rotor_x_offsets, dtype=float).reshape(4),
        y_offsets=np.asarray(controller.config.rotor_y_offsets, dtype=float).reshape(4),
        z_torque=np.asarray(controller.config.rotor_z_torque, dtype=float).reshape(4),
        max_thrust=float(np.max(controller.config.input_upper_bounds)),
    )

    radius = float(traj_cfg.get('radius', 3.0))
    altitude = float(traj_cfg.get('altitude', 1.0))
    linear_acc = float(traj_cfg.get('linear_acceleration', 0.25))
    max_speed = float(traj_cfg.get('max_speed', traj_cfg.get('speed', 1.0)))
    clockwise = bool(traj_cfg.get('clockwise', False))
    yaw_mode = str(traj_cfg.get('yaw_mode', 'follow')).lower()
    yaw_offset = float(traj_cfg.get('yaw_offset', 0.0))
    yaw_constant = float(traj_cfg.get('yaw_constant', 0.0))
    center = np.asarray(traj_cfg.get('center', [0.0, 0.0]), dtype=float).reshape(2)

    profile = str(traj_cfg.get('profile', 'stop')).lower()
    revolutions = float(traj_cfg.get('revolutions', 1.0))

    if profile == 'continuous':
        traj_derivs, yaw_derivs, t_ref = _continuous_loop_trajectory(radius=radius,
                                                                     altitude=altitude,
                                                                     center=center,
                                                                     max_speed=max_speed,
                                                                     clockwise=clockwise,
                                                                     yaw_mode=yaw_mode,
                                                                     yaw_offset=yaw_offset,
                                                                     yaw_constant=yaw_constant,
                                                                     dt=dt,
                                                                     revolutions=revolutions)
    else:
        traj_derivs, yaw_derivs, t_ref = _loop_stop_trajectory(radius=radius,
                                                               altitude=altitude,
                                                               center=center,
                                                               linear_acc=linear_acc,
                                                               max_speed=max_speed,
                                                               clockwise=clockwise,
                                                               yaw_mode=yaw_mode,
                                                               yaw_offset=yaw_offset,
                                                               yaw_constant=yaw_constant,
                                                               dt=dt,
                                                               revolutions=revolutions)

    ref = _minimum_snap_reference(traj_derivs, yaw_derivs, t_ref, quad, gravity=controller.gravity)
    ref['dt'] = dt
    return ref


def _loop_stop_trajectory(radius: float,
                          altitude: float,
                          center: np.ndarray,
                          linear_acc: float,
                          max_speed: float,
                          clockwise: bool,
                          yaw_mode: str,
                          yaw_offset: float,
                          yaw_constant: float,
                          dt: float,
                          revolutions: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius <= 0.0:
        raise ValueError('radius must be positive.')
    if dt <= 0.0:
        raise ValueError('dt must be positive.')

    total_angle = 2.0 * np.pi * max(revolutions, 0.0)
    if total_angle <= 0.0:
        raise ValueError('revolutions must be positive.')

    linear_acc = max(linear_acc, 1e-3)
    max_speed = max(max_speed, 1e-3)

    angular_acc = linear_acc / radius
    omega_target = max_speed / radius
    omega_target = max(omega_target, 1e-3)

    angle_acc = 0.5 * omega_target ** 2 / angular_acc
    if total_angle <= 2.0 * angle_acc:
        omega_peak = np.sqrt(total_angle * angular_acc)
        angle_acc = total_angle * 0.5
        angle_const = 0.0
        omega_target = omega_peak
    else:
        angle_const = total_angle - 2.0 * angle_acc

    t_acc = omega_target / angular_acc
    t_const = angle_const / omega_target if omega_target > 0.0 else 0.0
    t_total = 2.0 * t_acc + t_const

    if t_total <= 0.0:
        raise ValueError('trajectory duration is zero; check parameters.')

    t_ref = np.arange(0.0, t_total + dt * 0.5, dt, dtype=float)
    omega_vals = np.zeros_like(t_ref)
    angle_vals = np.zeros_like(t_ref)

    for idx in range(1, t_ref.size):
        t_prev = t_ref[idx - 1]
        t_cur = t_ref[idx]
        if t_cur <= t_acc:
            omega = omega_vals[idx - 1] + angular_acc * dt
        elif t_cur <= t_acc + t_const:
            omega = omega_target
        elif t_cur <= t_total:
            omega = max(omega_vals[idx - 1] - angular_acc * dt, 0.0)
        else:
            omega = 0.0
        omega_vals[idx] = omega
        angle_vals[idx] = angle_vals[idx - 1] + omega * dt

    angle_vals[-1] = total_angle
    omega_vals[-1] = 0.0

    direction = -1.0 if clockwise else 1.0
    angle_vec = direction * angle_vals
    w_vec = direction * omega_vals

    if angle_vec.size < 2:
        raise ValueError('trajectory duration too short relative to dt.')

    alpha_vec = np.gradient(w_vec, dt, edge_order=2 if w_vec.size > 2 else 1)
    alpha_dt = np.gradient(alpha_vec, dt, edge_order=2 if alpha_vec.size > 2 else 1)

    sin_angle = np.sin(angle_vec)
    cos_angle = np.cos(angle_vec)

    pos_x = center[0] + radius * sin_angle
    pos_y = center[1] + radius * cos_angle
    pos_z = np.full_like(pos_x, altitude)

    vel_x = radius * w_vec * cos_angle
    vel_y = -radius * w_vec * sin_angle
    vel_z = np.zeros_like(vel_x)

    acc_x = radius * (alpha_vec * cos_angle - (w_vec ** 2) * sin_angle)
    acc_y = -radius * (alpha_vec * sin_angle + (w_vec ** 2) * cos_angle)
    acc_z = np.zeros_like(acc_x)

    jerk_x = radius * (alpha_dt * cos_angle - alpha_vec * sin_angle * w_vec
                       - cos_angle * (w_vec ** 3) - 2.0 * sin_angle * w_vec * alpha_vec)
    jerk_y = -radius * (cos_angle * w_vec * alpha_vec + sin_angle * alpha_dt
                        - sin_angle * (w_vec ** 3) + 2.0 * cos_angle * w_vec * alpha_vec)
    jerk_z = np.zeros_like(jerk_x)

    positions = np.vstack((pos_x, pos_y, pos_z))
    velocities = np.vstack((vel_x, vel_y, vel_z))
    accelerations = np.vstack((acc_x, acc_y, acc_z))
    jerks = np.vstack((jerk_x, jerk_y, jerk_z))

    if yaw_mode == 'follow':
        yaw_traj = -angle_vec + yaw_offset
        yaw_rate = w_vec
    else:
        yaw_traj = np.full_like(angle_vec, yaw_constant)
        yaw_rate = np.zeros_like(w_vec)

    traj_derivs = np.stack((positions, velocities, accelerations, jerks), axis=0)
    yaw_derivs = np.vstack((yaw_traj, yaw_rate))

    return traj_derivs, yaw_derivs, t_ref


def _continuous_loop_trajectory(radius: float,
                                altitude: float,
                                center: np.ndarray,
                                max_speed: float,
                                clockwise: bool,
                                yaw_mode: str,
                                yaw_offset: float,
                                yaw_constant: float,
                                dt: float,
                                revolutions: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius <= 0.0:
        raise ValueError('radius must be positive.')
    if dt <= 0.0:
        raise ValueError('dt must be positive.')

    total_angle = 2.0 * np.pi * max(revolutions, 0.0)
    if total_angle <= 0.0:
        raise ValueError('revolutions must be positive.')

    omega = max_speed / radius if radius > 0.0 else 0.0
    omega = max(omega, 1e-3)
    samples = max(32, int(np.ceil(total_angle / (abs(omega) * dt))))

    base_theta = np.linspace(0.0, total_angle, samples, endpoint=False)
    if clockwise:
        theta = -base_theta
        omega_vec = -np.full(samples, omega, dtype=float)
    else:
        theta = base_theta
        omega_vec = np.full(samples, omega, dtype=float)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    pos_x = center[0] + radius * sin_theta
    pos_y = center[1] + radius * cos_theta
    pos_z = np.full_like(pos_x, altitude)

    vel_x = radius * omega_vec * cos_theta
    vel_y = -radius * omega_vec * sin_theta
    vel_z = np.zeros_like(vel_x)

    acc_x = -radius * (omega_vec ** 2) * sin_theta
    acc_y = -radius * (omega_vec ** 2) * cos_theta
    acc_z = np.zeros_like(acc_x)

    jerk_x = -radius * (omega_vec ** 3) * cos_theta
    jerk_y = radius * (omega_vec ** 3) * sin_theta
    jerk_z = np.zeros_like(jerk_x)

    positions = np.vstack((pos_x, pos_y, pos_z))
    velocities = np.vstack((vel_x, vel_y, vel_z))
    accelerations = np.vstack((acc_x, acc_y, acc_z))
    jerks = np.vstack((jerk_x, jerk_y, jerk_z))

    if yaw_mode == 'follow':
        yaw_traj = -theta + yaw_offset
        yaw_rate = -omega_vec
    else:
        yaw_traj = np.full_like(theta, yaw_constant)
        yaw_rate = np.zeros_like(theta)

    traj_derivs = np.stack((positions, velocities, accelerations, jerks), axis=0)
    yaw_derivs = np.vstack((yaw_traj, yaw_rate))
    t_ref = np.arange(samples, dtype=float) * dt

    return traj_derivs, yaw_derivs, t_ref


def _minimum_snap_reference(traj_derivs: np.ndarray,
                            yaw_derivs: np.ndarray,
                            t_ref: np.ndarray,
                            quad: _QuadModel,
                            gravity: float) -> Dict[str, np.ndarray]:
    if traj_derivs.shape[2] < 2:
        raise ValueError('Trajectory must contain at least two samples.')

    dt = float(np.mean(np.diff(t_ref)))

    accel = traj_derivs[2].T
    jerk = traj_derivs[3].T

    thrust_world = accel + np.array([0.0, 0.0, gravity])
    z_body = _normalize(thrust_world)

    total_thrust = quad.mass * np.einsum('ij,ij->i', z_body, thrust_world)
    total_thrust = np.clip(total_thrust, 1e-6, None)

    yaw = yaw_derivs[0]
    yaw_rate = yaw_derivs[1]

    x_world = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
    y_body = np.cross(z_body, x_world)
    invalid = np.linalg.norm(y_body, axis=1) < 1e-6
    if np.any(invalid):
        y_body[invalid] = np.array([0.0, 1.0, 0.0])
    y_body = _normalize(y_body)
    x_body = _normalize(np.cross(y_body, z_body))

    rot_mats = np.stack((x_body, y_body, z_body), axis=-1)
    quats = np.zeros((rot_mats.shape[0], 4), dtype=float)
    for idx, rot in enumerate(rot_mats):
        quat = _rotation_matrix_to_quat(rot)
        if idx > 0:
            quat = _ensure_quaternion_continuity(quats[idx - 1], quat)
        quats[idx] = quat

    a_proj = np.einsum('ij,ij->i', z_body, jerk)
    h_omega = (quad.mass / total_thrust)[:, np.newaxis] * (jerk - a_proj[:, np.newaxis] * z_body)

    rates = np.zeros_like(traj_derivs[1].T)
    rates[:, 0] = -np.einsum('ij,ij->i', h_omega, y_body)
    rates[:, 1] = np.einsum('ij,ij->i', h_omega, x_body)
    rates[:, 2] = -yaw_rate * z_body[:, 2]

    edge_order = 2 if rates.shape[0] > 2 else 1
    rate_dot = np.gradient(rates, axis=0, edge_order=edge_order) / dt
    rate_cross = np.column_stack((
        (quad.inertia[2] - quad.inertia[1]) * rates[:, 2] * rates[:, 1],
        (quad.inertia[0] - quad.inertia[2]) * rates[:, 0] * rates[:, 2],
        (quad.inertia[1] - quad.inertia[0]) * rates[:, 1] * rates[:, 0],
    ))

    tau = rate_dot * quad.inertia[np.newaxis, :] + rate_cross
    b = np.column_stack((tau, total_thrust))
    act_mat = quad.allocation_matrix
    act_inv = np.linalg.inv(act_mat)
    thrusts = (act_inv @ b.T).T
    thrusts = np.clip(thrusts, 0.0, quad.max_thrust)

    positions = traj_derivs[0].T
    velocities = traj_derivs[1].T

    return {
        'positions': positions,
        'velocities': velocities,
        'quaternions': quats,
        'body_rates': rates,
        'thrusts': thrusts,
        'yaws': yaw,
        'times': t_ref,
    }


def _normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return vec / norm


def _rotation_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    quat /= np.linalg.norm(quat)
    return quat


def _ensure_quaternion_continuity(prev: np.ndarray, current: np.ndarray) -> np.ndarray:
    return current if np.dot(prev, current) >= 0.0 else -current


__all__ = ['generate_loop_reference']
