"""Low-level mixing utilities for quadrotor rotor speed computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class AttitudeGains:
    roll_kp: float
    roll_kd: float
    pitch_kp: float
    pitch_kd: float
    yaw_kp: float
    yaw_kd: float


class RotorMixer:
    """Maps attitude commands to rotor speeds using a simple feedback law."""

    def __init__(self, params: Dict[str, object]) -> None:
        vehicle = params['vehicle']
        controller = params['controller']

        self.mass = float(vehicle['mass'])
        self.inertia = np.asarray(vehicle.get('inertia',
                                              [0.007, 0.0, 0.0,
                                               0.0, 0.007, 0.0,
                                               0.0, 0.0, 0.012]), dtype=float).reshape(3, 3)
        self.arm_length = float(vehicle['arm_length'])
        self.kf = float(vehicle['rotor_force_constant'])
        self.km = float(vehicle['rotor_moment_constant'])
        self.omega_min = float(vehicle.get('motor_min_speed', 0.0))
        self.omega_max = float(vehicle.get('motor_max_speed', 2000.0))
        self.drag_coefficients = np.asarray(vehicle.get('drag_coefficients',
                                                        [0.0, 0.0, 0.0]), dtype=float)

        gains = controller.get('attitude_gains', {})
        self.gains = AttitudeGains(
            roll_kp=float(gains.get('roll', {}).get('kp', 8.0)),
            roll_kd=float(gains.get('roll', {}).get('kd', 2.0)),
            pitch_kp=float(gains.get('pitch', {}).get('kp', 8.0)),
            pitch_kd=float(gains.get('pitch', {}).get('kd', 2.0)),
            yaw_kp=float(gains.get('yaw', {}).get('kp', 1.0)),
            yaw_kd=float(gains.get('yaw', {}).get('kd', 0.2)),
        )

        thrust_limits = controller.get('thrust_limits', [2.0, 25.0])
        self.thrust_min = float(thrust_limits[0])
        self.thrust_max = float(thrust_limits[1])

        self._allocation = np.array([
            [self.kf, self.kf, self.kf, self.kf],
            [0.0, self.kf * self.arm_length, 0.0, -self.kf * self.arm_length],
            [-self.kf * self.arm_length, 0.0, self.kf * self.arm_length, 0.0],
            [self.km, -self.km, self.km, -self.km],
        ])
        self._allocation_inv = np.linalg.inv(self._allocation)

    def compute_motor_speeds(self,
                              roll_cmd: float,
                              pitch_cmd: float,
                              yaw_rate_cmd: float,
                              thrust_cmd: float,
                              attitude: Tuple[float, float, float],
                              body_rates: Tuple[float, float, float]) -> np.ndarray:
        roll, pitch, yaw = attitude
        p_rate, q_rate, r_rate = body_rates

        thrust = np.clip(thrust_cmd, self.thrust_min, self.thrust_max)

        tau_x = self.gains.roll_kp * (roll_cmd - roll) - self.gains.roll_kd * p_rate
        tau_y = self.gains.pitch_kp * (pitch_cmd - pitch) - self.gains.pitch_kd * q_rate
        tau_z = self.gains.yaw_kp * (yaw_rate_cmd - r_rate) - self.gains.yaw_kd * r_rate

        generalized_forces = np.array([thrust, tau_x, tau_y, tau_z], dtype=float)
        rotor_sq = self._allocation_inv @ generalized_forces
        rotor_sq = np.clip(rotor_sq, self.omega_min ** 2, self.omega_max ** 2)
        rotor_speeds = np.sqrt(rotor_sq)
        return rotor_speeds


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return 0.0, 0.0, 0.0
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
