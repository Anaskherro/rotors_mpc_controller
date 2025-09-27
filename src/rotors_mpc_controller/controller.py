"""Core NMPC controller built on acados for quadrotor position tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    import casadi as ca
except ImportError as exc:  # pragma: no cover - handled at runtime.
    raise ImportError(
        'The acados Python interface is required to use rotors_mpc_controller. '
        'Install acados and ensure acados_template is on PYTHONPATH.'
    ) from exc


@dataclass
class ControllerParams:
    horizon_steps: int
    dt: float
    position_weight: np.ndarray
    velocity_weight: np.ndarray
    attitude_weight: np.ndarray
    rate_weight: np.ndarray
    thrust_weight: float
    terminal_position_weight: np.ndarray
    terminal_velocity_weight: np.ndarray
    terminal_attitude_weight: np.ndarray
    terminal_rate_weight: np.ndarray
    terminal_thrust_weight: float
    control_weight: np.ndarray
    control_lower: np.ndarray
    control_upper: np.ndarray
    regularization: float
    codegen_directory: Path


def _rotation_matrix(roll, pitch, yaw):
    cr = ca.cos(roll)
    sr = ca.sin(roll)
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cy = ca.cos(yaw)
    sy = ca.sin(yaw)
    return ca.vertcat(
        ca.horzcat(cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp, cp * sr, cp * cr),
    )


class PositionNMPC:
    """Nonlinear MPC with attitude and thrust dynamics."""

    def __init__(self, params: Dict[str, Dict[str, object]]) -> None:
        solver_cfg = params['solver']
        self.mass = float(params['vehicle']['mass'])
        world_cfg = params.get('world', {})
        self.gravity = float(world_cfg.get('gravity', 9.81))
        controller_cfg = params['controller']
        thrust_limits = controller_cfg.get('thrust_limits', [4.0, 32.0])
        self._thrust_limits = (float(thrust_limits[0]), float(thrust_limits[1]))
        vehicle_drag = params['vehicle'].get('drag_coefficients', [0.05, 0.05, 0.10])
        if len(vehicle_drag) != 3:
            raise ValueError('vehicle.drag_coefficients must contain 3 values')
        self.drag_coefficients = np.asarray(vehicle_drag, dtype=float)

        codegen_dir = Path(solver_cfg.get('codegen_directory',
                                          Path.home() / '.cache' / 'rotors_mpc_controller'))
        codegen_dir.mkdir(parents=True, exist_ok=True)

        self.config = ControllerParams(
            horizon_steps=int(solver_cfg['horizon_steps']),
            dt=float(solver_cfg['dt']),
            position_weight=np.asarray(solver_cfg['position_weight'], dtype=float),
            velocity_weight=np.asarray(solver_cfg['velocity_weight'], dtype=float),
            attitude_weight=np.asarray(solver_cfg['attitude_weight'], dtype=float),
            rate_weight=np.asarray(solver_cfg['rate_weight'], dtype=float),
            thrust_weight=float(solver_cfg['thrust_weight']),
            terminal_position_weight=np.asarray(solver_cfg['terminal_position_weight'], dtype=float),
            terminal_velocity_weight=np.asarray(solver_cfg['terminal_velocity_weight'], dtype=float),
            terminal_attitude_weight=np.asarray(solver_cfg['terminal_attitude_weight'], dtype=float),
            terminal_rate_weight=np.asarray(solver_cfg['terminal_rate_weight'], dtype=float),
            terminal_thrust_weight=float(solver_cfg['terminal_thrust_weight']),
            control_weight=np.asarray(solver_cfg['control_weight'], dtype=float),
            control_lower=np.asarray(solver_cfg['control_limits']['lower'], dtype=float),
            control_upper=np.asarray(solver_cfg['control_limits']['upper'], dtype=float),
            regularization=float(solver_cfg.get('regularization', 1e-2)),
            codegen_directory=codegen_dir,
        )

        self.nx = 13
        self.nu = 4
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        self._solver = self._create_solver()
        self._prev_solution = {
            'u': np.zeros((self.config.horizon_steps, self.nu), dtype=float),
            'x': np.zeros((self.config.horizon_steps + 1, self.nx), dtype=float),
        }
        self._prev_solution_valid = False
        self._last_thrust = self.mass * self.gravity

    # ------------------------------------------------------------------
    def _create_solver(self) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        ocp.model = self._build_model()

        ocp.solver_options.N_horizon = self.config.horizon_steps
        ocp.solver_options.tf = self.config.horizon_steps * self.config.dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.qp_solver_cond_N = min(self.config.horizon_steps, 5)
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.regularize_method = 'PROJECT_REDUC_HESS'
        ocp.solver_options.levenberg_marquardt = self.config.regularization

        ocp.code_export_directory = str(self.config.codegen_directory)
        ocp.json_file = 'rotors_nmpc_ocp.json'

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        Vx = np.zeros((self.ny, self.nx))
        Vu = np.zeros((self.ny, self.nu))
        Vx[:self.nx, :self.nx] = np.eye(self.nx)
        Vu[self.nx:, :] = np.eye(self.nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(self.nx)

        stage_state_weights = np.concatenate((
            self.config.position_weight,
            self.config.velocity_weight,
            self.config.attitude_weight,
            self.config.rate_weight,
            [self.config.thrust_weight],
        ))
        ocp.cost.W = np.diag(np.concatenate((stage_state_weights, self.config.control_weight)))

        terminal_state_weights = np.concatenate((
            self.config.terminal_position_weight,
            self.config.terminal_velocity_weight,
            self.config.terminal_attitude_weight,
            self.config.terminal_rate_weight,
            [self.config.terminal_thrust_weight],
        ))
        ocp.cost.W_e = np.diag(terminal_state_weights)
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)

        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.config.control_lower
        ocp.constraints.ubu = self.config.control_upper

        ocp.constraints.idxbx_0 = np.arange(self.nx)
        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)

        ocp.constraints.idxbx = np.arange(self.nx)
        ocp.constraints.lbx = -1e6 * np.ones(self.nx)
        ocp.constraints.ubx = 1e6 * np.ones(self.nx)

        return AcadosOcpSolver(ocp, json_file=ocp.json_file)

    # ------------------------------------------------------------------
    def _build_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = 'rotors_full_dynamics'

        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        xdot = ca.SX.sym('xdot', self.nx)

        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch, yaw = x[6], x[7], x[8]
        p_body, q_body, r_body = x[9], x[10], x[11]
        thrust = x[12]

        roll_acc, pitch_acc, yaw_acc, thrust_rate = u[0], u[1], u[2], u[3]

        R = _rotation_matrix(roll, pitch, yaw)
        thrust_body = ca.vertcat(0.0, 0.0, thrust)
        drag = ca.vertcat(self.drag_coefficients[0] * vx,
                          self.drag_coefficients[1] * vy,
                          self.drag_coefficients[2] * vz)
        accel_world = (1.0 / self.mass) * ca.mtimes(R, thrust_body) - drag - ca.vertcat(0.0, 0.0, self.gravity)

        tan_pitch = ca.tan(pitch)
        sec_pitch = 1.0 / ca.cos(pitch)

        roll_dot = p_body + q_body * ca.sin(roll) * tan_pitch + r_body * ca.cos(roll) * tan_pitch
        pitch_dot = q_body * ca.cos(roll) - r_body * ca.sin(roll)
        yaw_dot = q_body * ca.sin(roll) * sec_pitch + r_body * ca.cos(roll) * sec_pitch

        f_expl = ca.vertcat(
            vx,
            vy,
            vz,
            accel_world[0],
            accel_world[1],
            accel_world[2],
            roll_dot,
            pitch_dot,
            yaw_dot,
            roll_acc,
            pitch_acc,
            yaw_acc,
            thrust_rate,
        )

        model.f_impl_expr = xdot - f_expl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.z = ca.SX.sym('z', 0)
        model.p = ca.SX.sym('p', 0)
        return model

    # ------------------------------------------------------------------
    @property
    def horizon(self) -> int:
        return self.config.horizon_steps

    @property
    def dt(self) -> float:
        return self.config.dt

    # ------------------------------------------------------------------
    def solve(self,
              state: Dict[str, np.ndarray],
              reference: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
        position = state['position']
        velocity = state['velocity']
        roll, pitch, yaw = state['attitude']
        body_rates = state['body_rates']

        x0 = np.concatenate((position,
                             velocity,
                             np.array([roll, pitch, yaw], dtype=float),
                             body_rates,
                             np.array([self._last_thrust], dtype=float)))

        solver = self._solver
        solver.set(0, 'lbx', x0)
        solver.set(0, 'ubx', x0)
        solver.set(0, 'x', x0)

        if self._prev_solution_valid:
            solver.set(0, 'u', self._prev_solution['u'][0])
            for stage in range(1, self.config.horizon_steps):
                solver.set(stage, 'x', self._prev_solution['x'][stage])
                solver.set(stage, 'u', self._prev_solution['u'][stage])
            solver.set(self.config.horizon_steps, 'x', self._prev_solution['x'][-1])
        else:
            zero_u = np.zeros(self.nu)
            solver.set(0, 'u', zero_u)
            for stage in range(1, self.config.horizon_steps + 1):
                solver.set(stage, 'x', x0)
            for stage in range(self.config.horizon_steps):
                solver.set(stage, 'u', zero_u)

        first_ref_state = None
        for stage in range(self.config.horizon_steps):
            state_ref = self._stage_reference(reference, stage)
            if stage == 0:
                first_ref_state = state_ref.copy()
            yref = np.concatenate((state_ref, np.zeros(self.nu)))
            solver.set(stage, 'yref', yref)

        terminal_ref = self._stage_reference(reference, self.config.horizon_steps)
        solver.set(self.config.horizon_steps, 'yref', terminal_ref)

        status = solver.solve()
        if status != 0:
            self._prev_solution_valid = False
            self._prev_solution['u'].fill(0.0)
            self._prev_solution['x'].fill(0.0)
            self._last_thrust = self.mass * self.gravity
            fallback_command = np.array([0.0, 0.0, 0.0, self._last_thrust], dtype=float)
            info = {
                'control': np.zeros(self.nu),
                'reference_state': first_ref_state if first_ref_state is not None else np.zeros(self.nx),
                'predicted_state': x0,
                'residuals': self._solver.get_residuals(),
            }
            return fallback_command, status, info

        u0 = np.asarray(solver.get(0, 'u')).reshape(-1)

        for stage in range(self.config.horizon_steps):
            self._prev_solution['u'][stage] = np.asarray(solver.get(stage, 'u')).reshape(-1)
            self._prev_solution['x'][stage] = np.asarray(solver.get(stage, 'x')).reshape(-1)
        self._prev_solution['x'][-1] = np.asarray(solver.get(self.config.horizon_steps, 'x')).reshape(-1)
        self._prev_solution_valid = True

        x_next = np.asarray(solver.get(1, 'x')).reshape(-1)
        roll_cmd = x_next[6]
        pitch_cmd = x_next[7]
        yaw_rate_cmd = x_next[11]
        thrust_cmd = float(np.clip(x_next[12], self._thrust_limits[0], self._thrust_limits[1]))
        self._last_thrust = thrust_cmd

        command = np.array([roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd], dtype=float)

        info = {
            'control': u0.copy(),
            'reference_state': first_ref_state if first_ref_state is not None else np.zeros(self.nx),
            'predicted_state': x_next.copy(),
            'residuals': self._solver.get_residuals(),
        }
        return command, status, info

    # ------------------------------------------------------------------
    def _stage_reference(self, reference: Dict[str, np.ndarray], index: int) -> np.ndarray:
        pos = reference['positions'][index]
        vel = reference['velocities'][index]
        acc = reference['accelerations'][index]
        yaw = reference['yaws'][index]
        roll_ref, pitch_ref, thrust_ref = compute_attitude_from_accel(
            acc, self.mass, self.gravity, self._thrust_limits, yaw)

        state_ref = np.hstack((pos,
                               vel,
                               [roll_ref, pitch_ref, yaw],
                               np.zeros(3),
                               [thrust_ref]))
        return state_ref


def compute_attitude_from_accel(acc_cmd: np.ndarray,
                                mass: float,
                                gravity: float,
                                thrust_limits: Tuple[float, float],
                                yaw_ref: float) -> Tuple[float, float, float]:
    """Map desired world-frame acceleration to roll, pitch, thrust."""

    acc_cmd = np.asarray(acc_cmd, dtype=float)
    desired_force = mass * (acc_cmd + np.array([0.0, 0.0, gravity], dtype=float))
    norm_force = np.linalg.norm(desired_force)
    if norm_force < 1e-6:
        desired_force = np.array([0.0, 0.0, thrust_limits[0]], dtype=float)
        norm_force = thrust_limits[0]

    thrust = float(np.clip(norm_force, thrust_limits[0], thrust_limits[1]))
    z_b = desired_force / norm_force

    yaw = yaw_ref
    y_c = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
    x_b = np.cross(y_c, z_b)
    if np.linalg.norm(x_b) < 1e-6:
        x_b = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    else:
        x_b /= np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)
    y_b /= np.linalg.norm(y_b)

    R = np.column_stack((x_b, y_b, z_b))

    pitch = math.asin(-R[2, 0])
    roll = math.atan2(R[2, 1], R[2, 2])

    return roll, pitch, thrust
