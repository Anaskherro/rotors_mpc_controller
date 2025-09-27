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
    control_weight: np.ndarray
    terminal_weight: np.ndarray
    accel_limits: np.ndarray
    regularization: float
    codegen_directory: Path


class PositionNMPC:
    """Nonlinear MPC based on a triple-integrator model in acados."""

    def __init__(self, params: Dict[str, Dict[str, object]]) -> None:
        solver_cfg = params['solver']
        self.mass = float(params['vehicle']['mass'])
        world_cfg = params.get('world', {})
        self.gravity = float(world_cfg.get('gravity', 9.81))

        codegen_dir = Path(solver_cfg.get('codegen_directory',
                                          Path.home() / '.cache' / 'rotors_mpc_controller'))
        codegen_dir.mkdir(parents=True, exist_ok=True)

        self.config = ControllerParams(
            horizon_steps=int(solver_cfg['horizon_steps']),
            dt=float(solver_cfg['dt']),
            position_weight=np.asarray(solver_cfg.get('position_weight', [10.0, 10.0, 10.0]),
                                       dtype=float),
            velocity_weight=np.asarray(solver_cfg.get('velocity_weight', [2.0, 2.0, 2.0]),
                                       dtype=float),
            control_weight=np.asarray(solver_cfg.get('control_weight', [0.5, 0.5, 0.5]),
                                      dtype=float),
            terminal_weight=np.asarray(solver_cfg.get('terminal_weight',
                                                       [10.0, 10.0, 15.0, 2.0, 2.0, 3.0]),
                                       dtype=float),
            accel_limits=np.asarray(solver_cfg.get('accel_limits', [6.0, 6.0, 6.0]), dtype=float),
            regularization=float(solver_cfg.get('regularization', 1e-6)),
            codegen_directory=codegen_dir,
        )

        self.nx = 6
        self.nu = 3
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        self._solver = self._create_solver()
        self._prev_solution = {
            'u': np.zeros((self.config.horizon_steps, self.nu), dtype=float),
            'x': np.zeros((self.config.horizon_steps + 1, self.nx), dtype=float),
        }
        self._prev_solution_valid = False

    # ------------------------------------------------------------------
    def _create_solver(self) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        ocp.model = self._build_model()

        ocp.dims.N = self.config.horizon_steps
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

        # Linear least squares cost.
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        Vx = np.zeros((self.ny, self.nx))
        Vu = np.zeros((self.ny, self.nu))
        Vx[:self.nx, :self.nx] = np.eye(self.nx)
        Vu[self.nx:, :] = np.eye(self.nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(self.nx)

        stage_weights = np.concatenate((self.config.position_weight,
                                         self.config.velocity_weight,
                                         self.config.control_weight))
        ocp.cost.W = np.diag(stage_weights)
        ocp.cost.W_e = np.diag(self.config.terminal_weight)
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)

        # Input bounds (acceleration command limits).
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = -self.config.accel_limits
        ocp.constraints.ubu = self.config.accel_limits

        # Initial state equality constraint will be set at runtime.
        ocp.constraints.idxbx_0 = np.arange(self.nx)
        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)

        ocp.constraints.idxbx = np.arange(self.nx)
        lbx = -1e6 * np.ones(self.nx)
        ubx = 1e6 * np.ones(self.nx)
        ocp.constraints.lbx = lbx
        ocp.constraints.ubx = ubx

        solver = AcadosOcpSolver(ocp, json_file=ocp.json_file)
        return solver

    # ------------------------------------------------------------------
    def _build_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = 'rotors_position'

        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        xdot = ca.SX.sym('xdot', self.nx)

        vx = x[3]
        vy = x[4]
        vz = x[5]
        ax = u[0]
        ay = u[1]
        az = u[2]

        f_expl = ca.vertcat(vx, vy, vz, ax, ay, az)

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
              position: np.ndarray,
              velocity: np.ndarray,
              reference: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
        """Solve the NMPC problem for the current state.

        Parameters
        ----------
        position : np.ndarray
            Current position in world frame (3,).
        velocity : np.ndarray
            Current velocity in world frame (3,).
        reference : dict
            Reference dictionary with keys ``positions``, ``velocities`` and
            ``accelerations``; each array must have length ``horizon + 1``.
        """

        if position.shape[0] != 3 or velocity.shape[0] != 3:
            raise ValueError('Position and velocity must be 3D vectors.')

        x0 = np.concatenate((position, velocity))
        solver = self._solver

        solver.set(0, 'lbx', x0)
        solver.set(0, 'ubx', x0)
        solver.set(0, 'x', x0)

        # Warm start with previous solution.
        if self._prev_solution_valid:
            solver.set(0, 'u', self._prev_solution['u'][0])
            for stage in range(1, self.config.horizon_steps):
                solver.set(stage, 'x', self._prev_solution['x'][stage])
                solver.set(stage, 'u', self._prev_solution['u'][stage])
            solver.set(self.config.horizon_steps, 'x', self._prev_solution['x'][-1])
        else:
            zero_u = np.zeros(self.nu)
            solver.set(0, 'u', zero_u)
            for stage in range(1, self.config.horizon_steps):
                solver.set(stage, 'x', x0)
                solver.set(stage, 'u', zero_u)
            solver.set(self.config.horizon_steps, 'x', x0)

        for stage in range(self.config.horizon_steps):
            yref = np.hstack((reference['positions'][stage],
                               reference['velocities'][stage],
                               reference['accelerations'][stage]))
            solver.set(stage, 'yref', yref)

        terminal_ref = np.hstack((reference['positions'][-1], reference['velocities'][-1]))
        solver.set(self.config.horizon_steps, 'yref', terminal_ref)

        status = solver.solve()
        if status != 0:
            self._prev_solution_valid = False
            return np.zeros(self.nu), status

        u0 = np.asarray(solver.get(0, 'u')).reshape(-1)

        # Cache trajectories for warm start.
        for stage in range(self.config.horizon_steps):
            self._prev_solution['u'][stage] = np.asarray(solver.get(stage, 'u')).reshape(-1)
            self._prev_solution['x'][stage] = np.asarray(solver.get(stage, 'x')).reshape(-1)
        self._prev_solution['x'][-1] = np.asarray(
            solver.get(self.config.horizon_steps, 'x')
        ).reshape(-1)
        self._prev_solution_valid = True

        return u0, status


def compute_attitude_from_accel(acc_cmd: np.ndarray,
                                mass: float,
                                gravity: float,
                                thrust_limits: Tuple[float, float],
                                yaw_ref: float) -> Tuple[float, float, float]:
    """Map desired world-frame acceleration to roll, pitch, thrust."""

    if acc_cmd.shape[0] != 3:
        raise ValueError('Acceleration command must have length 3')

    desired_force = mass * (acc_cmd + np.array([0.0, 0.0, gravity], dtype=float))
    norm_force = np.linalg.norm(desired_force)
    if norm_force < 1e-6:
        desired_force = np.array([0.0, 0.0, thrust_limits[0]], dtype=float)
        norm_force = thrust_limits[0]

    thrust = float(np.clip(norm_force, thrust_limits[0], thrust_limits[1]))

    z_b = desired_force / np.linalg.norm(desired_force)

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
