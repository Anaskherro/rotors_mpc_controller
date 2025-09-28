"""Core NMPC controller built on acados for quadrotor position tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import hashlib
import shutil

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
    quaternion_weight: np.ndarray
    rate_weight: np.ndarray
    control_weight: np.ndarray
    terminal_weight: np.ndarray
    regularization: float
    solver_iter_max: int
    codegen_directory: Path
    mass: float
    inertia: np.ndarray
    gravity: float
    rotor_force_constant: float
    rotor_moment_constant: float
    rotor_x_offsets: np.ndarray
    rotor_y_offsets: np.ndarray
    rotor_z_torque: np.ndarray
    input_lower_bounds: np.ndarray
    input_upper_bounds: np.ndarray
    hover_thrust_per_motor: float
    motor_min_speed: float
    motor_max_speed: float


class PositionNMPC:
    """Nonlinear MPC based on a triple-integrator model in acados."""

    def __init__(self, params: Dict[str, Dict[str, object]]) -> None:
        self._solver: AcadosOcpSolver | None = None
        self.codegen_path: Path | None = None
        self._prev_solution: Dict[str, np.ndarray] | None = None
        self._prev_solution_valid = False
        self.reconfigure(params)

    # ------------------------------------------------------------------
    def reconfigure(self, params: Dict[str, Dict[str, object]]) -> None:
        """Rebuild the ACADOS model/solver with updated parameters."""

        old_solver = self._solver
        old_codegen_path = self.codegen_path

        solver_cfg = params['solver']
        vehicle_cfg = params['vehicle']
        world_cfg = params.get('world', {})

        self.mass = float(vehicle_cfg['mass'])
        self.gravity = float(world_cfg.get('gravity', 9.81))

        inertia_matrix = np.asarray(vehicle_cfg.get('inertia',
                                                    [0.007, 0.0, 0.0,
                                                     0.0, 0.007, 0.0,
                                                     0.0, 0.0, 0.012]),
                                    dtype=float).reshape(3, 3)
        inertia_diag = np.array([inertia_matrix[0, 0],
                                 inertia_matrix[1, 1],
                                 inertia_matrix[2, 2]], dtype=float)

        arm_length = float(vehicle_cfg.get('arm_length', 0.17))
        rotor_force_constant = float(vehicle_cfg.get('rotor_force_constant', 8.54858e-6))
        rotor_moment_constant = float(vehicle_cfg.get('rotor_moment_constant', 0.016))
        motor_min_speed = float(vehicle_cfg.get('motor_min_speed', 0.0))
        motor_max_speed = float(vehicle_cfg.get('motor_max_speed', 2000.0))

        configuration = params.get('vehicle', {}).get('rotor_configuration', '+')
        if configuration != '+':
            configuration = '+'  # enforce '+' layout for rotor-level NMPC

        if configuration == '+':
            rotor_x_offsets = np.array([arm_length, 0.0, -arm_length, 0.0], dtype=float)
            rotor_y_offsets = np.array([0.0, arm_length, 0.0, -arm_length], dtype=float)
        else:
            h = float(np.cos(np.pi / 4.0) * arm_length)
            rotor_x_offsets = np.array([h, -h, -h, h], dtype=float)
            rotor_y_offsets = np.array([-h, -h, h, h], dtype=float)

        rotor_z_torque = np.array([-rotor_moment_constant,
                                    rotor_moment_constant,
                                    -rotor_moment_constant,
                                    rotor_moment_constant], dtype=float)

        thrust_min = max(0.0, rotor_force_constant * motor_min_speed ** 2)
        thrust_max = rotor_force_constant * motor_max_speed ** 2
        hover_thrust_per_motor = self.mass * self.gravity / 4.0

        input_lower_bounds = np.full(4, thrust_min, dtype=float)
        input_upper_bounds = np.full(4, thrust_max, dtype=float)

        codegen_dir = Path(
            solver_cfg.get(
                'codegen_directory',
                Path.home() / '.cache' / 'rotors_mpc_controller'
            )
        ).expanduser()
        codegen_dir.mkdir(parents=True, exist_ok=True)

        self.config = ControllerParams(
            horizon_steps=int(solver_cfg['horizon_steps']),
            dt=float(solver_cfg['dt']),
            position_weight=np.asarray(solver_cfg.get('position_weight', [10.0, 10.0, 10.0]),
                                       dtype=float),
            velocity_weight=np.asarray(solver_cfg.get('velocity_weight', [2.0, 2.0, 2.0]),
                                       dtype=float),
            quaternion_weight=np.asarray(solver_cfg.get('quaternion_weight', [50.0, 50.0, 50.0, 50.0]),
                                         dtype=float),
            rate_weight=np.asarray(solver_cfg.get('rate_weight', [1.0, 1.0, 1.0]), dtype=float),
            control_weight=np.asarray(solver_cfg.get('control_weight', [0.5, 0.5, 0.5, 0.5]),
                                      dtype=float),
            terminal_weight=np.asarray(solver_cfg.get('terminal_weight',
                                                       [20.0, 20.0, 20.0,
                                                        10.0, 10.0, 10.0,
                                                        50.0, 50.0, 50.0, 50.0,
                                                        5.0, 5.0, 5.0]),
                                       dtype=float),
            regularization=float(solver_cfg.get('regularization', 1e-6)),
            solver_iter_max=int(solver_cfg.get('iter_max', 2)),
            codegen_directory=codegen_dir,
            mass=self.mass,
            inertia=inertia_diag,
            gravity=self.gravity,
            rotor_force_constant=rotor_force_constant,
            rotor_moment_constant=rotor_moment_constant,
            rotor_x_offsets=rotor_x_offsets,
            rotor_y_offsets=rotor_y_offsets,
            rotor_z_torque=rotor_z_torque,
            input_lower_bounds=input_lower_bounds,
            input_upper_bounds=input_upper_bounds,
            hover_thrust_per_motor=hover_thrust_per_motor,
            motor_min_speed=motor_min_speed,
            motor_max_speed=motor_max_speed,
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

        if old_solver is not None:
            del old_solver
        if old_codegen_path is not None and old_codegen_path != self.codegen_path:
            shutil.rmtree(old_codegen_path, ignore_errors=True)

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
        ocp.solver_options.qp_solver_iter_max = self.config.solver_iter_max
        ocp.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.regularize_method = 'PROJECT_REDUC_HESS'
        ocp.solver_options.levenberg_marquardt = self.config.regularization

        codegen_signature = repr((self.config.horizon_steps,
                                  self.config.dt,
                                  tuple(self.config.position_weight),
                                  tuple(self.config.velocity_weight),
                                  tuple(self.config.quaternion_weight),
                                  tuple(self.config.rate_weight),
                                  tuple(self.config.control_weight),
                                  tuple(self.config.terminal_weight),
                                  self.config.mass,
                                  tuple(self.config.inertia),
                                  self.config.gravity,
                                  self.config.rotor_force_constant,
                                  self.config.rotor_moment_constant,
                                  tuple(self.config.rotor_x_offsets),
                                  tuple(self.config.rotor_y_offsets),
                                  tuple(self.config.rotor_z_torque),
                                  tuple(self.config.input_lower_bounds),
                                  tuple(self.config.input_upper_bounds),
                                  self.config.regularization,
                                  self.config.solver_iter_max))
        digest = hashlib.sha1(codegen_signature.encode('utf-8')).hexdigest()[:10]
        self.codegen_path = (self.config.codegen_directory /
                             f'h{self.config.horizon_steps}_dt{int(self.config.dt * 1e6)}_{digest}')
        if self.codegen_path.exists():
            shutil.rmtree(self.codegen_path)
        self.codegen_path.mkdir(parents=True, exist_ok=True)

        ocp.code_export_directory = str(self.codegen_path)
        ocp.json_file = str(self.codegen_path / 'rotors_nmpc_ocp.json')

        # Linear least squares cost.
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        Vx = np.zeros((self.ny, self.nx))
        Vu = np.zeros((self.ny, self.nu))
        Vx[0:3, 0:3] = np.eye(3)              # position
        Vx[3:6, 3:6] = np.eye(3)              # velocity
        Vx[6:10, 6:10] = np.eye(4)            # quaternion
        Vx[10:13, 10:13] = np.eye(3)          # body rates
        Vu[13:17, :] = np.eye(self.nu)        # rotor thrusts
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(self.nx)

        stage_weights = np.concatenate((self.config.position_weight,
                                         self.config.velocity_weight,
                                         self.config.quaternion_weight,
                                         self.config.rate_weight,
                                         self.config.control_weight))
        ocp.cost.W = np.diag(stage_weights)
        ocp.cost.W_e = np.diag(self.config.terminal_weight)
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)

        # Input bounds (acceleration command limits).
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.config.input_lower_bounds
        ocp.constraints.ubu = self.config.input_upper_bounds

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
        model.name = 'rotors_plus_configuration'

        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        xdot = ca.SX.sym('xdot', self.nx)

        vx = x[3]
        vy = x[4]
        vz = x[5]
        qw = x[6]
        qx_ = x[7]
        qy_ = x[8]
        qz_ = x[9]
        wx = x[10]
        wy = x[11]
        wz = x[12]

        velocity = ca.vertcat(vx, vy, vz)
        omega = ca.vertcat(wx, wy, wz)

        qw2 = qw * qw
        qx2 = qx_ * qx_
        qy2 = qy_ * qy_
        qz2 = qz_ * qz_

        r11 = 1 - 2 * (qy2 + qz2)
        r12 = 2 * (qx_ * qy_ - qw * qz_)
        r13 = 2 * (qx_ * qz_ + qw * qy_)
        r21 = 2 * (qx_ * qy_ + qw * qz_)
        r22 = 1 - 2 * (qx2 + qz2)
        r23 = 2 * (qy_ * qz_ - qw * qx_)
        r31 = 2 * (qx_ * qz_ - qw * qy_)
        r32 = 2 * (qy_ * qz_ + qw * qx_)
        r33 = 1 - 2 * (qx2 + qy2)

        rotation = ca.vertcat(
            ca.hcat([r11, r12, r13]),
            ca.hcat([r21, r22, r23]),
            ca.hcat([r31, r32, r33]),
        )

        thrust_body = ca.vertcat(0, 0, ca.sum1(u))
        mass = float(self.config.mass)
        gravity = float(self.config.gravity)
        acc_world = rotation @ (thrust_body / mass)
        acc_world = acc_world - ca.vertcat(0, 0, gravity)

        qw_dot = 0.5 * (-qx_ * wx - qy_ * wy - qz_ * wz)
        qx_dot = 0.5 * (qw * wx + qy_ * wz - qz_ * wy)
        qy_dot = 0.5 * (qw * wy + qz_ * wx - qx_ * wz)
        qz_dot = 0.5 * (qw * wz + qx_ * wy - qy_ * wx)
        quat_dot = ca.vertcat(qw_dot, qx_dot, qy_dot, qz_dot)

        rotor_x = ca.DM(self.config.rotor_x_offsets)
        rotor_y = ca.DM(self.config.rotor_y_offsets)
        rotor_z = ca.DM(self.config.rotor_z_torque)

        tau_x = ca.dot(u, rotor_y)
        tau_y = ca.dot(u, -rotor_x)
        tau_z = ca.dot(u, rotor_z)
        torque = ca.vertcat(tau_x, tau_y, tau_z)

        inertia = ca.DM(self.config.inertia)
        Jomega = ca.vertcat(inertia[0] * wx,
                            inertia[1] * wy,
                            inertia[2] * wz)
        omega_cross = ca.vertcat(wy * Jomega[2] - wz * Jomega[1],
                                 wz * Jomega[0] - wx * Jomega[2],
                                 wx * Jomega[1] - wy * Jomega[0])
        inertia_inv = ca.DM(1.0 / self.config.inertia)
        omega_dot = ca.vertcat(inertia_inv[0] * (torque[0] - omega_cross[0]),
                                inertia_inv[1] * (torque[1] - omega_cross[1]),
                                inertia_inv[2] * (torque[2] - omega_cross[2]))

        f_expl = ca.vertcat(velocity,
                             acc_world,
                             quat_dot,
                             omega_dot)

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
    @property
    def hover_thrust(self) -> float:
        return self.config.hover_thrust_per_motor

    @property
    def rotor_force_constant(self) -> float:
        return self.config.rotor_force_constant

    @property
    def motor_speed_limits(self) -> Tuple[float, float]:
        return self.config.motor_min_speed, self.config.motor_max_speed

    # ------------------------------------------------------------------
    @property
    def input_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.config.input_lower_bounds, self.config.input_upper_bounds

    # ------------------------------------------------------------------
    def solve(self,
              state: Dict[str, np.ndarray],
              reference: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
        """Solve the NMPC problem for the current state.

        Parameters
        ----------
        state : dict
            Dictionary containing ``position`` (3,), ``velocity`` (3,),
            ``quaternion`` (4,) and ``body_rates`` (3,) arrays.
        reference : dict
            Reference dictionary with keys ``positions``, ``velocities``,
            ``quaternions``, ``body_rates`` (each of length ``horizon + 1``)
            and ``thrusts`` (length ``horizon``).
        """

        position = np.asarray(state['position'], dtype=float).reshape(3)
        velocity = np.asarray(state['velocity'], dtype=float).reshape(3)
        quaternion = np.asarray(state['quaternion'], dtype=float).reshape(4)
        body_rates = np.asarray(state['body_rates'], dtype=float).reshape(3)

        norm_q = np.linalg.norm(quaternion)
        if norm_q == 0.0:
            raise ValueError('Quaternion norm must be non-zero.')
        quaternion = quaternion / norm_q

        x0 = np.concatenate((position, velocity, quaternion, body_rates))
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
                               reference['quaternions'][stage],
                               reference['body_rates'][stage],
                               reference['thrusts'][stage]))
            solver.set(stage, 'yref', yref)

        terminal_ref = np.hstack((reference['positions'][-1],
                                   reference['velocities'][-1],
                                   reference['quaternions'][-1],
                                   reference['body_rates'][-1]))
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
