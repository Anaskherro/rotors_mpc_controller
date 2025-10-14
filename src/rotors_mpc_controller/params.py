"""Parameter loading utilities for the redesigned ACADOS-based controller."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple


def _try_import_rospy():
    try:
        import rospy  # type: ignore
        return rospy
    except Exception:
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing.
        raise RuntimeError('PyYAML is required to load controller parameters.') from exc

    if not path.is_file():
        raise FileNotFoundError(f'Parameter file not found: {path}')

    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Root of {path} must be a mapping.')
    return data


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get('ROTORS_MPC_PARAMS')
    if env:
        paths.append(Path(env).expanduser())

    default_path = Path(__file__).resolve().parent.parent / 'config' / 'params.yaml'
    paths.append(default_path)

    try:
        import rospkg  # type: ignore

        pkg_path = Path(rospkg.RosPack().get_path('rotors_mpc_controller'))
        paths.append(pkg_path / 'config' / 'params.yaml')
    except Exception:
        pass

    seen = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        unique.append(path)
        seen.add(path)
    return unique


def _recursive_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_solver(cfg: Dict[str, Any]) -> None:
    cfg['horizon_steps'] = int(cfg.get('horizon_steps', 20))
    cfg['dt'] = float(cfg.get('dt', 0.05))
    cfg['position_weight'] = [float(v) for v in cfg.get('position_weight', [10.0, 10.0, 8.0])]
    cfg['velocity_weight'] = [float(v) for v in cfg.get('velocity_weight', [1.0, 1.0, 0.2])]
    cfg['quaternion_weight'] = [float(v) for v in cfg.get('quaternion_weight',
                                                         [3.2, 3.2, 3.2, 3.2])]
    cfg['rate_weight'] = [float(v) for v in cfg.get('rate_weight', [1.4, 1.4, 0.4])]
    cfg['control_weight'] = [float(v) for v in cfg.get('control_weight',
                                                       [1.75, 1.75, 1.75, 1.75])]
    cfg['terminal_weight'] = [float(v) for v in cfg.get('terminal_weight',
                                                       [5.0, 5.0, 3.0,
                                                        2.0, 2.0, 2.0,
                                                        12.0, 12.0, 12.0, 18.5,
                                                        2.0, 2.0, 1.8])]
    cfg['regularization'] = float(cfg.get('regularization', 7.0e-3))
    cfg['iter_max'] = int(cfg.get('iter_max', 600))
    if 'codegen_directory' in cfg:
        cfg['codegen_directory'] = str(Path(cfg['codegen_directory']).expanduser())


def _coerce_vehicle(cfg: Dict[str, Any]) -> None:
    cfg['mass'] = float(cfg.get('mass', 0.68))
    inertia = cfg.get('inertia', [0.007, 0.0, 0.0, 0.0, 0.007, 0.0, 0.0, 0.0, 0.012])
    if len(inertia) != 9:
        raise ValueError('vehicle.inertia must contain 9 values (row-major 3x3).')
    cfg['inertia'] = [float(v) for v in inertia]
    cfg['arm_length'] = float(cfg.get('arm_length', 0.17))
    cfg['rotor_force_constant'] = float(cfg.get('rotor_force_constant', 8.54858e-6))
    cfg['rotor_moment_constant'] = float(cfg.get('rotor_moment_constant', 0.016))
    cfg['motor_min_speed'] = float(cfg.get('motor_min_speed', 0.0))
    cfg['motor_max_speed'] = float(cfg.get('motor_max_speed', 2000.0))
    drag = cfg.get('drag_coefficients', [0.0, 0.0, 0.0])
    if len(drag) != 3:
        raise ValueError('vehicle.drag_coefficients must contain 3 values.')
    cfg['drag_coefficients'] = [float(v) for v in drag]
    cfg['rotor_configuration'] = str(cfg.get('rotor_configuration', '+')).strip()


def _coerce_controller(cfg: Dict[str, Any]) -> None:
    thrust_limits = cfg.get('thrust_limits', [4.0, 20.0])
    if len(thrust_limits) != 2:
        raise ValueError('controller.thrust_limits must contain [min, max].')
    cfg['thrust_limits'] = [float(thrust_limits[0]), float(thrust_limits[1])]
    cfg.pop('attitude_gains', None)
    cfg.pop('max_tilt_deg', None)
    cfg.pop('max_tilt_angle', None)


def _coerce_world(cfg: Dict[str, Any]) -> None:
    cfg['gravity'] = float(cfg.get('gravity', 9.81))


def _ensure_required(cfg: Dict[str, Any]) -> None:
    required = {'solver', 'vehicle', 'controller', 'world', 'reference', 'topics', 'node'}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f'Missing required top-level sections: {sorted(missing)}')


def _coerce_reference(cfg: Dict[str, Any]) -> None:
    cfg['frame'] = cfg.get('frame', 'world')
    cfg['default_position'] = [float(v) for v in cfg.get('default_position', [1.0, 1.0, 1.0])]
    cfg['default_velocity'] = [float(v) for v in cfg.get('default_velocity', [0.0, 0.0, 0.0])]
    cfg['default_acceleration'] = [float(v) for v in cfg.get('default_acceleration', [0.0, 0.0, 0.0])]
    cfg['default_yaw'] = float(cfg.get('default_yaw', 0.0))
    traj_cfg = cfg.get('trajectory')
    if isinstance(traj_cfg, dict):
        traj = dict(traj_cfg)
        traj['enabled'] = bool(traj.get('enabled', False))
        traj['type'] = str(traj.get('type', 'circle'))
        traj_type = traj['type'].lower()
        traj['loop'] = bool(traj.get('loop', True))
        traj['yaw_mode'] = str(traj.get('yaw_mode', 'follow')).lower()
        if traj['yaw_mode'] not in {'follow', 'fixed'}:
            raise ValueError("reference.trajectory.yaw_mode must be 'follow' or 'fixed'.")
        traj['profile'] = str(traj.get('profile', 'stop')).lower()
        if traj['profile'] not in {'stop', 'continuous'}:
            raise ValueError("reference.trajectory.profile must be 'stop' or 'continuous'.")
        if 'radius' in traj:
            traj['radius'] = float(traj['radius'])
        if 'altitude' in traj:
            traj['altitude'] = float(traj['altitude'])
        else:
            traj['altitude'] = cfg['default_position'][2]
        if traj_type == 'helical':
            traj['pitch'] = float(traj.get('pitch', 0.5))
            traj['turns'] = float(traj.get('turns', 3.0))
        if traj_type == 'lemniscate':
            traj['inner_radius'] = float(traj.get('inner_radius', traj.get('radius', 2.0)))
            traj['outer_radius'] = float(traj.get('outer_radius', traj.get('radius', 3.0)))
        if 'revolutions' in traj:
            traj['revolutions'] = float(traj['revolutions'])
        else:
            traj['revolutions'] = 1.0
        traj['clockwise'] = bool(traj.get('clockwise', False))
        if 'linear_acceleration' in traj:
            traj['linear_acceleration'] = float(traj['linear_acceleration'])
        else:
            traj['linear_acceleration'] = 0.25
        if 'max_speed' in traj:
            traj['max_speed'] = float(traj['max_speed'])
        elif 'speed' in traj:
            traj['max_speed'] = float(traj.pop('speed'))
        else:
            traj['max_speed'] = 1.0
        if 'start_position_tolerance' in traj:
            traj['start_position_tolerance'] = float(traj['start_position_tolerance'])
        else:
            traj['start_position_tolerance'] = 0.15
        if 'stop_position_tolerance' in traj:
            traj['stop_position_tolerance'] = float(traj['stop_position_tolerance'])
        else:
            traj['stop_position_tolerance'] = traj['start_position_tolerance']
        traj['stop_velocity_tolerance'] = float(traj.get('stop_velocity_tolerance', 0.3))
        if 'center' in traj:
            center = traj['center']
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError('reference.trajectory.center must contain two values.')
            traj['center'] = [float(center[0]), float(center[1])]
        else:
            traj['center'] = [0.0, 0.0]
        if 'dt' in traj:
            traj['dt'] = float(traj['dt'])
        if 'yaw_offset' in traj:
            traj['yaw_offset'] = float(traj['yaw_offset'])
        else:
            traj['yaw_offset'] = 0.0
        if 'yaw_constant' in traj:
            traj['yaw_constant'] = float(traj['yaw_constant'])
        else:
            traj['yaw_constant'] = float(cfg['default_yaw'])
        cfg['trajectory'] = traj
    else:
        cfg['trajectory'] = None

    transition_cfg = cfg.get('transition', {})
    if isinstance(transition_cfg, dict):
        trans = dict(transition_cfg)
    else:
        trans = {}
    trans['enabled'] = bool(trans.get('enabled', True))
    trans['dt'] = float(trans.get('dt', 0.05))
    trans['max_speed'] = float(trans.get('max_speed', 1.0))
    trans['max_acceleration'] = float(trans.get('max_acceleration', 0.75))
    trans['distance_epsilon'] = float(trans.get('distance_epsilon', 0.02))
    trans['max_yaw_rate'] = float(trans.get('max_yaw_rate', 1.0))
    trans['min_duration'] = float(trans.get('min_duration', 0.0))
    yaw_mode = str(trans.get('yaw_mode', 'interpolate')).lower()
    if yaw_mode not in {'interpolate', 'hold'}:
        raise ValueError("reference.transition.yaw_mode must be 'interpolate' or 'hold'.")
    trans['yaw_mode'] = yaw_mode
    cfg['transition'] = trans


def _coerce_topics(cfg: Dict[str, Any]) -> None:
    required = ('state', 'motor', 'reference')
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing topic configuration '{key}'")
    for key in required:
        cfg[key] = str(cfg[key])
    cfg['reference_path'] = str(cfg.get('reference_path', '/mpc_controller/reference_path'))
    cfg['prediction_markers'] = str(cfg.get('prediction_markers', '/mpc_controller/prediction_markers'))


def _coerce_node(cfg: Dict[str, Any]) -> None:
    cfg['rate'] = float(cfg.get('rate', 50.0))
    cfg['log_interval'] = float(cfg.get('log_interval', 3.0))
    cfg.pop('max_tilt_deg', None)
    cfg.pop('yaw_rate_gain', None)
    cfg.pop('yaw_rate_limit', None)


def load_params() -> Dict[str, Any]:
    """Load configuration for the NMPC controller."""

    base: Dict[str, Any] | None = None
    for candidate in _candidate_paths():
        if candidate.is_file():
            base = _load_yaml(candidate)
            base['params_yaml'] = str(candidate)
            break
    if base is None:
        raise FileNotFoundError('No configuration file found for rotors_mpc_controller.')

    rospy = _try_import_rospy()
    if rospy is not None and rospy.core.is_initialized():
        try:
            overrides = rospy.get_param('~')
            if isinstance(overrides, dict):
                _recursive_update(base, overrides)
        except Exception:
            pass

    _ensure_required(base)
    _coerce_solver(base['solver'])
    _coerce_vehicle(base['vehicle'])
    _coerce_controller(base['controller'])
    _coerce_world(base['world'])
    _coerce_reference(base['reference'])
    _coerce_topics(base['topics'])
    _coerce_node(base['node'])

    return base


def apply_dynamic_configuration(params: Dict[str, Any], config: Any) -> Tuple[Dict[str, str], Dict[str, list], Dict[str, float]]:
    """Update parameter dictionary from a dynamic_reconfigure config object."""

    solver_cfg = params['solver']
    solver_cfg['horizon_steps'] = int(config.solver_horizon_steps)
    solver_cfg['dt'] = float(config.solver_dt)
    solver_cfg['position_weight'] = [float(config.solver_position_weight_x),
                                     float(config.solver_position_weight_y),
                                     float(config.solver_position_weight_z)]
    solver_cfg['velocity_weight'] = [float(config.solver_velocity_weight_x),
                                     float(config.solver_velocity_weight_y),
                                     float(config.solver_velocity_weight_z)]
    solver_cfg['quaternion_weight'] = [
        float(getattr(config, 'solver_quat_weight_w', solver_cfg['quaternion_weight'][0])),
        float(getattr(config, 'solver_quat_weight_x', solver_cfg['quaternion_weight'][1])),
        float(getattr(config, 'solver_quat_weight_y', solver_cfg['quaternion_weight'][2])),
        float(getattr(config, 'solver_quat_weight_z', solver_cfg['quaternion_weight'][3])),
    ]
    solver_cfg['rate_weight'] = [
        float(getattr(config, 'solver_rate_weight_x', solver_cfg['rate_weight'][0])),
        float(getattr(config, 'solver_rate_weight_y', solver_cfg['rate_weight'][1])),
        float(getattr(config, 'solver_rate_weight_z', solver_cfg['rate_weight'][2])),
    ]
    solver_cfg['control_weight'] = [
        float(getattr(config, 'solver_control_weight_f1', solver_cfg['control_weight'][0])),
        float(getattr(config, 'solver_control_weight_f2', solver_cfg['control_weight'][1])),
        float(getattr(config, 'solver_control_weight_f3', solver_cfg['control_weight'][2])),
        float(getattr(config, 'solver_control_weight_f4', solver_cfg['control_weight'][3])),
    ]
    solver_cfg['terminal_weight'] = [
        float(getattr(config, 'solver_terminal_weight_px', solver_cfg['terminal_weight'][0])),
        float(getattr(config, 'solver_terminal_weight_py', solver_cfg['terminal_weight'][1])),
        float(getattr(config, 'solver_terminal_weight_pz', solver_cfg['terminal_weight'][2])),
        float(getattr(config, 'solver_terminal_weight_vx', solver_cfg['terminal_weight'][3])),
        float(getattr(config, 'solver_terminal_weight_vy', solver_cfg['terminal_weight'][4])),
        float(getattr(config, 'solver_terminal_weight_vz', solver_cfg['terminal_weight'][5])),
        float(getattr(config, 'solver_terminal_weight_qw', solver_cfg['terminal_weight'][6])),
        float(getattr(config, 'solver_terminal_weight_qx', solver_cfg['terminal_weight'][7])),
        float(getattr(config, 'solver_terminal_weight_qy', solver_cfg['terminal_weight'][8])),
        float(getattr(config, 'solver_terminal_weight_qz', solver_cfg['terminal_weight'][9])),
        float(getattr(config, 'solver_terminal_weight_wx', solver_cfg['terminal_weight'][10])),
        float(getattr(config, 'solver_terminal_weight_wy', solver_cfg['terminal_weight'][11])),
        float(getattr(config, 'solver_terminal_weight_wz', solver_cfg['terminal_weight'][12])),
    ]
    solver_cfg['regularization'] = float(config.solver_regularization)
    solver_cfg['iter_max'] = int(getattr(config, 'solver_iter_max', solver_cfg['iter_max']))
    solver_cfg['codegen_directory'] = str(config.solver_codegen_directory)

    vehicle_cfg = params['vehicle']
    vehicle_cfg['mass'] = float(config.vehicle_mass)
    vehicle_cfg['inertia'] = [float(config.vehicle_inertia_xx),
                               float(config.vehicle_inertia_xy),
                               float(config.vehicle_inertia_xz),
                               float(config.vehicle_inertia_yx),
                               float(config.vehicle_inertia_yy),
                               float(config.vehicle_inertia_yz),
                               float(config.vehicle_inertia_zx),
                               float(config.vehicle_inertia_zy),
                               float(config.vehicle_inertia_zz)]
    vehicle_cfg['arm_length'] = float(config.vehicle_arm_length)
    vehicle_cfg['rotor_force_constant'] = float(config.vehicle_rotor_force_constant)
    vehicle_cfg['rotor_moment_constant'] = float(config.vehicle_rotor_moment_constant)
    vehicle_cfg['motor_min_speed'] = float(config.vehicle_motor_min_speed)
    vehicle_cfg['motor_max_speed'] = float(config.vehicle_motor_max_speed)
    vehicle_cfg['drag_coefficients'] = [float(config.vehicle_drag_x),
                                        float(config.vehicle_drag_y),
                                        float(config.vehicle_drag_z)]

    controller_cfg = params['controller']
    controller_cfg['thrust_limits'] = [float(config.controller_thrust_min),
                                       float(config.controller_thrust_max)]

    world_cfg = params['world']
    world_cfg['gravity'] = float(config.world_gravity)

    reference_cfg = params['reference']
    reference_cfg['frame'] = str(config.reference_frame)
    reference_cfg['default_position'] = [float(config.reference_position_x),
                                         float(config.reference_position_y),
                                         float(config.reference_position_z)]
    reference_cfg['default_velocity'] = [float(config.reference_velocity_x),
                                         float(config.reference_velocity_y),
                                         float(config.reference_velocity_z)]
    reference_cfg['default_yaw'] = float(config.reference_yaw)

    try:
        ref_path_topic = config.topic_reference_path
    except Exception:
        ref_path_topic = params.get('topics', {}).get('reference_path',
                                                      '/mpc_controller/reference_path')
    try:
        pred_marker_topic = config.topic_prediction_markers
    except Exception:
        pred_marker_topic = params.get('topics', {}).get('prediction_markers',
                                                         '/mpc_controller/prediction_markers')
    topics_cfg = {
        'state': str(config.topic_state),
        'motor': str(config.topic_motor),
        'reference': str(config.topic_reference),
        'reference_path': str(ref_path_topic),
        'prediction_markers': str(pred_marker_topic),
    }
    params['topics'] = topics_cfg

    node_cfg = params['node']
    node_cfg['rate'] = float(config.node_rate)
    node_cfg['log_interval'] = float(config.node_log_interval)

    reference_defaults = {
        'position': reference_cfg['default_position'],
        'velocity': reference_cfg['default_velocity'],
        'yaw': reference_cfg['default_yaw'],
        'frame': reference_cfg['frame'],
    }

    node_meta = {
        'rate': node_cfg['rate'],
        'log_interval': node_cfg['log_interval'],
    }

    return topics_cfg, reference_defaults, node_meta
