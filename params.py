"""Parameter utilities for the acados-based MPC controller."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _try_import_rospy():
    try:
        import rospy  # type: ignore
        return rospy
    except Exception:
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('PyYAML is required to load controller parameters.') from exc

    if not path.is_file():
        raise FileNotFoundError(f'Parameter file not found: {path}')

    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Root of {path} must be a mapping.')
    return data


def _candidate_paths() -> List[Path]:
    paths: List[Path] = []
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

    unique: List[Path] = []
    seen = set()
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


def _ensure_vector(cfg: Dict[str, Any], key: str, length: int, default: List[float]) -> List[float]:
    values = cfg.get(key, default)
    if len(values) != length:
        raise ValueError(f"Expected {length} values for '{key}', got {len(values)}")
    coerced = [float(v) for v in values]
    cfg[key] = coerced
    return coerced


def _coerce_solver(cfg: Dict[str, Any]) -> None:
    cfg['horizon_steps'] = int(cfg.get('horizon_steps', 20))
    cfg['dt'] = float(cfg.get('dt', 0.05))
    cfg['position_weight'] = _ensure_vector(cfg, 'position_weight', 3, [30.0, 30.0, 45.0])
    cfg['velocity_weight'] = _ensure_vector(cfg, 'velocity_weight', 3, [10.0, 10.0, 12.0])
    cfg['control_weight'] = _ensure_vector(cfg, 'control_weight', 3, [2.0, 2.0, 2.0])
    cfg['terminal_weight'] = _ensure_vector(cfg, 'terminal_weight', 6,
                                            [120.0, 120.0, 160.0, 20.0, 20.0, 25.0])
    cfg['accel_limits'] = _ensure_vector(cfg, 'accel_limits', 3, [6.0, 6.0, 6.0])
    cfg['regularization'] = float(cfg.get('regularization', 5e-3))
    if 'codegen_directory' in cfg:
        cfg['codegen_directory'] = str(Path(cfg['codegen_directory']).expanduser())


def _coerce_vehicle(cfg: Dict[str, Any]) -> None:
    cfg['mass'] = float(cfg.get('mass', 0.68))
    inertia = cfg.get('inertia', [0.007, 0.0, 0.0,
                                  0.0, 0.007, 0.0,
                                  0.0, 0.0, 0.012])
    if len(inertia) != 9:
        raise ValueError('vehicle.inertia must contain 9 values (row-major 3x3).')
    cfg['inertia'] = [float(v) for v in inertia]
    cfg['arm_length'] = float(cfg.get('arm_length', 0.17))
    cfg['rotor_force_constant'] = float(cfg.get('rotor_force_constant', 8.54858e-6))
    cfg['rotor_moment_constant'] = float(cfg.get('rotor_moment_constant', 0.016))
    cfg['motor_min_speed'] = float(cfg.get('motor_min_speed', 0.0))
    cfg['motor_max_speed'] = float(cfg.get('motor_max_speed', 2000.0))
    cfg['drag_coefficients'] = _ensure_vector(cfg, 'drag_coefficients', 3, [0.0, 0.0, 0.0])


def _coerce_controller(cfg: Dict[str, Any]) -> None:
    thrust_limits = cfg.get('thrust_limits', [2.0, 25.0])
    if len(thrust_limits) != 2:
        raise ValueError('controller.thrust_limits must contain [min, max].')
    cfg['thrust_limits'] = [float(thrust_limits[0]), float(thrust_limits[1])]

    gains = cfg.get('attitude_gains', {})
    for axis in ('roll', 'pitch', 'yaw'):
        gains.setdefault(axis, {})
        axis_cfg = gains[axis]
        axis_cfg['kp'] = float(axis_cfg.get('kp', 4.0))
        axis_cfg['kd'] = float(axis_cfg.get('kd', 1.2 if axis != 'yaw' else 0.35))
    cfg['attitude_gains'] = gains

    # Accept legacy "max_tilt_angle" key while ensuring canonical entry exists.
    if 'max_tilt_deg' in cfg:
        tilt_deg = cfg['max_tilt_deg']
    elif 'max_tilt_angle' in cfg:
        tilt_deg = cfg['max_tilt_angle']
    else:
        tilt_deg = 18.0
    cfg['max_tilt_deg'] = float(tilt_deg)
    if 'max_tilt_angle' in cfg:
        cfg['max_tilt_angle'] = float(tilt_deg)


def _coerce_world(cfg: Dict[str, Any]) -> None:
    cfg['gravity'] = float(cfg.get('gravity', 9.81))


def _coerce_reference(cfg: Dict[str, Any]) -> None:
    cfg['frame'] = cfg.get('frame', 'world')
    cfg['default_position'] = _ensure_vector(cfg, 'default_position', 3, [0.0, 0.0, 1.0])
    cfg['default_velocity'] = _ensure_vector(cfg, 'default_velocity', 3, [0.0, 0.0, 0.0])
    cfg['default_acceleration'] = _ensure_vector(cfg, 'default_acceleration', 3, [0.0, 0.0, 0.0])
    cfg['default_yaw'] = float(cfg.get('default_yaw', 0.0))


def _coerce_topics(cfg: Dict[str, Any]) -> None:
    required = {'state', 'command', 'motor', 'reference'}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing topic configuration entries: {sorted(missing)}")
    for key in required:
        cfg[key] = str(cfg[key])


def _coerce_node(cfg: Dict[str, Any]) -> None:
    cfg['rate'] = float(cfg.get('rate', 60.0))
    cfg['max_tilt_deg'] = float(cfg.get('max_tilt_deg', 18.0))
    cfg['yaw_rate_gain'] = float(cfg.get('yaw_rate_gain', 0.9))
    cfg['yaw_rate_limit'] = float(cfg.get('yaw_rate_limit', 1.2))
    cfg['log_interval'] = float(cfg.get('log_interval', 3.0))


def _ensure_required(cfg: Dict[str, Any]) -> None:
    required = {'solver', 'vehicle', 'controller', 'world', 'reference', 'topics', 'node'}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f'Missing required top-level sections: {sorted(missing)}')


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


_SENTINEL = object()


def _config_search(obj: Any, name: str):
    try:
        return getattr(obj, name)
    except AttributeError:
        pass

    try:
        return obj[name]  # type: ignore[index]
    except Exception:
        pass

    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for value in obj.values():
            result = _config_search(value, name)
            if result is not _SENTINEL:
                return result

    if isinstance(obj, (list, tuple)):
        for item in obj:
            result = _config_search(item, name)
            if result is not _SENTINEL:
                return result

    groups = getattr(obj, 'groups', None)
    if isinstance(groups, list):
        for group in groups:
            result = _config_search(group, name)
            if result is not _SENTINEL:
                return result

    parameters = getattr(obj, 'parameters', None)
    if isinstance(parameters, list):
        for param in parameters:
            if isinstance(param, dict):
                if param.get('name') == name:
                    return param.get('value', _SENTINEL)
            else:
                result = _config_search(param, name)
                if result is not _SENTINEL:
                    return result

    return _SENTINEL


def _config_value(cfg_obj: Any, name: str, default: Any) -> Any:
    result = _config_search(cfg_obj, name)
    return default if result is _SENTINEL else result


def apply_dynamic_configuration(params: Dict[str, Any], config: Any) -> Tuple[Dict[str, str], Dict[str, list], Dict[str, float]]:
    """Update parameter dictionary from a dynamic_reconfigure config object."""

    solver_cfg = params['solver']
    solver_cfg['horizon_steps'] = int(_config_value(config, 'solver_horizon_steps', solver_cfg['horizon_steps']))
    solver_cfg['dt'] = float(_config_value(config, 'solver_dt', solver_cfg['dt']))

    pos_w = solver_cfg['position_weight']
    solver_cfg['position_weight'] = [
        float(_config_value(config, 'solver_position_weight_x', pos_w[0])),
        float(_config_value(config, 'solver_position_weight_y', pos_w[1])),
        float(_config_value(config, 'solver_position_weight_z', pos_w[2])),
    ]

    vel_w = solver_cfg['velocity_weight']
    solver_cfg['velocity_weight'] = [
        float(_config_value(config, 'solver_velocity_weight_x', vel_w[0])),
        float(_config_value(config, 'solver_velocity_weight_y', vel_w[1])),
        float(_config_value(config, 'solver_velocity_weight_z', vel_w[2])),
    ]

    ctrl_w = solver_cfg['control_weight']
    solver_cfg['control_weight'] = [
        float(_config_value(config, 'solver_control_weight_x', ctrl_w[0])),
        float(_config_value(config, 'solver_control_weight_y', ctrl_w[1])),
        float(_config_value(config, 'solver_control_weight_z', ctrl_w[2])),
    ]

    term_w = solver_cfg['terminal_weight']
    solver_cfg['terminal_weight'] = [
        float(_config_value(config, 'solver_terminal_weight_px', term_w[0])),
        float(_config_value(config, 'solver_terminal_weight_py', term_w[1])),
        float(_config_value(config, 'solver_terminal_weight_pz', term_w[2])),
        float(_config_value(config, 'solver_terminal_weight_vx', term_w[3])),
        float(_config_value(config, 'solver_terminal_weight_vy', term_w[4])),
        float(_config_value(config, 'solver_terminal_weight_vz', term_w[5])),
    ]

    accel_limits = solver_cfg['accel_limits']
    solver_cfg['accel_limits'] = [
        float(_config_value(config, 'solver_accel_limit_x', accel_limits[0])),
        float(_config_value(config, 'solver_accel_limit_y', accel_limits[1])),
        float(_config_value(config, 'solver_accel_limit_z', accel_limits[2])),
    ]

    solver_cfg['regularization'] = float(_config_value(config, 'solver_regularization', solver_cfg['regularization']))
    solver_cfg['codegen_directory'] = str(_config_value(config, 'solver_codegen_directory', solver_cfg.get('codegen_directory', '')))

    vehicle_cfg = params['vehicle']
    inertia = vehicle_cfg['inertia']
    vehicle_cfg['mass'] = float(_config_value(config, 'vehicle_mass', vehicle_cfg['mass']))
    vehicle_cfg['inertia'] = [
        float(_config_value(config, 'vehicle_inertia_xx', inertia[0])),
        float(_config_value(config, 'vehicle_inertia_xy', inertia[1])),
        float(_config_value(config, 'vehicle_inertia_xz', inertia[2])),
        float(_config_value(config, 'vehicle_inertia_yx', inertia[3])),
        float(_config_value(config, 'vehicle_inertia_yy', inertia[4])),
        float(_config_value(config, 'vehicle_inertia_yz', inertia[5])),
        float(_config_value(config, 'vehicle_inertia_zx', inertia[6])),
        float(_config_value(config, 'vehicle_inertia_zy', inertia[7])),
        float(_config_value(config, 'vehicle_inertia_zz', inertia[8])),
    ]
    vehicle_cfg['arm_length'] = float(_config_value(config, 'vehicle_arm_length', vehicle_cfg['arm_length']))
    vehicle_cfg['rotor_force_constant'] = float(_config_value(config, 'vehicle_rotor_force_constant', vehicle_cfg['rotor_force_constant']))
    vehicle_cfg['rotor_moment_constant'] = float(_config_value(config, 'vehicle_rotor_moment_constant', vehicle_cfg['rotor_moment_constant']))
    vehicle_cfg['motor_min_speed'] = float(_config_value(config, 'vehicle_motor_min_speed', vehicle_cfg['motor_min_speed']))
    vehicle_cfg['motor_max_speed'] = float(_config_value(config, 'vehicle_motor_max_speed', vehicle_cfg['motor_max_speed']))
    drag = vehicle_cfg['drag_coefficients']
    vehicle_cfg['drag_coefficients'] = [
        float(_config_value(config, 'vehicle_drag_x', drag[0])),
        float(_config_value(config, 'vehicle_drag_y', drag[1])),
        float(_config_value(config, 'vehicle_drag_z', drag[2])),
    ]

    controller_cfg = params['controller']
    controller_cfg['thrust_limits'] = [
        float(_config_value(config, 'controller_thrust_min', controller_cfg['thrust_limits'][0])),
        float(_config_value(config, 'controller_thrust_max', controller_cfg['thrust_limits'][1])),
    ]
    attitude = controller_cfg.get('attitude_gains', {})
    attitude.setdefault('roll', {})
    attitude.setdefault('pitch', {})
    attitude.setdefault('yaw', {})
    attitude['roll']['kp'] = float(_config_value(config, 'controller_roll_kp', attitude['roll'].get('kp', 4.0)))
    attitude['roll']['kd'] = float(_config_value(config, 'controller_roll_kd', attitude['roll'].get('kd', 1.2)))
    attitude['pitch']['kp'] = float(_config_value(config, 'controller_pitch_kp', attitude['pitch'].get('kp', 4.0)))
    attitude['pitch']['kd'] = float(_config_value(config, 'controller_pitch_kd', attitude['pitch'].get('kd', 1.2)))
    attitude['yaw']['kp'] = float(_config_value(config, 'controller_yaw_kp', attitude['yaw'].get('kp', 1.5)))
    attitude['yaw']['kd'] = float(_config_value(config, 'controller_yaw_kd', attitude['yaw'].get('kd', 0.35)))
    controller_cfg['attitude_gains'] = attitude
    tilt_value = _config_search(config, 'controller_max_tilt_deg')
    if tilt_value is _SENTINEL:
        tilt_value = _config_search(config, 'controller_max_tilt_angle')
    controller_cfg['max_tilt_deg'] = float(tilt_value if tilt_value is not _SENTINEL else controller_cfg.get('max_tilt_deg', 18.0))

    world_cfg = params['world']
    world_cfg['gravity'] = float(_config_value(config, 'world_gravity', world_cfg.get('gravity', 9.81)))

    reference_cfg = params['reference']
    reference_cfg['frame'] = str(_config_value(config, 'reference_frame', reference_cfg.get('frame', 'world')))
    def_pos = reference_cfg['default_position']
    reference_cfg['default_position'] = [
        float(_config_value(config, 'reference_position_x', def_pos[0])),
        float(_config_value(config, 'reference_position_y', def_pos[1])),
        float(_config_value(config, 'reference_position_z', def_pos[2])),
    ]
    def_vel = reference_cfg['default_velocity']
    reference_cfg['default_velocity'] = [
        float(_config_value(config, 'reference_velocity_x', def_vel[0])),
        float(_config_value(config, 'reference_velocity_y', def_vel[1])),
        float(_config_value(config, 'reference_velocity_z', def_vel[2])),
    ]
    def_acc = reference_cfg['default_acceleration']
    reference_cfg['default_acceleration'] = [
        float(_config_value(config, 'reference_acceleration_x', def_acc[0])),
        float(_config_value(config, 'reference_acceleration_y', def_acc[1])),
        float(_config_value(config, 'reference_acceleration_z', def_acc[2])),
    ]
    reference_cfg['default_yaw'] = float(_config_value(config, 'reference_yaw', reference_cfg.get('default_yaw', 0.0)))

    topics_cfg = {
        'state': str(_config_value(config, 'topic_state', params['topics']['state'])),
        'command': str(_config_value(config, 'topic_command', params['topics']['command'])),
        'motor': str(_config_value(config, 'topic_motor', params['topics']['motor'])),
        'reference': str(_config_value(config, 'topic_reference', params['topics']['reference'])),
    }
    params['topics'] = topics_cfg

    node_cfg = params['node']
    node_cfg['rate'] = float(_config_value(config, 'node_rate', node_cfg['rate']))
    node_cfg['max_tilt_deg'] = float(_config_value(config, 'node_max_tilt_deg', node_cfg['max_tilt_deg']))
    node_cfg['yaw_rate_gain'] = float(_config_value(config, 'node_yaw_rate_gain', node_cfg['yaw_rate_gain']))
    node_cfg['yaw_rate_limit'] = float(_config_value(config, 'node_yaw_rate_limit', node_cfg['yaw_rate_limit']))
    node_cfg['log_interval'] = float(_config_value(config, 'node_log_interval', node_cfg['log_interval']))

    reference_defaults = {
        'position': reference_cfg['default_position'],
        'velocity': reference_cfg['default_velocity'],
        'acceleration': reference_cfg['default_acceleration'],
        'yaw': reference_cfg['default_yaw'],
        'frame': reference_cfg['frame'],
    }

    node_meta = {
        'rate': node_cfg['rate'],
        'max_tilt_deg': node_cfg['max_tilt_deg'],
        'yaw_rate_gain': node_cfg['yaw_rate_gain'],
        'yaw_rate_limit': node_cfg['yaw_rate_limit'],
        'log_interval': node_cfg['log_interval'],
    }

    return topics_cfg, reference_defaults, node_meta
