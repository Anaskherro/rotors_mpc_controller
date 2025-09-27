"""Parameter loading utilities for the redesigned ACADOS-based controller."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


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
    cfg['horizon_steps'] = int(cfg['horizon_steps'])
    cfg['dt'] = float(cfg['dt'])
    cfg['position_weight'] = [float(v) for v in cfg.get('position_weight', [10.0, 10.0, 10.0])]
    cfg['velocity_weight'] = [float(v) for v in cfg.get('velocity_weight', [2.0, 2.0, 2.0])]
    cfg['control_weight'] = [float(v) for v in cfg.get('control_weight', [0.5, 0.5, 0.5])]
    cfg['terminal_weight'] = [float(v) for v in cfg.get('terminal_weight',
                                                       [20.0, 20.0, 30.0, 5.0, 5.0, 7.0])]
    cfg['accel_limits'] = [float(v) for v in cfg.get('accel_limits', [6.0, 6.0, 6.0])]
    cfg['regularization'] = float(cfg.get('regularization', 1e-6))
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


def _coerce_controller(cfg: Dict[str, Any]) -> None:
    thrust_limits = cfg.get('thrust_limits', [2.0, 25.0])
    if len(thrust_limits) != 2:
        raise ValueError('controller.thrust_limits must contain [min, max].')
    cfg['thrust_limits'] = [float(thrust_limits[0]), float(thrust_limits[1])]

    gains = cfg.get('attitude_gains', {})
    for axis in ('roll', 'pitch', 'yaw'):
        gains.setdefault(axis, {})
        axis_cfg = gains[axis]
        axis_cfg['kp'] = float(axis_cfg.get('kp', 8.0 if axis != 'yaw' else 1.0))
        axis_cfg['kd'] = float(axis_cfg.get('kd', 2.0 if axis != 'yaw' else 0.2))
    cfg['attitude_gains'] = gains


def _coerce_world(cfg: Dict[str, Any]) -> None:
    cfg['gravity'] = float(cfg.get('gravity', 9.81))


def _ensure_required(cfg: Dict[str, Any]) -> None:
    required = {'solver', 'vehicle', 'controller', 'world', 'reference', 'topics'}
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

    return base
