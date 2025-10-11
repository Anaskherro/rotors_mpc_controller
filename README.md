# rotors_mpc_controller

A ROS package that provides a nonlinear model predictive controller (NMPC) for quadrotor trajectory tracking in Gazebo. The controller uses [acados](https://github.com/acados/acados) for fast optimal control, and interfaces with the [RotorS simulator](https://github.com/ethz-asl/rotors_simulator) (included as a git submodule).

The package targets ROS Noetic on Ubuntu 20.04 and has been tested against the RotorS "hummingbird" configuration. The NMPC now drives the four rotor thrusts directly, so no auxiliary mixer is required.

## Features

- Rotor-level NMPC solved with acados and CasADi using a 13-state quadrotor model.
- Direct mapping from solver outputs to rotor speeds with saturation handling.
- Warm-start caching with fail-safe reuse of the last valid command.
- Periodic logging of the internal NMPC state, references, and final thrust commands.
- Configurable parameters via YAML for solver weights, vehicle properties, and thrust limits.
- Live tuning through `dynamic_reconfigure` (`rqt_reconfigure`) with automatic acados solver regeneration whenever solver parameters change.
- Gazebo launch integration that spawns the RotorS hummingbird and the NMPC node.

## Repository layout

```
rotors_mpc_controller/
├── cfg/                  # dynamic_reconfigure definitions
├── config/               # YAML configuration (solver, vehicle, topics)
├── launch/               # Example launch files
├── nodes/                # ROS node entry points (Python)
├── src/rotors_mpc_controller/
│   ├── controller.py     # acados-based NMPC core
│   ├── params.py         # Parameter loading & validation
│   └── reference.py      # Reference management utilities
├── third_party/
│   └── rotors_simulator/ # RotorS simulator (git submodule)
├── README.md
├── LICENSE
└── .gitmodules
```

## Prerequisites

- **ROS Noetic** (tested on Ubuntu 20.04). Make sure your workspace is initialized: `source /opt/ros/noetic/setup.bash`.
- **acados** built with Python bindings. Follow the official [acados installation guide](https://docs.acados.org/installation/index.html). Ensure that `ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` refer to your acados build, and that `acados_template` is on `PYTHONPATH`.
- **CasADi** ≥ 3.6.6 is recommended by acados. Noetic ships 3.5.1 by default; consider installing a newer wheel in a virtual environment if you need the speed-ups.
- Python dependencies: the nodes rely only on the ROS Noetic Python stack and NumPy (already a ROS dependency).

## Clone and build

```bash
cd ~/catkin_ws/src
# Clone this repository and pull submodules
git clone --recurse-submodules https://github.com/anaskherro/rotors_mpc_controller.git
cd rotors_mpc_controller
# If you cloned without --recurse-submodules, run:
#   git submodule update --init --recursive

# Install RotorS dependencies (from their README)
rosdep install --from-paths third_party/rotors_simulator --ignore-src -r -y

cd ~/catkin_ws
catkin build rotors_mpc_controller
source devel/setup.bash
```

> **Note**: The first time you run the NMPC node, acados will generate solver code in `~/.cache/rotors_mpc_controller/acados`. Subsequent parameter changes made via `rqt_reconfigure` or YAML edits automatically rebuild the solver; you no longer need to delete the cache manually.

## Running the simulator

```bash
roslaunch rotors_mpc_controller hummingbird_mpc.launch
```

This launch file will:

1. Start Gazebo with the RotorS world.
2. Spawn the hummingbird vehicle.
3. Launch `mpc_controller_node`, which now maps solver thrusts directly to motor speeds.

You can command a new setpoint by publishing a `geometry_msgs/PoseStamped` to `/mpc_controller/setpoint`. Example:

```bash
rostopic pub /mpc_controller/setpoint geometry_msgs/PoseStamped "header: {frame_id: world}
pose: {position: {x: 2.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}" -1
```

## Configuration

All runtime parameters live in [`config/params.yaml`](config/params.yaml). Key sections:

- `solver`: NMPC horizon, discretization, quadratic weights, and regularisation.
- `vehicle`: mass, inertia, arm length, rotor constants, motor speed limits.
- `controller`: rotor thrust limits.
- `reference`: default hold position/velocity/yaw.
- `topics`: ROS topic names for state, motor speeds, and setpoint input.
- `node`: execution rate and logging interval.

Shipped defaults (also pre-populated in `rqt_reconfigure`) now match the tuned rotor-level setup:

- `horizon_steps = 20`, `dt = 0.05 s`, `iter_max = 600`, `regularization = 7e-3`.
- Position weights `[10, 10, 8]`, velocity weights `[1, 1, 0.2]`, quaternion weights `[3.2 × 4]`, rate weights `[1.4, 1.4, 0.4]`.
- Control penalty `1.75` per rotor and terminal weights `[5, 5, 3, 2, 2, 2, 12, 12, 12, 18.5, 2, 2, 1.8]`.
- Thrust window `[4, 20]` N per motor; the default reference holds the vehicle at `(0, 0, 1)` with zero velocity.

Changes take effect immediately when adjusted through `rqt_reconfigure`. For YAML edits, the node rebuilds the solver during startup using the updated values.

## Live tuning with dynamic reconfigure

The controller exposes all solver, vehicle, controller, reference, and topic settings through ROS dynamic reconfigure. To adjust parameters on the fly:

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

Select the `rotors_mpc_controller` namespace and tweak the sliders. Solver-related changes regenerate the acados model instantly, while other entries update the running controller without a restart. The `solver_horizon_steps` and `solver_iter_max` sliders now span up to 600 to simplify large-horizon experiments.

## Logging and debugging

Every `node.log_interval` seconds (default 3 s) the MPC node prints:

```
MPC log: status=<acados_status> pos=<current> vel=<current_vel> quat=<current_quat>
         ref_pos=<ref_position> ref_vel=<ref_velocity> ref_quat=<ref_quaternion>
         ref_thrust=<reference_thrust> cmd=<clipped_thrust>
```

Use this to diagnose divergence, saturation, or solver failures. If acados returns a non-zero status, the node reuses the last valid command and logs the snapshot so you can inspect what happened.

## Adding as a dependency

To use the controller in another catkin workspace:

```bash
cd ~/catkin_ws/src
git submodule add https://github.com/anaskherro/rotors_mpc_controller.git rotors_mpc_controller
cd rotors_mpc_controller
git submodule update --init --recursive
```

Then follow the build steps above.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).

## Contributing

Pull requests and issues are welcome. Please open an issue describing the change before submitting large feature PRs.
