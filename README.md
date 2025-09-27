# rotors_mpc_controller

A ROS package that provides a nonlinear model predictive controller (NMPC) for quadrotor trajectory tracking in Gazebo. The controller uses [acados](https://github.com/acados/acados) for fast optimal control, and interfaces with the [RotorS simulator](https://github.com/ethz-asl/rotors_simulator) (included as a git submodule).

The package targets ROS Noetic on Ubuntu 20.04 and has been tested against the RotorS "hummingbird" configuration. It publishes roll/pitch/yaw-rate/thrust commands that are converted to motor speeds via a lightweight PID mixer.

## Features

- Translational NMPC solved with acados and CasADi.
- Automatic conversion from desired world-frame acceleration to attitude and thrust via thrust-vector alignment.
- Warm-start caching with fail-safe reuse of the last valid command.
- Periodic logging of the internal NMPC state, references, and final commands.
- Configurable parameters via YAML for solver weights, limits, and low-level attitude gains.
- Gazebo launch integration that spawns the RotorS hummingbird with both the NMPC and low-level controller.

## Repository layout

```
rotors_mpc_controller/
├── config/               # YAML configuration (solver, vehicle, topics)
├── launch/               # Example launch files
├── nodes/                # ROS node entry points (Python)
├── src/rotors_mpc_controller/
│   ├── controller.py     # acados-based NMPC core
│   ├── low_level.py      # PID rotor mixer utilities
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
git clone --recurse-submodules https://github.com/<your-user>/rotors_mpc_controller.git
cd rotors_mpc_controller
# If you cloned without --recurse-submodules, run:
#   git submodule update --init --recursive

# Install RotorS dependencies (from their README)
rosdep install --from-paths third_party/rotors_simulator --ignore-src -r -y

cd ~/catkin_ws
catkin build rotors_mpc_controller
source devel/setup.bash
```

> **Note**: The first time you run the NMPC node, acados will generate solver code in `~/.cache/rotors_mpc_controller/acados`. If you change the horizon length or discretization in the YAML, delete that directory so acados regenerates the solver with the new settings.

## Running the simulator

```bash
roslaunch rotors_mpc_controller hummingbird_mpc.launch
```

This launch file will:

1. Start Gazebo with the RotorS world.
2. Spawn the hummingbird vehicle.
3. Launch `mpc_controller_node` (NMPC) and `low_level_controller_node` (PID mixer).

You can command a new setpoint by publishing a `geometry_msgs/PoseStamped` to `/mpc_controller/setpoint`. Example:

```bash
rostopic pub /mpc_controller/setpoint geometry_msgs/PoseStamped "header: {frame_id: world}
pose: {position: {x: 2.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}" -1
```

## Configuration

All runtime parameters live in [`config/params.yaml`](config/params.yaml). Key sections:

- `solver`: NMPC horizon, discretization, quadratic weights, acceleration limits, damping.
- `vehicle`: mass, inertia, arm length, rotor constants, motor speed limits.
- `controller`: thrust limits and attitude PID gains.
- `reference`: default hold position/velocity/yaw.
- `topics`: ROS topic names for state, command, and motors.
- `node`: execution rate, max tilt, yaw-rate controller gains, and logging interval.

Changes take effect on restart. When altering horizon or dt, delete `~/.cache/rotors_mpc_controller/acados` to force solver regeneration.

## Logging and debugging

Every `node.log_interval` seconds (default 3 s) the MPC node prints:

```
MPC log: status=<acados_status> pos=<current> vel=<current_vel>
         ref_pos=<ref_position> ref_vel=<ref_velocity> ref_acc=<ref_acc>
         acc_cmd=<solver_output> command=<roll pitch yaw_rate thrust>
```

Use this to diagnose divergence, saturation, or solver failures. If acados returns a non-zero status, the node reuses the last valid command and logs the snapshot so you can inspect what happened.

## Adding as a dependency

To use the controller in another catkin workspace:

```bash
cd ~/catkin_ws/src
git submodule add https://github.com/<your-user>/rotors_mpc_controller.git rotors_mpc_controller
cd rotors_mpc_controller
git submodule update --init --recursive
```

Then follow the build steps above.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).

## Contributing

Pull requests and issues are welcome. Please open an issue describing the change before submitting large feature PRs.

