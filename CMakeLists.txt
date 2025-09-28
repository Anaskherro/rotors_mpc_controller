cmake_minimum_required(VERSION 3.0.2)
project(rotors_mpc_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  mav_msgs
  dynamic_reconfigure
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/rotors_mpc.cfg
)

catkin_package()

catkin_install_python(PROGRAMS
  nodes/mpc_controller_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
