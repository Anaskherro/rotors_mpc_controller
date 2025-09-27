cmake_minimum_required(VERSION 3.0.2)
project(rotors_mpc_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  mav_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  nodes/mpc_controller_node
  nodes/low_level_controller_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
