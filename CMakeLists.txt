cmake_minimum_required(VERSION 3.0.2)
project(rotors_mpc_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  mav_msgs
  std_srvs
  dynamic_reconfigure
  rqt_gui
  rqt_gui_py
  python_qt_binding
  visualization_msgs
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

install(FILES
  resource/mpc_control_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resource
)
