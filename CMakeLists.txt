cmake_minimum_required(VERSION 3.0.2)
project(hhi_force)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  iiwa_tools
  xela_server_ros
)

## Declare ROS messages, services, and actions
# Uncomment if you have custom messages or services
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )
# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   sensor_msgs
# )

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS iiwa_tools roscpp rospy sensor_msgs std_msgs xela_server_ros
)

## Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare C++ executables and link libraries
# Uncomment and modify if you have C++ nodes
# add_executable(node_name src/node_name.cpp)
# target_link_libraries(node_name ${catkin_LIBRARIES})
# add_dependencies(node_name ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# Mark executable scripts (Python etc.) for installation
# in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  nodes/xela_test.py
  nodes/AB_Pub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install rules
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

