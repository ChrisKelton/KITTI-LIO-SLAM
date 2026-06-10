cmake_minimum_required(VERSION 3.10)
project(slam_msgs)

set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Path to target Python interpreter" FORCE)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/CloudInfo.msg"
    DEPENDENCIES std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs sensor_msgs)

ament_package()
