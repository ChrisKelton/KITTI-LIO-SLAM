cmake_minimum_required(VERSION 3.10)
project(tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Path to target Python interpreter" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(GTSAM REQUIRED)

include_directories(include/tracking)

add_executable(tracking
    src/main.cpp
    src/TrackingNode.cpp
    include/tracking/TrackingNode.h
    include/tracking/utils/Config.h
)

ament_target_dependencies(tracking
    rclcpp std_msgs vision_msgs geometry_msgs
)

target_link_libraries(tracking gtsam)

install(TARGETS tracking
DESTINATION lib/${PROJECT_NAME})

ament_package()
