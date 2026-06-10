cmake_minimum_required(VERSION 3.10)
project(image_projection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Path to target Python interpreter" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)

find_package(slam_msgs REQUIRED)
find_package(slam_utilities REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(OpenCV REQUIRED)

include_directories(include/image_projection ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(image_projection
    src/main.cpp
        include/image_projection/ImageProjectionNode.h
        include/image_projection/utils/Config.h
)

ament_target_dependencies(image_projection
    rclcpp std_msgs sensor_msgs nav_msgs pcl_conversions pcl_ros tf2 tf2_geometry_msgs slam_msgs slam_utilities
)

target_link_libraries(image_projection ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen slam_utilities::SlamUtilities)

install(TARGETS image_projection
DESTINATION lib/${PROJECT_NAME})

ament_package()
