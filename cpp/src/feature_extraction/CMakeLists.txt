cmake_minimum_required(VERSION 3.10)
project(feature_extraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Path to target Python interpreter" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(slam_msgs REQUIRED)
find_package(slam_utilities REQUIRED)

include_directories(include/lidar ${PCL_INCLUDE_DIRS})

add_executable(feature_extraction
        src/main.cpp
        src/LidarFeatureExtractionNode.cpp
        include/lidar/LidarFeatureExtractionNode.h
        include/lidar/utils/Config.h
)

ament_target_dependencies(feature_extraction
    rclcpp std_msgs sensor_msgs pcl_conversions pcl_ros slam_msgs slam_utilities
)

target_link_libraries(feature_extraction ${PCL_LIBRARIES} Eigen3::Eigen slam_utilities::SlamUtilities)

install(TARGETS feature_extraction
DESTINATION lib/${PROJECT_NAME})

ament_package()
