cmake_minimum_required(VERSION 3.10)
project(feature_extraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(include/lidar ${PCL_INCLUDE_DIRS})

add_executable(feature_extraction
        src/object_detection_node.cpp
        src/LidarFeatureExtractionNode.cpp
        include/lidar/LidarFeatureExtractionNode.h
        include/lidar/utils/Config.h
)

ament_target_dependencies(feature_extraction
    rclcpp std_msgs sensor_msgs pcl_conversions pcl_ros
)

target_include_directories(feature_extraction PRIVATE /mnt/c/Users/cblim/Documents/Misc/code/DBScan-PCL-Optimized/dbscan/include)
target_link_libraries(feature_extraction ${PCL_LIBRARIES} Eigen3::Eigen dbscan)

install(TARGETS feature_extraction
DESTINATION lib/${PROJECT_NAME})

ament_package()
