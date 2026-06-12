cmake_minimum_required(VERSION 3.10)
project(slam_utilities)

set(CMAKE_CXX_STANDARD 17)
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Path to target Python interpreter" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(slam_msgs REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

add_library(SlamUtilities INTERFACE)

target_include_directories(SlamUtilities INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(SlamUtilities INTERFACE
    rclcpp sensor_msgs tf2 tf2_geometry_msgs pcl_conversions slam_msgs
)

target_link_libraries(SlamUtilities INTERFACE
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    gtsam
    Boost::serialization
)

install(TARGETS SlamUtilities EXPORT SlamUtilitiesTargets)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(SlamUtilitiesTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp sensor_msgs tf2 tf2_geometry_msgs pcl_conversions slam_msgs PCL Eigen3 GTSAM Boost)

ament_package()
