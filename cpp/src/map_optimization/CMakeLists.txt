cmake_minimum_required(VERSION 3.10)
project(map_optimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Path to target Python interpreter" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(slam_msgs REQUIRED)
find_package(slam_srvs REQUIRED)
find_package(slam_utilities REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTSAM REQUIRED)

find_package(OpenCV REQUIRED)

# Without this, the `#pragma omp parallel for` directives in scan matching
# (cornerOptimization/surfOptimization, 30 iters/frame) and transformPointCloud
# compile to serial loops, so numberOfCores has no effect. Required for real-time
# scan-to-map registration on dense (64-beam) clouds.
find_package(OpenMP REQUIRED)

include_directories(include/map_optimization ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(map_optimization
    src/main.cpp
    src/MapOptimizationNode.cpp
    include/map_optimization/MapOptimizationNode.h
    include/map_optimization/utils/Config.h
)

ament_target_dependencies(map_optimization
    rclcpp std_msgs sensor_msgs nav_msgs visualization_msgs tf2_ros slam_msgs slam_srvs slam_utilities
)

target_link_libraries(map_optimization ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen slam_utilities::SlamUtilities gtsam OpenMP::OpenMP_CXX)

install(TARGETS map_optimization
DESTINATION lib/${PROJECT_NAME})

ament_package()
