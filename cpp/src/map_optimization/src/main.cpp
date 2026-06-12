//
// Created by ckelton on 6/12/26.
//

#include <rclcpp/rclcpp.hpp>

#include "MapOptimizationNode.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapOptimizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}