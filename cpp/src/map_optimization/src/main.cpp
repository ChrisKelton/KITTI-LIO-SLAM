//
// Created by ckelton on 6/12/26.
//

#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "MapOptimizationNode.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapOptimizationNode>();

    std::thread loopClosureThread(&MapOptimizationNode::loopClosureThread, node);
    std::thread visualizeMapThread(&MapOptimizationNode::visualizeGlobalMapThread, node);

    rclcpp::spin(node);
    rclcpp::shutdown();

    loopClosureThread.join();
    visualizeMapThread.join();
    return EXIT_SUCCESS;
}
