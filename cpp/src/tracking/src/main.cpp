//
// Created by ckelton on 6/20/26.
//
#include <rclcpp/rclcpp.hpp>

#include "TrackingNode.h"


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrackingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
