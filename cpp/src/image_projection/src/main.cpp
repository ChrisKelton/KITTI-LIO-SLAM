//
// Created by ckelton on 6/9/26.
//
#include <rclcpp/rclcpp.hpp>

#include "ImageProjectionNode.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageProjectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
