//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef KITTI_LIO_SLAM_IMUPREINTEGRATION_H
#define KITTI_LIO_SLAM_IMUPREINTEGRATION_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>


class ImuPreintegrationNode : public rclcpp::Node {
public:
    ImuPreintegrationNode() : Node("imu_preintegration");
};

#endif //KITTI_LIO_SLAM_IMUPREINTEGRATION_H