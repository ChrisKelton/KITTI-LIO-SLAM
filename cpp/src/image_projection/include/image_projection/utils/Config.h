//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef KITTI_LIO_SLAM_CONFIG_H
#define KITTI_LIO_SLAM_CONFIG_H

#include <Eigen/Dense>


class Config {
public:
    Config() {};
    ~Config() {};

    std::string lidarFrame;
    std::string imuTopic;
    std::string odomTopic;
    std::string pointCloudTopic;

    uint N_SCAN;
    uint Horizon_SCAN;

    // Extrinsic Rotations between IMU -> LiDAR frame.
    std::vector<double> extRotV;  // 3x3 matrix in vector form
    std::vector<double> extRotRPYV;
    Eigen::Matrix3d extRot;  // 3x3 matrix, IMU acceleration & gyroscope axis transformation to LiDAR frame
    Eigen::Matrix3d extRPY;  // 3x3 matrix, Transformation from the LiDAR frame to the IMU orientation frame
    Eigen::Quaterniond extQRPY;  // 1x4 quaternion, IMU orientation axis transformation to LiDAR frame

    float lidarMinRange;
    float lidarMaxRange;
    uint downsampleRate;
    uint lidarCurvatureFeatureExtractionNeighbors;
};

#endif //KITTI_LIO_SLAM_CONFIG_H