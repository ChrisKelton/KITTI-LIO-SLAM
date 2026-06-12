//
// Created by ckelton on 6/11/26.
//
#pragma once

#ifndef IMU_IMUPREINTEGRATIONNODECONFIG_H
#define IMU_IMUPREINTEGRATIONNODECONFIG_H

#include <string>
#include <vector>

#include <Eigen/Dense>


class Config {
public:
    Config() {}
    ~Config() {}

    std::string imuTopic;
    std::string odomTopic;

    std::string odometryFrame;

    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;

    float imuGravity;

    // Extrinsic Rotations between IMU -> LiDAR frame.
    std::vector<double> extRotV;  // 3x3 matrix in vector form
    std::vector<double> extRotRPYV;
    std::vector<double> extTransV;
    Eigen::Matrix3d extRot;  // 3x3 matrix, IMU acceleration & gyroscope axis transformation to LiDAR frame
    Eigen::Matrix3d extRPY;  // 3x3 matrix, Transformation from the LiDAR frame to the IMU orientation frame
    Eigen::Quaterniond extQRPY;  // 1x4 quaternion, IMU orientation axis transformation to LiDAR frame
    Eigen::Vector3d extTrans;

    int reset_graph_every_n_keyframes;
};

#endif //IMU_IMUPREINTEGRATIONNODECONFIG_H