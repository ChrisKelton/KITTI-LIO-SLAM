//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef IMU_IMUPREINTEGRATIONNODE_H
#define IMU_IMUPREINTEGRATIONNODE_H

#include <deque>
#include <memory>
#include <mutex>

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "slam_utilities/State.h"
#include "utils/ImuPreintegrationNodeConfig.h"



class ImuPreintegrationNode final : public rclcpp::Node {
public:
    ImuPreintegrationNode();
    ~ImuPreintegrationNode() override;

private:
    // unique_ptr so the config is freed automatically (was a raw pointer the destructor never deleted).
    std::unique_ptr<Config> config;

    std::mutex mtx;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    bool systemInitialized = false;
    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;

    // ISAM2 Solver — unique_ptr so each resetOptimization() frees the previous solver. The raw
    // pointers leaked one ISAM2 + one graph on every reset (init, every N keyframes, every failure).
    std::unique_ptr<gtsam::ISAM2> isam2;
    // Nonlinear factor graph to add new factors
    std::unique_ptr<gtsam::NonlinearFactorGraph> graph;
    // New variables to add to the factor graph
    gtsam::Values values_new;
    // Estimated new states of our variable nodes
    gtsam::Values values_est;

    // IMU Preintegration
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> preint_params;
    std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> preint_gtsam_opt;
    std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> preint_gtsam_raw;
    // All Imu messages without care for keyframes from lidar, I think just for comparing between the predicted states
    // of optimizing the path and unoptimized path.
    std::deque<sensor_msgs::msg::Imu> imuMsgsRaw;
    // Imu messages within the keyframe specifically
    std::deque<sensor_msgs::msg::Imu> imuMsgsOpt;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    slam::State prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    slam::State prevStateOdom;
    // gtsam::imuBias::ConstantBias prevBiasOdom;

    // T_bl: transform points from lidar frame to imu/body frame
    gtsam::Pose3 imu2Lidar;
    // T_lb: transform points from imu/body frame to lidar frame
    gtsam::Pose3 lidar2Imu;

    // 1-based, b/c key 0 is reserved for the priors from the lidarPose when initializing the system
    int key = 1;

    const double delta_t = 0;

    // ***********************************************************
    // Constructor Functions
    // ***********************************************************
    void setup_config();

    void initialize();

    void initialize_noise();

    void initialize_preint_params();

    void setup_subpub();

    // ***********************************************************
    // Reset Functions
    // ***********************************************************
    void resetOptimization() {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        // Assigning to the unique_ptrs frees the previously held solver/graph automatically.
        isam2 = std::make_unique<gtsam::ISAM2>(optParameters);

        graph = std::make_unique<gtsam::NonlinearFactorGraph>();

        const gtsam::Values newGraphValues;
        values_new = newGraphValues;
    }

    void resetParams() {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    // ***********************************************************
    // Pub/Sub Functions
    // ***********************************************************
    void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);

    void imuHandler(const sensor_msgs::msg::Imu::ConstSharedPtr& msg);

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur) const;
};

#endif //IMU_IMUPREINTEGRATIONNODE_H