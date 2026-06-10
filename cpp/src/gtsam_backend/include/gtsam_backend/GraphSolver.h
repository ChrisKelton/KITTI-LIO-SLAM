//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef KITTI_LIO_SLAM_GRAPHSOLVER_H
#define KITTI_LIO_SLAM_GRAPHSOLVER_H

// Graphs
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <rclcpp/rclcpp.hpp>

using gtsam::symbol_shorthand::X;  // Pose3 (x, y, z, r, p, y)
using gtsam::symbol_shorthand::V;  // Vel   (xdot, ydot, zdot)
using gtsam::symbol_shorthand::B;  // Bias  (ax, ay, az, gx, gy, gz)

class GraphSolver {
public:
    GraphSolver(bool isam_enable_detailed_results = false) {
        this->graph = new gtsam::NonlinearFactorGraph();

        // ISAM2 solver
        gtsam::ISAM2Params isam_params;
        isam_params.relinearizeThreshold = 0.01;
        isam_params.relinearizeSkip = 1;
        isam_params.cacheLinearizedFactors = false;
        isam_params.enabledDetailedResults = isam_enable_detailed_results;
        isam_params.print();
        this->isam2 = new gtsam::ISAM2(isam_params);
    };
};

#endif //KITTI_LIO_SLAM_GRAPHSOLVER_H