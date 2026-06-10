//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef KITTI_LIO_SLAM_CONFIG_H
#define KITTI_LIO_SLAM_CONFIG_H

class Config {
public:
    Config() {};
    ~Config() {};

    // Velodyne sensor configuration
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // Voxel filter params
    float surfLeafSize;
};

#endif //KITTI_LIO_SLAM_CONFIG_H