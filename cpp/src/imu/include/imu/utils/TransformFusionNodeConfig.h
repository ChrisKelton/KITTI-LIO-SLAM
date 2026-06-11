//
// Created by ckelton on 6/10/26.
//
#pragma once

#ifndef IMU_TRANSFORMFUSIONNODECONFIG_H
#define IMU_TRANSFORMFUSIONNODECONFIG_H

#include <string>

class TransformFusionNodeConfig {
public:
    std::string lidarFrame;
    std::string baselinkFrame;
    std::string mapFrame;
    std::string odometryFrame;

    std::string odomTopic;
};

#endif //IMU_TRANSFORMFUSIONNODECONFIG_H