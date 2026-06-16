//
// Created by ckelton on 6/11/26.
//
#pragma once

#ifndef MAP_OPTIMIZATION_CONFIG_H
#define MAP_OPTIMIZATION_CONFIG_H

#include <cstdint>
#include <string>


class Config {
public:
    Config() {}
    ~Config() {}

    int numberOfCores = 2;

    std::string gpsTopic;
    bool savePCD;
    std::string savePCDDirectory;
    // Selects the scan2MapOptimization back-end: true -> GTSAM LevenbergMarquardtOptimizer,
    // false -> the original hand-rolled Gauss-Newton/Levenberg solver.
    bool useGtsamScanMatcher;
    std::string odometryFrame;
    std::string lidarFrame;

    float mappingCornerLeafSize;
    float mappingSurfLeafSize;
    double mappingProcessInterval;

    uint32_t N_SCAN;
    uint32_t Horizon_SCAN;

    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    float surroundingKeyframeAddingDistThreshold;
    float surroundingKeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    float imuRPYWeight;
    float z_tolerance;
    float rotation_tolerance;
};

#endif //MAP_OPTIMIZATION_CONFIG_H