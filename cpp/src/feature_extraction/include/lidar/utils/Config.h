//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef FEATURE_EXTRACTION_CONFIG_H
#define FEATURE_EXTRACTION_CONFIG_H

#include <cstdint>
#include <string>

class Config {
public:
    Config() {}
    ~Config() {}

    std::string lidarFrame;

    // Velodyne sensor configuration
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // Voxel filter params
    float surfLeafSize;
    float edgeThreshold;
    float surfThreshold;

    std::uint32_t lidarCurvatureFeatureExtractionNeighbors;
    std::uint32_t pixelDiffTh;
};

#endif //FEATURE_EXTRACTION_CONFIG_H
