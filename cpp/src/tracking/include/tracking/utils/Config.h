//
// Created by ckelton on 6/20/26.
//
#pragma once

#ifndef TRACKING_CONFIG_H
#define TRACKING_CONFIG_H

enum DataAssociationCost {
    L2,  // 0
    SquaredEuclidean,  // 1
    Manhattan,  // 2
};

inline std::ostream& operator<<(std::ostream& os, DataAssociationCost cost_) {
    switch (cost_) {
        case DataAssociationCost::L2:               return os << "L2";
        case DataAssociationCost::SquaredEuclidean: return os << "SquaredEuclidean";
        case DataAssociationCost::Manhattan:        return os << "Manhattan";
        default:                                    return os << "UNKNOWN";
    }
}


class Config {
public:
    Config() {};
    ~Config() {};

    int maxMissedFrames;
    float priorNoise;
    float measNoise;

    int dataAssociationCostFlag;
    float dataAssociationGateTh;
};


#endif //TRACKING_CONFIG_H
