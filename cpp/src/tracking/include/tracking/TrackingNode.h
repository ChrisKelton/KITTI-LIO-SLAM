//
// Created by ckelton on 6/20/26.
//

#pragma once

#ifndef TRACKING_TRACKINGNODE_H
#define TRACKING_TRACKINGNODE_H

#include <Eigen/Dense>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include "utils/Config.h"

inline auto& Car = gtsam::symbol_shorthand::A;
inline auto& Pedestrian = gtsam::symbol_shorthand::B;

const std::string CarClassName = "Car";
const std::string PedestrianClassName = "Pedestrian";

struct UnpackedDetection {
    Eigen::Vector3f center;
    Eigen::Vector4f orientation;
    Eigen::Vector3f size;
    std::string predClassName;
    float predClassScore;

    UnpackedDetection() = default;

    UnpackedDetection(const Eigen::Vector3f& center_, const Eigen::Vector4f& orientation_, const Eigen::Vector3f& size_, std::string className_, float classScore_) : center(center_), orientation(orientation_), size(size_), predClassName(className_), predClassScore(classScore_) {};
};

inline std::ostream& operator<<(std::ostream& os, const UnpackedDetection& d) {
    std::streamsize original_precision = os.precision();
    std::ios_base::fmtflags original_flags = os.flags();

    os << std::fixed << std::setprecision(2) << "Bounding Box Center - x: " << d.center(0) << ", y: " << d.center(1) << ", z: " << d.center(2) << "\n";
    os << "Size - x: " << d.size(0) << ", y: " << d.size(1) << ", z: " << d.size(2) << "\n";
    os << "Orientation - x: " << d.orientation(0) << ", y: " << d.orientation(1) << ", z: " << d.orientation(2) << ", w: " << d.orientation(3);

    os.precision(original_precision);
    os.flags(original_flags);

    return os;
}

class TrackingNode : public rclcpp::Node {
public:
    TrackingNode();
    ~TrackingNode() override;

private:
    std::unique_ptr<Config> config;

    // gtsam
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    gtsam::Values currentEstimate;
    std::unique_ptr<gtsam::ISAM2> isam2;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr subDetections3d;

    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pubTrackedDetections3d;

    // std::unordered_map<int, gtsam::Key> activeTracks;
    std::vector<gtsam::Key> activeCarTrackSymbols;
    std::vector<int> activeCarTrackIds;
    std::vector<UnpackedDetection> activeCarDetections;
    std::vector<Eigen::Vector3f> currentCarTrackPositions;
    std::vector<gtsam::Key> activePedestrianTrackSymbols;
    std::vector<int> activePedestrianTrackIds;
    std::vector<UnpackedDetection> activePedestrianDetections;
    std::vector<Eigen::Vector3f> currentPedestrianTrackPositions;
    int carHeadCnt = 0;
    int pedestrianHeadCnt = 0;
    std::unordered_map<gtsam::Key, int> missedCounters;

    gtsam::noiseModel::Isotropic::shared_ptr priorNoise;
    gtsam::noiseModel::Isotropic::shared_ptr measNoise;

    DataAssociationCost dataAssociationCost;
    std::vector<UnpackedDetection> currentDetections;

    void setup_config();
    void initialize();
    void setup_subpub();

    void detectionsHandler(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg);
    void associateData(vision_msgs::msg::Detection3DArray::ConstSharedPtr msg, std::vector<std::pair<gtsam::Key, int>>& assignments, std::vector<int>& unassigned);
    std::unordered_map<gtsam::Key, Eigen::Vector3f> getCarTrackPositions();
    std::unordered_map<gtsam::Key, Eigen::Vector3f> getPedestrianTrackPositions();
    void pruneLostTracks(std::vector<gtsam::Key> tracksToRemove);

    void pruneFromSymbol(const gtsam::Key& symbol);
    void getActiveInfoFromSymbol(const gtsam::Key& symbol, int& trackId, UnpackedDetection& detection, int& idx, std::string& detectionType);
    void getCarIndexFromSymbol(const gtsam::Key& symbol, int& idx);
    void getPedestrianIndexFromSymbol(const gtsam::Key& symbol, int& idx);
    void getCarIndexFromTrackId(const int& trackId, int& idx);
    void getPedestrianIndexFromTrackId(const int& trackId, int& idx);

    void publishTrackedDetections(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& incomingMsg) const;
};


#endif //TRACKING_TRACKINGNODE_H
