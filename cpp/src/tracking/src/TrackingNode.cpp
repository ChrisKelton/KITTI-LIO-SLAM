//
// Created by ckelton on 6/20/26.
//
#include <iostream>
#include <sstream>
#include <iomanip>

#include <gtsam/geometry/Point3.h>

#include <geometry_msgs/msg/point.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "TrackingNode.h"


TrackingNode::TrackingNode() : Node("tracking") {
    config = std::make_unique<Config>();
    setup_config();
    initialize();
    setup_subpub();

    RCLCPP_INFO(this->get_logger(), "Setup TrackingNode");
}

TrackingNode::~TrackingNode() {}

void TrackingNode::setup_config() {
    config->maxMissedFrames = this->declare_parameter("maxMissedFrames", 5);

    config->priorNoise = this->declare_parameter("priorNoise", 1.0);
    config->measNoise = this->declare_parameter("measNoise", 0.2);

    config->dataAssociationCostFlag = this->declare_parameter("dataAssociationCostFlag", 0);
    dataAssociationCost = static_cast<DataAssociationCost>(config->dataAssociationCostFlag);
    config->dataAssociationGateTh = this->declare_parameter("dataAssociationGateTh", 3.0);
}

void TrackingNode::initialize() {
    gtsam::ISAM2Params isam2_params;
    isam2_params.relinearizeThreshold = 0.1;
    isam2_params.relinearizeSkip = 1;
    isam2 = std::make_unique<gtsam::ISAM2>(isam2_params);

    priorNoise = gtsam::noiseModel::Isotropic::Sigma(3, config->priorNoise);
    measNoise = gtsam::noiseModel::Isotropic::Sigma(3, config->measNoise);
}

void TrackingNode::setup_subpub() {
    subDetections3d = this->create_subscription<vision_msgs::msg::Detection3DArray>("detections3d", 1, std::bind(&TrackingNode::detectionsHandler, this, std::placeholders::_1));

    pubTrackedDetections3d = this->create_publisher<vision_msgs::msg::Detection3DArray>("tracked_detections", 1);
}

void TrackingNode::detectionsHandler(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& msg) {
    // 1. Run Data Association
    std::vector<std::pair<gtsam::Key, int>> assignments;
    std::vector<int> unassigned;
    // assignments will be (activeTrackIds[car/ped[i]], msg->detections[j])
    associateData(msg, assignments, unassigned);

    std::vector<gtsam::Key> assignmentKeys;
    assignmentKeys.reserve(assignments.size());
    for (const auto& [key, vals] : assignments) {
        assignmentKeys.push_back(key);
    }

    // 2. Process Successful Assignments (Matches)
    for (const auto& [symbol, meas_idx] : assignments) {
        int idx;
        int track_id;
        UnpackedDetection activeDetection;
        std::string detectionType;
        getActiveInfoFromSymbol(symbol, track_id, activeDetection, idx, detectionType);
        auto detection = currentDetections[meas_idx];

        // Reset the missed frame counter b/c the object was seen
        missedCounters[symbol] = 0;

        // Add measurement to graph
        gtsam::Point3 meas_point(detection.center(0), detection.center(1), detection.center(2));
        auto factor = gtsam::PriorFactor<gtsam::Point3>(symbol, meas_point, measNoise);
        graph.push_back(factor);
        initialEstimate.insert(symbol, meas_point);

        if (detectionType == "car") {
            activeCarDetections[idx] = detection;
        } else if (detectionType == "pedestrian") {
            activePedestrianDetections[idx] = detection;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Got unsupported detectionType '%s'.", detectionType.c_str());
        }
    }

    // 3. Process Unmatched Detections (New Tracks)
    for (const auto& meas_idx : unassigned) {
        auto detection = currentDetections[meas_idx];
        int new_track_id;
        gtsam::Key new_symbol;
        if (detection.predClassName == CarClassName) {
            new_track_id = carHeadCnt;
            new_symbol = Car(new_track_id);
            activeCarTrackIds.push_back(new_track_id);
            activeCarTrackSymbols.push_back(new_symbol);
            activeCarDetections.push_back(detection);
            carHeadCnt++;
        } else if (detection.predClassName == PedestrianClassName) {
            new_track_id = pedestrianHeadCnt;
            new_symbol = Pedestrian(new_track_id);
            activePedestrianTrackIds.push_back(new_track_id);
            activePedestrianTrackSymbols.push_back(new_symbol);
            activePedestrianDetections.push_back(detection);
            pedestrianHeadCnt++;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Encountered unexpected class name '%s'. Defaulting to 'Car'.", detection.predClassName.c_str());
            new_track_id = carHeadCnt;
            new_symbol = Car(new_track_id);
            activeCarTrackIds.push_back(new_track_id);
            activeCarTrackSymbols.push_back(new_symbol);
            activeCarDetections.push_back(detection);
            carHeadCnt++;
        }
        // initialize to -1, b/c they will be counted as 'missed' in the next stage
        missedCounters[new_symbol] = -1;

        // Add prior to graph
        gtsam::Point3 meas_point(detection.center(0), detection.center(1), detection.center(2));
        auto prior = gtsam::PriorFactor<gtsam::Point3>(new_symbol, meas_point, priorNoise);
        graph.push_back(prior);
        initialEstimate.insert(new_symbol, meas_point);
    }

    // 4. Handle active tracks that missed this frame
    std::vector<gtsam::Key> tracksToRemove;
    for (const auto& symbol : activeCarTrackSymbols) {
        if (auto it = std::find(assignmentKeys.begin(), assignmentKeys.end(), symbol); it == assignmentKeys.end()) {
            missedCounters[symbol]++;

            if (missedCounters[symbol] >= config->maxMissedFrames) {
                tracksToRemove.push_back(symbol);
            }
        }
    }
    for (const auto& symbol : activePedestrianTrackSymbols) {
        if (auto it = std::find(assignmentKeys.begin(), assignmentKeys.end(), symbol); it == assignmentKeys.end()) {
            missedCounters[symbol]++;

            if (missedCounters[symbol] >= config->maxMissedFrames) {
                tracksToRemove.push_back(symbol);
            }
        }
    }

    // 5. Push updates to iSAM2
    if (!graph.empty() || initialEstimate.size() > 0) {
        isam2->update(graph, initialEstimate);
    }
    graph.resize(0);
    initialEstimate.clear();

    // 6. Clean up lost tracks from memory
    pruneLostTracks(tracksToRemove);

    // 7. Publish tracked detections
    publishTrackedDetections(msg);
}


std::vector<int> linear_sum_assignment_jonker_volgenant_eigen(const Eigen::MatrixXf& costMatrix) {
    // Efficient implementations require a square matrix, but the algorithm doesn't require a square matrix
    int n = costMatrix.rows();
    std::vector<int> row_to_col(n, -1);
    std::vector<int> col_to_row(n, -1);

    // Dual variables
    Eigen::VectorXf u = Eigen::VectorXf::Zero(n);  // Row duals
    Eigen::VectorXf v = Eigen::VectorXf::Zero(n);  // Column duals

    constexpr float INF = std::numeric_limits<float>::infinity();

    // 1. Column Reduction
    for (int j = 0; j < n; ++j) {
        // Get minimum of each column (which track id minimizes the cost function for each measurement)
        v[j] = costMatrix.col(j).minCoeff();
    }

    // 2. Augmenting Row Reduction (Auction phase)
    std::vector<int> free_rows;
    free_rows.reserve(n);

    for (int i = 0; i < n; ++i) {
        float min1 = INF;
        float min2 = INF;
        int j1 = -1;

        for (int j = 0; j < n; ++j) {
            float redCost = costMatrix(i, j) - v[j];
            if (redCost < min1) {
                min2 = min1;
                min1 = redCost;
                j1 = j;
            } else if (redCost < min2) {
                min2 = redCost;
            }
        }

        // column minimum cost for row i
        u[i] = min1;
        // j1 is the column with this minimum cost for row i
        if (j1 != -1 && min1 < min2) {
            // subtract off the difference in the two columns that have the minimum cost for row i, where the initial value in v[j1] is the minimum cost in that column
            v[j1] -= (min2 - min1);
        }

        if (j1 != -1 && col_to_row[j1] == -1) {
            // mapping of which column in row i has the min cost
            row_to_col[i] = j1;
            // mapping of row i to which column has the min cost
            col_to_row[j1] = i;
        } else {
            free_rows.push_back(i);
        }
    }

    // 3. Augmentation Phase (Dijkstra shortest paths)
    Eigen::VectorXf min_r(n);
    std::vector<int> parent(n);
    std::vector<bool> visited(n);

    for (int free_row : free_rows) {
        min_r.setConstant(INF);
        std::fill(parent.begin(), parent.end(), free_row);
        std::fill(visited.begin(), visited.end(), false);

        // Initialize distances from the unassigned row
        for (int j = 0; j < n; ++j) {
            // for each free_row, calculate the offset from the original cost using the min cost of the column of that row
            // i.e., u[free_row] = costMatrix.row(free_row).minCoeff();
            // and the augmented minimum coefficient of each column (v[j]), augmented by the difference in the lowest 2 costs in the row.
            min_r[j] = costMatrix(free_row, j) - u[free_row] - v[j];
        }

        int assigned_col = -1;
        int last_scanned_col = -1;

        while (assigned_col == -1) {
            // Find unvisited column with minimum reduced cost
            float min_val = INF;
            int j_star = -1;
            for (int j = 0; j < n; ++j) {
                if (!visited[j] && min_r[j] < min_val) {
                    min_val = min_r[j];
                    j_star = j;
                }
            }

            if (j_star == -1) break;  // Safeguard for disconnected graphs

            visited[j_star] = true;
            last_scanned_col = j_star;
            int i_star = col_to_row[j_star];

            if (i_star == -1) {
                assigned_col = j_star;
            } else {
                // Relax edges out of the newly reached row
                for (int j = 0; j < n; ++j) {
                    if (!visited[j]) {
                        float new_dist = min_val + (costMatrix(i_star, j) - u[i_star] - v[j]);
                        if (new_dist < min_r[j]) {
                            min_r[j] = new_dist;
                            parent[j] = i_star;
                        }
                    }
                }
            }
        }

        // Update dual variables
        float theta = min_r[last_scanned_col];
        u[free_row] += theta;
        for (int j = 0; j < n; ++j) {
            if (visited[j]) {
                v[j] -= (theta - min_r[j]);
                int i = col_to_row[j];
                if (i != -1) u[i] += (theta - min_r[j]);
            }
        }

        // Backtrace and augment
        int curr_col = assigned_col;
        while (curr_col != -1) {
            int i = parent[curr_col];
            int next_col = row_to_col[i];
            row_to_col[i] = curr_col;
            col_to_row[curr_col] = i;
            curr_col = next_col;
        }
    }

    return row_to_col;
}

void associateData_(
    const std::vector<UnpackedDetection>& measurements,
    const std::unordered_map<gtsam::Key, Eigen::Vector3f>& trackPositions,
    std::vector<std::pair<gtsam::Key, int>>& assignments,  // [(symbol, measurements_idx), ...]
    std::vector<int>& unassigned,  // [measurements_idx, ...]
    const rclcpp::Logger& logger,
    const DataAssociationCost& dataAssociationCost = static_cast<DataAssociationCost>(0),
    float gate_th = 3.0) {

    std::function<float(const Eigen::Vector3f&, const Eigen::Vector3f&)> costFunc;
    if (dataAssociationCost == DataAssociationCost::L2) {
        costFunc = [](const Eigen::Vector3f& p0, const Eigen::Vector3f& p1) {
            return (p0-p1).norm();
        };
    } else if (dataAssociationCost == DataAssociationCost::SquaredEuclidean) {
        costFunc = [](const Eigen::Vector3f& p0, const Eigen::Vector3f& p1) {
            return (p0 - p1).squaredNorm();
        };
    } else if (dataAssociationCost == DataAssociationCost::Manhattan) {
        costFunc = [](const Eigen::Vector3f& p0, const Eigen::Vector3f& p1) {
            return (p0 - p1).cwiseAbs().sum();
        };
    } else {
        std::stringstream ss;
        ss << dataAssociationCost;
        RCLCPP_WARN(logger, "Unidentified dataAssociationCost: '%s'. Defaulting to 'L2-norm'.", ss.str().c_str());
        costFunc = [](const Eigen::Vector3f& p0, const Eigen::Vector3f& p1) {
            return (p0-p1).norm();
        };
    }

    Eigen::MatrixXf costMatrix = Eigen::MatrixXf::Zero(trackPositions.size(), measurements.size());
    std::vector<gtsam::Key> symbols;
    symbols.reserve(trackPositions.size());
    for (const auto& pair : trackPositions) {
        symbols.push_back(pair.first);
    }
    for (int i = 0; i < trackPositions.size(); i++) {
        for (int j = 0; j < measurements.size(); j++) {
            //
            //                                           measurements
            //           -----------------------------------------------------------------------
            //           | dist(id0, meas0)  |  dist(id0, meas1)  |  ...  |  dist(id0, measN)  |
            // track ids | dist(id1, meas0)  |        ...         |  ...  |  dist(id1, measN)  |
            //           |        ...        |        ...         |  ...  |         ...        |
            //           | dist(idM, meas0)  |        ...         |  ...  |  dist(idM, measN)  |
            //           -----------------------------------------------------------------------
            costMatrix(i, j) = costFunc(trackPositions.at(symbols[i]), measurements[j].center);
        }
    }

    std::vector<int> trackIndicesToMeasurements = linear_sum_assignment_jonker_volgenant_eigen(costMatrix);
    for (int i = 0; i < measurements.size(); i++) {
        unassigned.push_back(i);
    }

    for (int track_idx = 0; track_idx < trackIndicesToMeasurements.size(); ++track_idx) {
        int meas_idx = trackIndicesToMeasurements[track_idx];
        if (costMatrix(track_idx, meas_idx) < gate_th) {
            gtsam::Key symbol = symbols[track_idx];
            assignments.push_back(std::make_pair(symbol, meas_idx));
            unassigned.erase(std::remove(unassigned.begin(), unassigned.end(), meas_idx), unassigned.end());
        }
    }
}

void TrackingNode::associateData(vision_msgs::msg::Detection3DArray::ConstSharedPtr msg, std::vector<std::pair<gtsam::Key, int>>& assignments, std::vector<int>& unassigned) {
    RCLCPP_INFO(this->get_logger(), "Received array with %zu detections in frame: %s", msg->detections.size(), msg->header.frame_id.c_str());
    if (msg->detections.size() == 0)
        return;

    auto carTrackPositions = getCarTrackPositions();
    auto pedestrianTrackPositions = getPedestrianTrackPositions();

    if (carTrackPositions.size() == 0 && pedestrianTrackPositions.size() == 0)
        return;

    // Should be fine, b/c we use different gtsam::symbol_shorthand:: values for both anyway
    carTrackPositions.merge(pedestrianTrackPositions);
    if (pedestrianTrackPositions.size() != 0) {
        RCLCPP_ERROR(this->get_logger(), "Overlapping track IDs between Cars and Pedestrians");
    }

    currentDetections.clear();
    // std::vector<UnpackedDetection> measCarDetections;
    // std::vector<UnpackedDetection> measPedestrianDetections;
    for (const auto& detection : msg->detections) {
        std::string class_id;
        float score = 0.0;
        for (const auto& result : detection.results) {
            RCLCPP_INFO(this->get_logger(), "   Class ID: %s, Confidence Score: %.2f", result.hypothesis.class_id.c_str(), result.hypothesis.score);
            // TODO: Look at divvying up detections based on what the detector says the score is or ignore whether it is a car or pedestrian or whatever, maybe can give better matching and we don't necessarily care about maintaining the detection type, just the track...but also using the score information and what the type is could possibly help with false id flips.
            if (result.hypothesis.score > score) {
                class_id = result.hypothesis.class_id;
                score = result.hypothesis.score;
            }
        }

        const auto& center = detection.bbox.center.position;
        const auto& orientation = detection.bbox.center.orientation;
        const auto& size = detection.bbox.size;

        UnpackedDetection unpackedDetection(Eigen::Vector3f(center.x, center.y, center.z),
                                            Eigen::Vector4f(orientation.x, orientation.y, orientation.z,
                                                            orientation.w),
                                            Eigen::Vector3f(size.x, size.y, size.z),
                                            class_id,
                                            score);

        std::stringstream ss;
        ss << unpackedDetection;
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

        if (!detection.id.empty()) {
            RCLCPP_INFO(this->get_logger(), "    Object ID: %s", detection.id.c_str());
        }

        currentDetections.push_back(unpackedDetection);
    }

    // At this point `carTrackPositions` is all trackPositions
    // assignments will be the indices (i, j) for (carTrackPositions[i], msg->detections[j])
    associateData_(currentDetections, carTrackPositions, assignments, unassigned, this->get_logger(), dataAssociationCost, config->dataAssociationGateTh);
}

std::unordered_map<gtsam::Key, Eigen::Vector3f> TrackingNode::getCarTrackPositions() {
    currentEstimate = isam2->calculateEstimate();
    currentCarTrackPositions.clear();
    std::unordered_map<gtsam::Key, Eigen::Vector3f> trackPositions;
    for (int i = 0; i < activeCarTrackSymbols.size() - 1; i++) {
        if (currentEstimate.find(activeCarTrackSymbols[i]) != currentEstimate.end()) {
            auto& pos = currentEstimate.at<gtsam::Point3>(activeCarTrackSymbols[i]);
            const Eigen::Vector3f pos_eig(pos.x(), pos.y(), pos.z());
            currentCarTrackPositions.push_back(pos_eig);
            trackPositions[activeCarTrackSymbols[i]] = pos_eig;
        }
    }
    return trackPositions;
}


std::unordered_map<gtsam::Key, Eigen::Vector3f> TrackingNode::getPedestrianTrackPositions() {
    currentEstimate = isam2->calculateEstimate();
    currentPedestrianTrackPositions.clear();
    std::unordered_map<gtsam::Key, Eigen::Vector3f> trackPositions;
    for (int i = 0; i < activePedestrianTrackSymbols.size() - 1; i++) {
        if (currentEstimate.find(activePedestrianTrackSymbols[i]) != currentEstimate.end()) {
            auto& pos = currentEstimate.at<gtsam::Point3>(activePedestrianTrackSymbols[i]);
            const Eigen::Vector3f pos_eig(pos.x(), pos.y(), pos.z());
            currentPedestrianTrackPositions.push_back(pos_eig);
            trackPositions[activePedestrianTrackSymbols[i]] = pos_eig;
        }
    }
    return trackPositions;
}

void TrackingNode::getActiveInfoFromSymbol(const gtsam::Key& symbol, int& trackId, UnpackedDetection& detection, int& idx, std::string& detectionType) {
    trackId = -1;
    getCarIndexFromSymbol(symbol, idx);
    if (idx != -1) {
        trackId = activeCarTrackIds[idx];
        detection = activeCarDetections[idx];
        detectionType = "car";
        return;
    }
    getPedestrianIndexFromSymbol(symbol, idx);
    if (idx != -1) {
        trackId = activePedestrianTrackIds[idx];
        detection = activePedestrianDetections[idx];
        detectionType = "pedestrian";
        return;
    }
}

void TrackingNode::getCarIndexFromSymbol(const gtsam::Key &symbol, int &idx) {
    idx = -1;
    if (const auto it = std::find(activeCarTrackSymbols.begin(), activeCarTrackSymbols.end(), symbol); it != activeCarTrackSymbols.end()) {
        idx = std::distance(activeCarTrackSymbols.begin(), it);
    }
}

void TrackingNode::getPedestrianIndexFromSymbol(const gtsam::Key &symbol, int &idx) {
    idx = -1;
    if (const auto it = std::find(activePedestrianTrackSymbols.begin(), activePedestrianTrackSymbols.end(), symbol); it != activePedestrianTrackSymbols.end()) {
        idx = std::distance(activePedestrianTrackSymbols.begin(), it);
    }
}


void TrackingNode::getCarIndexFromTrackId(const int& trackId, int& idx) {
    idx = -1;
    if (const auto it = std::find(activeCarTrackIds.begin(), activeCarTrackIds.end(), trackId); it != activeCarTrackIds.end()) {
        idx = std::distance(activeCarTrackIds.begin(), it);
    }
}

void TrackingNode::getPedestrianIndexFromTrackId(const int& trackId, int& idx) {
    idx = -1;
    if (const auto it = std::find(activePedestrianTrackIds.begin(), activePedestrianTrackIds.end(), trackId); it != activePedestrianTrackIds.end()) {
        idx = std::distance(activePedestrianTrackIds.begin(), it);
    }
}

void TrackingNode::pruneLostTracks(std::vector<gtsam::Key> tracksToRemove) {
    if (tracksToRemove.empty()) {
        RCLCPP_INFO(this->get_logger(), "No tracks to remove");
        return;
    }
    RCLCPP_INFO(this->get_logger(), "Removing '%zu' tracks", tracksToRemove.size());

    for (const auto& symbol : tracksToRemove) {
        pruneFromSymbol(symbol);
    }

    isam2->update(graph, initialEstimate, tracksToRemove);
}

void TrackingNode::pruneFromSymbol(const gtsam::Key& symbol) {
    int idx;
    getCarIndexFromSymbol(symbol, idx);
    missedCounters.erase(symbol);
    if (idx != -1) {
        RCLCPP_INFO(this->get_logger(), "Car track '%d' removed.", idx);
        // activeCarTrackSymbols.erase(std::remove(activeCarTrackSymbols.begin(), activeCarTrackSymbols.end(), idx),
        //                             activeCarTrackSymbols.end());
        // activeCarTrackIds.erase(std::remove(activeCarTrackIds.begin(), activeCarTrackIds.end(), idx),
        //                         activeCarTrackIds.end());
        // activeCarDetections.erase(std::remove(activeCarDetections.begin(), activeCarDetections.end(), idx),
        //                           activeCarDetections.end());
        activeCarTrackSymbols.erase(activeCarTrackSymbols.begin() + idx);
        activeCarTrackIds.erase(activeCarTrackIds.begin() + idx);
        activeCarDetections.erase(activeCarDetections.begin() + idx);
        return;
    }
    getPedestrianIndexFromSymbol(symbol, idx);
    if (idx != -1) {
        RCLCPP_INFO(this->get_logger(), "Pedestrian track '%d' removed.", idx);
        // activePedestrianTrackSymbols.erase(
        //     std::remove(activePedestrianTrackSymbols.begin(), activePedestrianTrackSymbols.end(), idx),
        //     activePedestrianTrackSymbols.end());
        // activePedestrianTrackIds.erase(
        //     std::remove(activePedestrianTrackIds.begin(), activePedestrianTrackIds.end(), idx),
        //     activePedestrianTrackIds.end());
        // activePedestrianDetections.erase(
        //     std::remove(activePedestrianDetections.begin(), activePedestrianDetections.end(), idx),
        //     activePedestrianDetections.end());
        activePedestrianTrackSymbols.erase(activePedestrianTrackSymbols.begin() + idx);
        activePedestrianTrackIds.erase(activePedestrianTrackIds.begin() + idx);
        activePedestrianDetections.erase(activePedestrianDetections.begin() + idx);
        return;
    }
}

void convertUnpackedDetectionsToDetectionMsgs(
    const std::vector<UnpackedDetection>& detections,
    const std::vector<int>& trackIds,
    vision_msgs::msg::Detection3DArray& msg) {

    for (int i = 0; i < detections.size(); i++) {
        const auto& detection = detections[i];
        auto detectionMsg = vision_msgs::msg::Detection3D();
        detectionMsg.header = msg.header;

        vision_msgs::msg::ObjectHypothesisWithPose hypothesisMsg;
        hypothesisMsg.hypothesis.class_id = "car_" + std::to_string(trackIds[i]);
        hypothesisMsg.hypothesis.score = detection.predClassScore;

        auto position = geometry_msgs::msg::Point();
        position.x = detection.center(0);
        position.y = detection.center(1);
        position.z = detection.center(2);
        hypothesisMsg.pose.pose.position = position;
        auto orientation = geometry_msgs::msg::Quaternion();
        orientation.x = detection.orientation(0);
        orientation.y = detection.orientation(1);
        orientation.z = detection.orientation(2);
        orientation.w = detection.orientation(3);
        hypothesisMsg.pose.pose.orientation = orientation;

        detectionMsg.results.push_back(hypothesisMsg);

        detectionMsg.bbox.center.position = hypothesisMsg.pose.pose.position;
        detectionMsg.bbox.center.orientation = hypothesisMsg.pose.pose.orientation;
        auto size = geometry_msgs::msg::Vector3();
        size.x = detection.size(0);
        size.y = detection.size(1);
        size.z = detection.size(2);
        detectionMsg.bbox.size = size;

        msg.detections.push_back(detectionMsg);
    }
}

void TrackingNode::publishTrackedDetections(const vision_msgs::msg::Detection3DArray::ConstSharedPtr& incomingMsg) const {
    auto msg = vision_msgs::msg::Detection3DArray();
    msg.header = incomingMsg->header;

    convertUnpackedDetectionsToDetectionMsgs(activeCarDetections, activeCarTrackIds, msg);
    convertUnpackedDetectionsToDetectionMsgs(activePedestrianDetections, activePedestrianTrackIds, msg);

    pubTrackedDetections3d->publish(msg);
}
