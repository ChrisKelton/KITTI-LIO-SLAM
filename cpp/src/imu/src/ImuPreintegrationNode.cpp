//
// Created by ckelton on 6/10/26.
//

#include <cmath>

#include "ImuPreintegrationNode.h"
#include "slam_utilities/utility.h"


ImuPreintegrationNode::ImuPreintegrationNode() : Node("imu_preintegration") {
    config = new Config();
    setup_config();
    initialize();
    setup_subpub();
}

ImuPreintegrationNode::~ImuPreintegrationNode() {}

void ImuPreintegrationNode::setup_config() {
    config->imuTopic = this->declare_parameter("imuTopic", "imu_raw");
    config->odomTopic = this->declare_parameter("odomTopic", "odometry/imu");

    config->odometryFrame = this->declare_parameter("odometryFrame", "odom");

    config->imuAccNoise = this->declare_parameter<float>("imuAccNoise", 0.01);
    config->imuGyrNoise = this->declare_parameter<float>("imuGyrNoise", 0.001);
    config->imuAccBiasN = this->declare_parameter<float>("imuAccBiasN", 0.0002);
    config->imuGyrBiasN = this->declare_parameter<float>("imuGyrBiasN", 0.00003);

    config->imuGravity = this->declare_parameter<float>("imuGravity", 9.80511);

    config->extRotV    = this->declare_parameter<std::vector<float>>("extRotV",    std::vector<float>());
    config->extRotRPYV = this->declare_parameter<std::vector<float>>("extRotRPYV", std::vector<float>());
    config->extRot  = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config->extRotV.data(),    3, 3);
    config->extRPY  = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config->extRotRPYV.data(), 3, 3);
    config->extQRPY = Eigen::Quaterniond(config->extRPY).inverse();
    config->extTransV = this->declare_parameter("extTransV", std::vector<double>());
    config->extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config->extTransV.data(), 3, 1);
    imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                             gtsam::Point3(-config->extTrans.x(), -config->extTrans.y(), -config->extTrans.z()));
    lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(config->extTrans.x(), config->extTrans.y(), config->extTrans.z()));

    config->reset_graph_every_n_keyframes = this->declare_parameter<int>("reset_graph_every_n_keyframes", 100);
}

void ImuPreintegrationNode::initialize() {
    initialize_noise();
    initialize_preint_params();
}

void ImuPreintegrationNode::initialize_noise() {
    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());  // rad,rad,rad,m,m,m
    // TODO: Verify this value is correct, the LIO-SAM code has this at 1e4, but that would mean the noise on the velocity of the IMU is 1000 m/s
    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-4);  // m/s
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);  // 1e-2 ~ 1e-3 good for LIO-SAM app
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());  // rad,rad,rad,m,m,m
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());  // rad,rad,rad,m,m,m

    // I believe this is deprecated with the usage of gtsam::CombinedImuMeasurement also estimtaing the bias
    noiseModelBetweenBias = (gtsam::Vector(6) << config->imuAccBiasN, config->imuAccBiasN, config->imuGyrBiasN, config->imuGyrBiasN, config->imuGyrBiasN).finished();
}

void ImuPreintegrationNode::initialize_preint_params() {
    // MakeSharedU initializes gravity in ENU coordinate system (used by ROS2); i.e., [0, 0, -g]
    // MakeSharedD initialized gravity in NED coordinate system; i.e., [0, 0, g]
    preint_params = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(config->imuGravity);
    preint_params->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * std::pow(config->imuAccNoise, 2);  // acc white noise in continuous
    preint_params->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * std::pow(config->imuGyrNoise, 2);  // gyro white noise in continuous
    preint_params->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * std::pow(1e-4, 2);  // error committed in integrating position from velocities

    const gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
    preint_gtsam_opt = new gtsam::PreintegratedCombinedMeasurements(preint_params, prior_imu_bias);
    preint_gtsam_raw = new gtsam::PreintegratedCombinedMeasurements(preint_params, prior_imu_bias);
}

void ImuPreintegrationNode::setup_subpub() {
    subImu = this->create_subscription<sensor_msgs::msg::Imu>(
        config->imuTopic, 2000, std::bind(&ImuPreintegrationNode::imuHandler, this, std::placeholders::_1)
    );
    subOdometry = this->create_subscription<nav_msgs::msg::Odometry>(
        "mapping/odometry_incremental", 5, std::bind(&ImuPreintegrationNode::odometryHandler, this, std::placeholders::_1)
    );

    pubImuOdometry = this->create_publisher<nav_msgs::msg::Odometry>(config->odomTopic+"_incremental", 2000);
}

void ImuPreintegrationNode::odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    std::lock_guard lock(mtx);

    double currentCorrectionTime = ROS_TIME(msg);

    // make sure we have imu data to integrate
    if (imuMsgsOpt.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received odometry msg, but we have no IMU msgs...");
        return;
    }

    const float p_x = msg->pose.pose.position.x;
    const float p_y = msg->pose.pose.position.y;
    const float p_z = msg->pose.pose.position.z;
    const float r_x = msg->pose.pose.orientation.x;
    const float r_y = msg->pose.pose.orientation.y;
    const float r_z = msg->pose.pose.orientation.z;
    const float r_w = msg->pose.pose.orientation.w;
    bool degenerate = static_cast<int>(msg->pose.covariance[0] == 1 ? true : false);
    auto lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

    // initialize system
    if (!systemInitialized) {
        resetOptimization();

        // pop old IMU messages
        while (!imuMsgsOpt.empty()) {
            if (ROS_TIME(&imuMsgsOpt.front()) < currentCorrectionTime - delta_t) {
                lastImuT_opt = ROS_TIME(&imuMsgsOpt.front());
                imuMsgsOpt.pop_front();
            } else {
                break;
            }
        }

        // initial pose
        prevPose_ = lidarPose.compose(lidar2Imu);
        const gtsam::PriorFactor priorPose(X(0), prevPose_, priorPoseNoise);
        graph->add(priorPose);
        // initial velocity
        prevVel_ = gtsam::Vector3(0, 0, 0);
        const gtsam::PriorFactor priorVel(V(0), prevVel_, priorVelNoise);
        graph->add(priorVel);
        // initial bias
        prevBias_ = gtsam::imuBias::ConstantBias();
        const gtsam::PriorFactor priorBias(B(0), prevBias_, priorBiasNoise);
        graph->add(priorBias);
        // add values
        values_new.insert(X(0), prevPose_);
        values_new.insert(V(0), prevVel_);
        values_new.insert(B(0), prevBias_);
        // optimize once
        isam2->update(*graph, values_new);
        graph->resize(0);
        values_new.clear();

        preint_gtsam_opt->resetIntegrationAndSetBias(prevBias_);
        preint_gtsam_raw->resetIntegrationAndSetBias(prevBias_);

        key = 1;
        systemInitialized = true;
        return;
    }

    // reset graph for speed
    if (key == config->reset_graph_every_n_keyframes) {
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(isam2->marginalCovariance(X(key-1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(isam2->marginalCovariance(V(key-1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(isam2->marginalCovariance(B(key-1)));
        // reset graph
        resetOptimization();
        // add pose
        const gtsam::PriorFactor priorPose(X(0), prevPose_, updatedPoseNoise);
        graph->add(priorPose);
        // add velocity
        const gtsam::PriorFactor priorVel(V(0), prevVel_, updatedVelNoise);
        graph->add(priorVel);
        // add bias
        const gtsam::PriorFactor priorBias(B(0), prevBias_, updatedBiasNoise);
        graph->add(priorBias);
        // add values
        values_new.insert(X(0), prevPose_);
        values_new.insert(V(0), prevVel_);
        values_new.insert(B(0), prevBias_);
        // optimize once
        isam2->update(*graph, values_new);
        graph->resize(0);
        values_new.clear();

        key = 1;
    }

    // integrate imu data and optimize
    while (!imuMsgsOpt.empty()) {
        // pop and integrate imu data that is between two optimization
        sensor_msgs::msg::Imu *thisImu = &imuMsgsOpt.front();
        double imuTime = ROS_TIME(thisImu);
        if (imuTime < currentCorrectionTime - delta_t) {
            double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
            preint_gtsam_opt->integrateMeasurement(
                gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
                dt
            );
            lastImuT_opt = imuTime;
            imuMsgsOpt.pop_front();
        } else {
            break;
        }
    }
    // add imu factor to graph
    auto imuFactor = gtsam::CombinedImuFactor(X(key-1), V(key-1), X(key), V(key), B(key-1), B(key), *preint_gtsam_opt);
    graph->add(imuFactor);
    // Bias between factors is accounted for in CombinedImuFactor

    // add pose factor
    gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
    // use a higher noise if the pose is degenerate (covariance == 1)
    const gtsam::PriorFactor pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
    graph->add(pose_factor);
    // insert predicted values
    // gtsam::NavState propState_ = preint_gtsam_opt->predict(prevState_, prevBias_);
    gtsam::NavState propNavState_ = preint_gtsam_opt->predict(gtsam::NavState(prevState_.pose(), prevState_.v()), prevState_.b());
    gtsam::State propState_(propNavState_.pose(), propNavState_.v(), prevState_.b());
    values_new.insert(X(key), propState_.pose());
    values_new.insert(V(key), propState_.v());
    values_new.insert(B(key), propState_.b());
    // optimize
    isam2->update(*graph, values_new);
    isam2->update();
    graph->resize(0);
    values_new.clear();
    // Overwrite the beginning of the preintegration for the next step
    values_est = isam2->calculateEstimate();
    prevPose_ = values_est.at<gtsam::Pose3>(X(key));
    prevVel_ = values_est.at<gtsam::Vector3>(V(key));
    prevBias_ = values_est.at<gtsam::imuBias::ConstantBias>(B(key));
    prevState_ = gtsam::State(prevPose_, prevVel_, prevBias_);
    // Reset the optimization preintegration object
    preint_gtsam_opt->resetIntegrationAndSetBias(prevBias_);
    // check optimization
    if (failureDetection(prevVel_, prevBias_)) {
        resetParams();
        return;
    }

    // After optimization, re-propagate imu odometry preintegration
    prevStateOdom = prevState_;
    // first pop imu messages older than current correction data
    double lastImuQT = -1;
    while (!imuMsgsRaw.empty() && ROS_TIME(&imuMsgsRaw.front()) < currentCorrectionTime - delta_t) {
        lastImuQT = ROS_TIME(&imuMsgsRaw.front());
        imuMsgsRaw.pop_front();
    }
    // repropagate
    if (!imuMsgsRaw.empty()) {
        // reset bias use the newly optimized bias
        preint_gtsam_raw->resetIntegrationAndSetBias(prevStateOdom.b());
        // integrate imu message from the beginning of this optimization
        for (int i = 0; i < static_cast<int>(imuMsgsRaw.size()); ++i) {
            sensor_msgs::msg::Imu *thisImu = &imuMsgsRaw.at(i);
            double imuTime = ROS_TIME(thisImu);
            double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
            preint_gtsam_raw->integrateMeasurement(
                gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z),
                dt
            );
            lastImuQT = imuTime;
        }
    }

    ++key;
    doneFirstOpt = true;
}

void ImuPreintegrationNode::imuHandler(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
    std::lock_guard lock(mtx);

    // Converts from imu frame -> lidar frame
    sensor_msgs::msg::Imu thisImu = imuConverter(*msg, config->extRot, config->extQRPY, this->get_logger());

    imuMsgsOpt.push_back(thisImu);
    imuMsgsRaw.push_back(thisImu);

    if (!doneFirstOpt) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *(this->get_clock()), 5,
                             "Waiting for first optimization. Have '%zu' IMU measurements.", imuMsgsRaw.size());
        return;
    }

    const double imuTime = ROS_TIME(&thisImu);
    const double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
    lastImuT_imu = imuTime;

    // integrate single imu message
    preint_gtsam_raw->integrateMeasurement(
        gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
        gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z),
        dt
    );

    // predict odometry
    // gtsam::NavState currentState = preint_gtsam_raw->predict(prevStateOdom, prevBiasOdom);
    gtsam::NavState currentNavState = preint_gtsam_opt->predict(gtsam::NavState(prevStateOdom.pose(), prevStateOdom.v()), prevStateOdom.b());
    gtsam::State currentState(currentNavState.pose(), currentNavState.v(), prevStateOdom.b());

    // publish odometry
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = thisImu.header.stamp;
    odometry.header.frame_id = config->odometryFrame;
    odometry.child_frame_id = "odom_imu";

    // transform imu pose to lidar
    gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
    gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

    odometry.pose.pose.position.x = lidarPose.translation().x();
    odometry.pose.pose.position.y = lidarPose.translation().y();
    odometry.pose.pose.position.z = lidarPose.translation().z();
    odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
    odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
    odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
    odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

    odometry.twist.twist.linear.x = currentState.velocity().x();
    odometry.twist.twist.linear.y = currentState.velocity().y();
    odometry.twist.twist.linear.z = currentState.velocity().z();
    odometry.twist.twist.angular.x = thisImu.angular_velocity.x + currentState.gyroscope().x();
    odometry.twist.twist.angular.y = thisImu.angular_velocity.y + currentState.gyroscope().y();
    odometry.twist.twist.angular.z = thisImu.angular_velocity.z + currentState.gyroscope().z();
    // odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
    // odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
    // odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();

    pubImuOdometry->publish(odometry);
}

bool ImuPreintegrationNode::failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur) const {
    if (const Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z()); vel.norm() > 30) {
        RCLCPP_WARN(this->get_logger(), "Large velocity: '%.2f' m/s, reset IMU-preintegration!", vel.norm());
        return true;
    }

    const Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
    const Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
    if (ba.norm() > 1.0 || bg.norm() > 1.0) {
        RCLCPP_WARN(this->get_logger(), "Large bias: '%.2f' m/s^2 & '%.2f' rad/s, reset IMU-preintegration!", ba.norm(), bg.norm());
        return true;
    }

    return false;
}
