//
// Created by ckelton on 6/11/26.
//
#pragma once

#ifndef SLAM_UTILITIES_STATE_H
#define SLAM_UTILITIES_STATE_H

#include <iostream>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>


namespace slam {

class State {
  private:
    gtsam::Pose3 pose_;
    gtsam::Vector3 velocity_;
    gtsam::imuBias::ConstantBias bias_;

  public:
    State()
        : pose_(gtsam::Pose3()),
          velocity_(gtsam::Vector3::Zero()),
          bias_(gtsam::imuBias::ConstantBias()) {}

    State(const State& state) = default;

    State(const gtsam::Pose3& pose, const gtsam::Vector3& velocity, const gtsam::imuBias::ConstantBias& bias)
        : pose_(pose), velocity_(velocity), bias_(bias) {}

    void set_pose(const gtsam::Pose3& pose)                        { pose_     = pose; }
    void set_velocity(const gtsam::Vector3& velocity)              { velocity_ = velocity; }
    void set_bias(const gtsam::imuBias::ConstantBias& bias)        { bias_     = bias; }

    const gtsam::Pose3&                  pose()     const { return pose_; }
    const gtsam::Vector3&                v()        const { return velocity_; }
    const gtsam::Vector3&                velocity() const { return v(); }
    const gtsam::imuBias::ConstantBias&  b()        const { return bias_; }
    const gtsam::imuBias::ConstantBias&  bias()     const { return b(); }

    // Rotation as unit quaternion
    gtsam::Quaternion q()          const { return pose_.rotation().toQuaternion(); }
    gtsam::Quaternion quaternion() const { return q(); }

    // Translation
    gtsam::Vector3 p()        const { return pose_.translation(); }
    gtsam::Vector3 position() const { return p(); }

    // Bias components
    const gtsam::Vector3& ba() const { return bias_.accelerometer(); }
    const gtsam::Vector3& bg() const { return bias_.gyroscope(); }

    friend std::ostream& operator<<(std::ostream& os, const State& state) {
        os << std::fixed
           << "[STATE]: q = " << state.q().x() << ", " << state.q().y() << ", "
                               << state.q().z() << ", " << state.q().w() << " | "
           << "p = "  << state.p()(0)  << ", " << state.p()(1)  << ", " << state.p()(2)  << " | "
           << "v = "  << state.v()(0)  << ", " << state.v()(1)  << ", " << state.v()(2)  << " | "
           << "ba = " << state.ba()(0) << ", " << state.ba()(1) << ", " << state.ba()(2) << " | "
           << "bg = " << state.bg()(0) << ", " << state.bg()(1) << ", " << state.bg()(2);
        return os;
    }

    void print(const std::string& s = "") const {
        std::cout << s << *this << std::endl;
    }

    // pose_ is set to the relative transform from this to other (this->inverse() * other);
    // velocity_ and bias_ are arithmetic differences.
    State& operator-=(const State& other) {
        pose_     = pose_.between(other.pose_);
        velocity_ = velocity_ - other.velocity_;
        bias_     = bias_ - other.bias_;
        return *this;
    }

    friend State operator-(State lhs, const State& rhs) {
        lhs -= rhs;
        return lhs;
    }
};

}  // namespace slam

#endif //SLAM_UTILITIES_STATE_H
