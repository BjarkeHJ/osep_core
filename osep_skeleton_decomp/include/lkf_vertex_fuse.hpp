#ifndef KALMAN_VERTEX_FUSION_
#define KALMAN_VERTEX_FUSION_

#include <Eigen/Core>

struct VertexLKF {
    Eigen::Vector3f x;
    Eigen::Matrix3f P;

    void initFrom(const Eigen::Ref<const Eigen::Vector3f>& x0,
                  const Eigen::Ref<const Eigen::Matrix3f>& P0) {
        x = x0;
        P = P0;
    }

    void update(const Eigen::Ref<const Eigen::Vector3f>& z,
                const Eigen::Ref<const Eigen::Matrix3f>& Q,
                const Eigen::Ref<const Eigen::Matrix3f>& R) {
        // Prediction
        const Eigen::Vector3f x_pred = x;
        const Eigen::Matrix3f P_pred = P + Q;
        // Innovation and Gain
        const Eigen::Matrix3f S = P_pred + R;
        const Eigen::Matrix3f K = P_pred * S.inverse();
        // Correction
        x = x_pred + K * (z - x_pred);
        P = (Eigen::Matrix3f::Identity() - K) * P_pred;
    }
};

#endif