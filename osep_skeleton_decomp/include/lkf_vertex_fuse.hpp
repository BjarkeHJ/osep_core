#ifndef KALMAN_VERTEX_FUSION_
#define KALMAN_VERTEX_FUSION_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pcl/common/common.h>

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



struct LocalFrame {
    Eigen::Matrix3f U; // columns: tanget, n1, n2
    bool valid = false;
};

inline LocalFrame defaultFrame() {
    LocalFrame lf;
    lf.U.setIdentity();
    lf.valid = true;
    return lf;
}

inline Eigen::MatrixXf makeAnisotropic(const Eigen::Matrix3f& U, float sig_t, float sig_n) {
    Eigen::Matrix3f L = Eigen::Matrix3f::Zero();
    L(0,0) = sig_t * sig_t; // tangent variance
    L(1,1) = sig_n * sig_n; // normal variance
    L(2,2) = sig_n * sig_n; // normal variance
    return U * L * U.transpose(); // world frame covariance 
}

inline bool passGate(const Eigen::Vector3f& x, const Eigen::Vector3f& z, const Eigen::Matrix3f& U, float sig_t, float sig_n, float gate2) {
    Eigen::Vector3f dl = U.transpose() * (z - x);
    float md2 = (dl[0]*dl[0]) / (sig_t*sig_t) + (dl[1]*dl[1] + dl[2]*dl[2]) / (sig_n*sig_n);
    return md2 <= gate2;
}

inline LocalFrame refineLocalFrameFromNeighbors(const Eigen::Vector3f& x, const std::vector<int>& nn_idx, const pcl::PointCloud<pcl::PointXYZ>& ref) {
    if (nn_idx.size() < 3) return defaultFrame();

    // Compute covariance around x
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (int id : nn_idx) {
        c += ref.points[id].getVector3fMap();
    }
    c /= static_cast<float>(nn_idx.size());

    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (int id : nn_idx) {
        Eigen::Vector3f d = ref.points[id].getVector3fMap() - c;
        C += d * d.transpose();
    }
    C /= std::max(1, static_cast<int>(nn_idx.size() - 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    if (es.info() != Eigen::Success) return defaultFrame();

    // largest eigenvector -> tangent
    Eigen::Vector3f e0 = es.eigenvectors().col(2); // ascending order
    Eigen::Vector3f e1 = es.eigenvectors().col(1);
    Eigen::Vector3f e2 = es.eigenvectors().col(0);

    LocalFrame lf;
    lf.U.col(0) = e0.normalized();
    lf.U.col(1) = e1.normalized();
    lf.U.col(2) = e2.normalized();
    lf.valid = true;
    return lf;
}


#endif