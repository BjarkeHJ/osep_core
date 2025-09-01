#ifndef ROSA_HPP_
#define ROSA_HPP_

#include <iostream>
#include <chrono>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)

   /* std::cout << (ok ? "[SUCCESS] " : "[FAILED] ") << #fn << std::endl; \ */


struct RosaConfig {
    int max_points;
    int min_points;
    float pts_dist_lim; 
    int ne_knn;
    int nb_knn;
    float max_proj_range;
    
    int niter_drosa;
    int niter_dcrosa;
    int niter_smooth;
    float alpha_recenter;
    float radius_smooth;
};

struct RosaData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr orig_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr nrms_;

    std::vector<std::vector<int>> surf_nbs;
    std::vector<std::vector<int>> simi_nbs;

    Eigen::MatrixXf skelver;
    
    size_t pcd_size_;
};

class Rosa {
public:
    explicit Rosa(const RosaConfig& cfg);
    bool rosa_run();
    pcl::PointCloud<pcl::PointXYZ>& input_cloud() { return *RD.orig_; } // Return mutable ref so node can write into preallocated buffer
    Eigen::MatrixXf& output_vertices() { return RD.skelver; } // Return extracted skeleton vertices

private:
    /* Functions */
    bool preprocess();
    bool rosa_init();
    bool similarity_neighbor_extraction();
    bool drosa();
    bool dcrosa();
    bool vertex_sampling();
    bool vertex_smooth();

    /* Helpers */
    std::vector<int> compute_active_samples(const int seed, const Eigen::Vector3f& p, const Eigen::Vector3f& n);
    float similarity_metric(const Eigen::Vector3f& p1, const Eigen::Vector3f& v1, const Eigen::Vector3f& p2, const Eigen::Vector3f& v2, const float range_r, const float scale=5.0f);
    Eigen::Vector3f compute_symmetrynormal(const std::vector<int>& idxs);
    float symmnormal_variance(Eigen::Vector3f& symm_nrm, std::vector<int>& idxs);
    Eigen::Vector3f cov_eigs_from_normals(const std::vector<int>& idxs);
    Eigen::Vector3f closest_projection_point(const std::vector<int>& idxs);
    bool local_line_fit(const std::vector<int>& nb, Eigen::Vector3f& mean_out, Eigen::Vector3f& dir_out, float& conf_out, const int min_nb);

    /* Params */
    RosaConfig cfg_;
    float leaf_size;
    bool running;

    /* Utils */
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree_{new pcl::search::KdTree<pcl::PointXYZ>};
    pcl::VoxelGrid<pcl::PointNormal> vgf_;

    /* Data */
    RosaData RD;
    Eigen::MatrixXf pset;
    Eigen::MatrixXf vset;
    Eigen::MatrixXf vvar;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud;

    // Temporaries for moving large clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pt_;
    pcl::PointCloud<pcl::Normal>::Ptr tmp_nrm_;
    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_pn_;
    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_pn_ds_;
};

/* HELPER FUNCTION */
inline float estimate_leaf_from_bbox(const pcl::PointCloud<pcl::PointXYZ>& cloud, size_t target) {
    if (cloud.empty() || target == 0) return 0.0f;

    pcl::PointXYZ minp, maxp;
    pcl::getMinMax3D(cloud, minp, maxp);
    const float dx = std::max(1e-6f, maxp.x - minp.x);
    const float dy = std::max(1e-6f, maxp.y - minp.y);
    const float dz = std::max(1e-6f, maxp.z - minp.z);
    const float vol = dx * dy * dz;
    float leaf = std::cbrtf(vol / std::max<float>(1.0, target));
    return leaf;
}

inline Eigen::Matrix3f create_orthonormal_frame(Eigen::Vector3f &v) {
    const float n = v.norm();
    if (n == 0.0) {
        return Eigen::Matrix3f::Identity();
    }
    const Eigen::Vector3f z = v / n;
    // Picking a helper axis least aligned with vector z
    Eigen::Vector3f a = (std::abs(z.x()) < 0.9) ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitY();
    // Project onto the plane orthogonal to vector z
    Eigen::Vector3f x = (a - a.dot(z) * z).normalized();
    Eigen::Vector3f y = z.cross(x);
    Eigen::Matrix3f M;
    M.col(0) = x;
    M.col(1) = y;
    M.col(2) = z;
    return M;
}

#endif // ROSA_HPP_