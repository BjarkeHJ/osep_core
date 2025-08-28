#ifndef VIEWPOINT_MANAGER_HPP_
#define VIEWPOINT_MANAGER_HPP_

#include "types.hpp"

#include <iostream>
#include <chrono>
#include <unordered_set>
#include <pcl/octree/octree.h>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)

struct ViewpointConfig {
    float map_voxel_size;
    float vpt_disp_dist;
};

struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const noexcept {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};



struct VptHandle {
    int vid; // Vertex id (in gskel)
    int idx; // Index inside that vertex's vpts vector
};

struct ViewpointData {
    size_t gskel_size;
    std::vector<Vertex> gskel;
    std::unordered_map<int,int> gskel_vid2idx;
    
    std::vector<std::vector<int>> gadj;
    std::vector<int> updated_vertices;
    std::vector<std::vector<int>> branches;

    pcl::PointCloud<pcl::PointXYZ>::Ptr gmap;
    
    std::vector<Viewpoint> global_vpts;
    std::vector<VptHandle> global_vpts_handles; // for referencing into each vertex viewpoints
};


class ViewpointManager {
public:
    ViewpointManager(const ViewpointConfig &cfg);
    bool viewpoint_run();

    pcl::PointCloud<pcl::PointXYZ>& input_map() { return *VD.gmap; }
    void update_skeleton(const std::vector<Vertex>& verts);

    std::vector<Vertex>& output_skeleton() { return VD.gskel; }

private:
    /* Functions */
    bool fetch_updates();
    bool branch_extract();
    bool build_all_vpts();

    bool viewpoint_sampling();
    bool viewpoint_filtering();
    bool viewpoint_scoring();

    bool viewpoint_visibility_graph();

    /* Helper */
    std::vector<int> walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges);
    std::vector<Viewpoint> generate_viewpoint(int id);
    void build_local_frame(const int vid, Eigen::Vector3f& that, Eigen::Vector3f& n1hat, Eigen::Vector3f& n2hat);
    float distance_to_free_space(const Eigen::Vector3f& p_in, const Eigen::Vector3f dir_in);

    /* Viewpoint Handle Helpers */
    inline bool is_valid_handle(const VptHandle& h) {
        return h.vid >= 0 && h.vid < static_cast<int>(VD.gskel.size()) && h.idx >= 0 && h.idx < static_cast<int>(VD.gskel[h.vid].vpts.size());
    }
    inline Viewpoint& get_vp_from_handle(const VptHandle& h) {
        return VD.gskel[h.vid].vpts[h.idx];
    }
    inline void erase_handles_for_vertex(int vid) {
        VD.global_vpts_handles.erase(std::remove_if(VD.global_vpts_handles.begin(), VD.global_vpts_handles.end(), [vid](const VptHandle& h){ return h.vid == vid; }), VD.global_vpts_handles.end());
    }
    inline void append_handles_for_vertex(int vid) {
        const auto& v = VD.gskel[vid];
        for (int j=0; j<(int)v.vpts.size(); ++j) {
            VD.global_vpts_handles.push_back({vid, j});
        }
    }

    /* Viewpoint Sampling Helpers */
    inline float deg2rad(float d) { return d * float(M_PI) / 180.0f; }
    inline float wrapPi(float a) {
        float twoPi = 2.0f * float(M_PI);
        float x = std::fmod(a + float(M_PI), twoPi);
        if (x < 0) x += twoPi;
        return x-float(M_PI);
    }
    inline float yaw_to_face(const Eigen::Vector3f& c, const Eigen::Vector3f& p) {
        // yaw to face target p from camera c
        Eigen::Vector2f d = (p.head<2>() - c.head<2>());
        return std::atan2(d.y(), d.x());
    }
    inline Eigen::Quaternionf yaw_to_quat(float yaw) { return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())); }
    inline bool is_not_safe_dist(const Eigen::Vector3f& p) {
        std::vector<int> ids; std::vector<float> d2s; pcl::PointXYZ qp(p.x(), p.y(), p.z());
        return octree_->radiusSearch(qp, cfg_.vpt_disp_dist, ids, d2s) > 0;
    }

    /* Params */
    ViewpointConfig cfg_;
    bool running;

    /* Data */
    ViewpointData VD;

    // std::shared_ptr<pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ>> octree_;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;

    std::unordered_map<int,int> vid2idx_;
    std::vector<int> idx2vid_;
    std::vector<int> degree_;
    std::vector<int> is_endpoint_;

    /* Utils */


};

#endif