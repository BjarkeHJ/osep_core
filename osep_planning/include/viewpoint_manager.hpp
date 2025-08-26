#ifndef VIEWPOINT_MANAGER_HPP_
#define VIEWPOINT_MANAGER_HPP_

#include <iostream>
#include <chrono>
#include <unordered_set>
#include <pcl/common/common.h>
#include <Eigen/Core>
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

struct Viewpoint {
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;

    int corresp_vertex_id = -1;
    float score = 0.0f;
    bool in_path = false;
    bool visited = false;
    bool invalid = false;
};

struct Vertex {
    int vid = -1;
    std::vector<int> nb_ids;
    pcl::PointXYZ position;
    int type = 0;
    bool pos_update = false;
    bool type_update = false;

    std::vector<Viewpoint> vpts; // Vertex viewpoints
};


struct ViewpointData {
    size_t gskel_size;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map;
    std::vector<Vertex> global_skel;
    std::vector<std::vector<int>> global_adj;
    std::vector<int> updated_vertices;
    std::vector<std::vector<int>> branches;

    std::vector<Viewpoint> global_vpts;
};    


class ViewpointManager {
public:
    ViewpointManager(const ViewpointConfig &cfg);
    bool viewpoint_run();
    std::vector<Vertex>& input_skeleton() { return VD.global_skel; }
    pcl::PointCloud<pcl::PointXYZ>& input_map() { return *VD.global_map; }
    std::vector<Viewpoint>& output_vpts() { return VD.global_vpts; }

private:
    /* Functions */
    bool fetch_updated_vertices();
    bool branch_extract();

    bool build_all_vpts();

    bool viewpoint_sampling();
    bool viewpoint_filtering();
    bool viewpoint_visibility_graph();

    /* Helper */
    std::vector<int> walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges);
    std::vector<Viewpoint> generate_viewpoint(int id);
    std::vector<Viewpoint> sample_vp(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& directions, std::vector<float> dists, int vertex_id);

    float distance_to_free_space(const Eigen::Vector3f& p, const Eigen::Vector3f dir_in);
    bool map_occupied_at_index(int ix, int iy, int iz);

    /* Params */
    ViewpointConfig cfg_;
    bool running;

    /* Data */
    ViewpointData VD;

    std::shared_ptr<pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ>> octree_;

    std::unordered_map<int,int> vid2idx_;
    std::vector<int> idx2vid_;
    std::vector<int> degree_;
    std::vector<int> is_endpoint_;

    /* Utils */


};

#endif