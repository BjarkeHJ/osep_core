#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include "types.hpp"
#include "graph.hpp"
#include "rho.hpp"

#include <iostream>
#include <chrono>
#include <unordered_set>
#include <queue>
#include <algorithm>

#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree_flann.h>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)


struct PlannerConfig {
    float graph_radius;
    float map_voxel_size;
    float safe_dist;

    // RHO params
    float budget = 200.0f;          // max travel cost for this horizon (meters or edge cost units)
    float lambda = 0.1f;           // travel-vs-reward trade-off

    float subgraph_radius = 200.0f; // dijkstra radius cutoff
    int subgraph_max_nodes = 50;

    float hysteresis = 0.15f;      // replan only if new_score > old*(1+hysteresis)

    int dfs_max_depth = 10;     // max number of viewpoints in receding horizon plan
    int dfs_beam_width = 5;    // beam width for dfs expansion (0 = full expansion)

    float geometric_bias = 1.0f;   // optional: add cost penalty for geometric edges
    float topo_bonus = 10.0f;       // optional: subtract cost on topological edges
};

struct DronePose {
    Eigen::Vector3f pos;
    Eigen::Quaternionf ori;
};

struct PlannerData {
    Graph graph;
    DronePose drone_pose;
    RHState rhs; // Receding horizon state
    std::vector<Viewpoint> path_out;

    std::unordered_map<uint64_t, int> h2g; // Map unique vpt id (handle) to graph node index (this tick) it = h2g.find(vptid) gid = it->second
    std::vector<uint64_t> g2h; // Map graph node index to unique viewpoint id (handle) g2h[gid] -> vptid

    std::unordered_map<uint64_t, float> score_ema;

};


class PathPlanner {
public:
    PathPlanner(const PlannerConfig& cfg);

    void update_skeleton(const std::vector<Vertex>& verts);
    void set_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> oct = nullptr) {
       gmap = std::move(cloud);
       octree_ = std::move(oct); 
    }
    void set_vid2idx(std::unordered_map<int,int>& vid2idx) { gskel_vid2idx = vid2idx; } // Copy current unique vid mapping
    void set_drone_pose(const Eigen::Vector3f& pos, const Eigen::Quaternionf ori) {
        if (!pos.allFinite()) return;
        if (!ori.coeffs().allFinite() || ori.norm() < 1e-6f) return;
        PD.drone_pose.pos = pos;
        PD.drone_pose.ori = ori;
        plan_path = true; // recieved valid drone pose
    }

    const Graph& get_graph() { return PD.graph; }
    bool plan(std::vector<Vertex>& gskel);

    const std::vector<Viewpoint>& current_path() { return PD.path_out; }

    bool get_next_target(Viewpoint& out);
    bool get_next_k_targets(std::vector<Viewpoint>& out_k, int k);
    bool get_start(Viewpoint& out);
    bool notify_reached(std::vector<Vertex>& gskel);

private:
    /* Functions */
    bool build_graph(std::vector<Vertex>& gskel);
    bool rh_plan_tick();
    bool path_refine_tick();
    bool set_path(std::vector<Vertex>& gskel);

    /* Helper */
    void rehydrate_state(std::vector<Vertex>& gskel);
    int pick_start_gid_near_drone();
    std::vector<int> build_subgraph(int start_gid, std::vector<char>& allow_transit);
    void bounded_dfs_plan(int start_gid, const std::vector<int>& subgraph, const std::vector<char> allow_transit, std::vector<int>& path, float& best_score);

    void dijkstra(const std::vector<char>& allow, int s, std::vector<float>& dist, std::vector<int>& parent, const float radius=std::numeric_limits<float>::infinity());
    bool line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b);
    float edge_cost(const GraphEdge& e);
    float node_reward(const GraphNode& n);
    uint64_t retarget_head();

    
    float path_travel_cost(const std::vector<int>& gids);
    std::vector<int> handles_to_gids(const std::vector<uint64_t>& h);
    float path_score_from_gids(const std::vector<int>& gids);
    int common_prefix_len(const std::vector<int>& A, const std::vector<int>& B);


    bool mark_visited_in_skeleton(uint64_t hid, std::vector<Vertex>& gskel);    
    
    inline Eigen::Quaternionf yaw_to_quat(float yaw) { return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())); }

    inline bool is_in_subgraph(int gid, const std::unordered_set<int>& sub_set) { return sub_set.find(gid) != sub_set.end(); }

    /* Params & Data */
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr gmap;
    std::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>> octree_;
    std::unordered_map<int,int> gskel_vid2idx;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> vpt_kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vpt_cloud;
    
    
    PlannerConfig cfg_;
    PlannerData PD;
    
    bool running;
    bool plan_path = false;
};


#endif