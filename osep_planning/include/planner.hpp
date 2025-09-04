#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include "types.hpp"
#include "graph.hpp"
#include "rho.hpp"

#include <iostream>
#include <chrono>
#include <unordered_set>
#include <queue>

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
    float budget = 1000.0f;          // max travel cost for this horizon (meters or edge cost units)
    float lambda = 1.0f;           // travel-vs-reward trade-off
    float subgraph_radius = 200.0f; // dijkstra radius cutoff
    int   subgraph_max_nodes = 50;
    float hysteresis = 0.15f;      // replan only if new_score > old*(1+hysteresis)
    int   warm_steps = 1;          // how many first steps to commit before re-planning

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
    bool set_path(std::vector<Vertex>& gskel);

    /* Helper */
    int pick_start_gid_near_drone();
    std::vector<int> build_subgraph(int start_gid, std::vector<char>& allow_transit);
    std::vector<int> greedy_plan(int start_gid, const std::vector<int>& cand, const std::vector<char>& allow_transit, float budget_left);
    void dijkstra(const std::vector<char>& allow, int s, std::vector<float>& dist, std::vector<int>& parent, const float radius=std::numeric_limits<float>::infinity());
    bool line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b);
    float edge_cost(GraphEdge& e);
    float node_reward(const GraphNode& n);
    float path_travel_cost(const std::vector<int>& gids);

    bool mark_visited_in_skeleton(uint64_t hid, std::vector<Vertex>& gskel);
    
    inline Eigen::Quaternionf yaw_to_quat(float yaw) { return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())); }

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