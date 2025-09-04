#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <numeric>
#include <unordered_set>
#include <Eigen/Core>
#include <pcl/common/common.h>

struct GraphNode {
    int gid; // global graph node id
    int vid; // owning vertex id
    int k; // index in that vertex's vpts
    uint64_t vptid; // unique persistent viewpoint id
    Eigen::Vector3f p; // position
    float yaw;
    float score;
};

struct GraphEdge {
    int to;
    float w; // weight
    bool topo; // topo=true (topological grap) topo=false (geometric graph)
};

struct Graph {
    std::vector<GraphNode> nodes; // gid -> node 
    std::vector<std::vector<GraphEdge>> adj; // adjacency matrix
};

/* Graph Helper Functions */

// EdgeKey: pack undirected (a,b) with a<b to dedup edges
static inline uint64_t ek(int a, int b) {
    if (a > b) std::swap(a,b);
    return ( (uint64_t(uint32_t(a)) << 32) ^ uint64_t(uint32_t(b)) );
}

// squared distance
static inline float d2(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return (a - b).squaredNorm();
}

// add undirected edge if not present
static inline void add_edge(Graph& G, int u, int v, float w, std::unordered_set<uint64_t>& edge_set, bool topo) {
    if (u == v) return;
    uint64_t k = ek(u, v);
    if (edge_set.insert(k).second) {
        G.adj[u].push_back({v, w, topo});
        G.adj[v].push_back({u, w, topo});
    }
}

#endif // GRAPH_HPP_