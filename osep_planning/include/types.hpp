#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <numeric>
#include <unordered_set>
#include <Eigen/Core>
#include <pcl/common/common.h>


struct Viewpoint {
    Eigen::Vector3f position;
    float yaw;
    Eigen::Quaternionf orientation;
    
    int target_vid = -1; // corresponding vertex id
    int target_vp_pos = -1; // index of corresponding vertex vpts vector
    
    float score = 0.0f;
    
    bool updated = false;
    bool invalid = false;
    
    bool in_path = false;
    bool visited = false;
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

struct GraphNode {
    int gid; // global graph node id
    int vid; // owning vertex id
    int k; // index in that vertex's vpts
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

// HaskKey: pack (vid, k) -> 64bit key (stable handle)
static inline uint64_t hk(int vid, int k) {
    return ( (uint64_t(uint32_t(vid)) << 32) ^ uint64_t(uint32_t(k)) );
}

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
static inline void add_edge(Graph& G, int u, int v, float w, std::unordered_set<uint64_t>& edge_set, bool topo=true) {
    if (u == v) return;
    uint64_t k = ek(u, v);
    if (edge_set.insert(k).second) {
        G.adj[u].push_back({v, w, topo});
        G.adj[v].push_back({u, w, topo});
    }
}


struct DSU {
    std::vector<int> p, r;
    explicit DSU(int n): p(n), r(n,0) { std::iota(p.begin(), p.end(), 0); }
    int f(int x){ return p[x]==x?x:p[x]=f(p[x]); }
    bool uni(int a,int b){
        a=f(a); b=f(b); if(a==b) return false;
        if(r[a]<r[b]) std::swap(a,b);
        p[b]=a; if(r[a]==r[b]) r[a]++;
        return true;
    }
};

#endif