/* 

Main paht planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
    gmap.reset(new pcl::PointCloud<pcl::PointXYZ>);
    vpt_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    running = 1;
}


bool PathPlanner::plan_path(std::vector<Vertex>& gskel) {
    /* Main public function - Runs the path planning pipeline */

    // std::cout << "Drone Position: ";
    // for (int i=0; i<3; ++i) {
    //     std::cout << PD.drone_pose.pos[i] << " ";
    // }
    // std::cout << "\n";

    build_graph(gskel);
    generate_path(gskel);
    return 1;
}

bool PathPlanner::build_graph(std::vector<Vertex>& gskel) {
    PD.graph.nodes.clear();
    PD.graph.adj.clear();
    Graph G;

    std::unordered_map<int, std::vector<int>> vids_to_gid;
    vids_to_gid.reserve(gskel.size() * 2);
    
    // Create nodes
    for (const Vertex& v : gskel) {
        for (int k=0; k<static_cast<int>(v.vpts.size()); ++k) {
            const Viewpoint& vp = v.vpts[k];

            // if (vp.invalid) continue;
            GraphNode n;
            n.gid = static_cast<int>(G.nodes.size());
            n.vid = v.vid;
            n.k = k;
            n.p = vp.position;
            n.yaw = vp.yaw;
            n.score = vp.score;
            vids_to_gid[v.vid].push_back(n.gid);
            G.nodes.push_back(std::move(n));
        }
    }

    G.adj.resize(G.nodes.size());
    if (G.nodes.empty()) {
        PD.graph = std::move(G);
        return 0;
    }
    
    std::unordered_set<uint64_t> edge_set;
    edge_set.reserve(G.nodes.size() * 4);
    for (const auto& u : gskel) {
        auto itU = vids_to_gid.find(u.vid);
        if (itU == vids_to_gid.end() || itU->second.empty()) continue;
        
        if (u.type == 1) {
            // connect in order
            const auto& gids = itU->second;
            const int m = static_cast<int>(gids.size());
            if (m < 2) continue;

            for (int i=0; i+1<m; ++i) {
                int g0 = gids[i];
                int g1 = gids[i+1];
                const auto& p0 = G.nodes[g0].p;
                const auto& p1 = G.nodes[g1].p;
                const float dd2 = d2(p0, p1);
                if (line_of_sight(p0, p1)) {
                    add_edge(G, g0, g1, std::sqrt(dd2), edge_set, true); // add topological graph
                }
            }
            continue;
        }
        else {
            for (int nb_vid : u.nb_ids) {
                const int nb_idx = gskel_vid2idx[nb_vid];
                const auto& v = gskel[nb_idx];
                auto itV = vids_to_gid.find(v.vid);
                if (itV == vids_to_gid.end() || itV->second.empty()) continue;

                const auto& u_gids = itU->second;
                const auto& v_gids = itV->second;

                for (int gu : u_gids) {
                    const auto& pu = G.nodes[gu].p;
                    int best_gv = -1;
                    float best_d2 = std::numeric_limits<float>::infinity();

                    for (int gv : v_gids) {
                        const auto& pv = G.nodes[gv].p;
                        float dd2 = d2(pu, pv);
                        if (dd2 < best_d2 && line_of_sight(pu, pv)) {
                            best_d2 = dd2;
                            best_gv = gv;
                        }
                    }

                    if (best_gv >= 0) {
                        add_edge(G, gu, best_gv, std::sqrt(best_d2), edge_set, true); // add topological graph
                    }
                }
            }
        }
    }

    PD.graph = std::move(G);
    return 1;
}

bool PathPlanner::generate_path(std::vector<Vertex>& gskel) {

    return 1;
}


/* HELPERS */

bool PathPlanner::line_of_sight(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    if (!octree_) {
        return 0; // nothing to do...
    }

    Eigen::Vector3f d = b - a;
    float L = d.norm();
    if (L <= 1e-6f) return 1;
    Eigen::Vector3f u = d / L;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::AlignedPointTVector centers;
    octree_->getIntersectedVoxelCenters(a, u, centers);

    const float tol = 0.6f * cfg_.map_voxel_size;
    const float tol2 = tol * tol;
    for (const auto& c : centers) {
        Eigen::Vector3f cv = c.getVector3fMap();
        if ((cv - a).squaredNorm() <= tol2) continue;
        if ((cv - b).squaredNorm() <= tol2) continue;
        return 0;
    }
    return 1;
}



/*
IDEAS:
- Start: Drone position 

- Some path cost proportional to the steps needed along the skeleton adjacency
- Some path reward

*/








