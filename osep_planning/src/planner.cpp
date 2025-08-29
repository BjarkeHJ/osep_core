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

    build_graph(gskel);
    generate_path(gskel);
    return 1;
}

bool PathPlanner::build_graph(std::vector<Vertex>& gskel) {
    PD.graph.nodes.clear();
    PD.graph.adj.clear();

    Graph G;
    std::unordered_map<uint64_t, int> handle2gid;
    handle2gid.reserve(2048);

    std::unordered_map<int, std::vector<int>> vids_to_gid;
    vids_to_gid.reserve(gskel.size() * 2);
    
    // Create nodes
    int gid = 0;
    for (const Vertex& v : gskel) {
        for (int k=0; k<static_cast<int>(v.vpts.size()); ++k) {
            const Viewpoint& vp = v.vpts[k];
            // if (vp.invalid) continue;

            GraphNode n;
            n.gid = gid;
            n.vid = v.vid;
            n.k = k;
            n.p = vp.position;
            n.yaw = vp.yaw;
            n.score = vp.score;

            G.nodes.push_back(n);
            vids_to_gid[v.vid].push_back(gid);
            handle2gid[ hk(v.vid, k) ] = gid;

            ++gid;
        }
    }

    G.adj.resize(G.nodes.size());
    if (G.nodes.empty()) return G;

    


    // const float vis_graph_radius = 20.0f;

    // vpt_cloud->points.clear();
    // std::vector<std::vector<int>> vidx2cidx(gskel.size());
    
    // std::unordered_map<int,int> cidx2vidx;
    // cidx2vidx.reserve(gskel.size());

    // pcl::PointXYZ vp_p;

    // int vidx = 0;
    // int cidx = 0;
    // for (Vertex& v : gskel) {
    //     for (Viewpoint& vp : v.vpts) {
    //         vp_p = {vp.position.x(), vp.position.y(), vp.position.z()};
    //         vpt_cloud->points.push_back(vp_p);
    //         vidx2cidx[vidx][cidx] = cidx;
    //         ++cidx;
    //     }
    //     ++vidx;
    // }

    // const int N = static_cast<int>(vpt_cloud->points.size());
    // vpt_kdtree.setInputCloud(vpt_cloud);
    // std::vector<int> ids;
    // std::vector<float> d2s;

    // for (int i=0; i<N; ++i) {
    //     ids.clear();
    //     d2s.clear();
    //     pcl::PointXYZ qp = vpt_cloud->points[i];
    //     int n_nbs = vpt_kdtree.radiusSearch(qp, vis_graph_radius, ids, d2s);

    //     for (int jj=0; jj<n_nbs; ++jj) {
    //         int j = ids[jj];
    //         if (j <= i) continue; // undirected graph



    //     }
    // }

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








