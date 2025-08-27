/* 

Main algorithm for the OSEP viewpoint manager

TODO: Incorporate 2d_local_costmap into the viewpoint sampling process.

*/

#include "viewpoint_manager.hpp"

ViewpointManager::ViewpointManager(const ViewpointConfig& cfg) : cfg_(cfg) {
    VD.global_map.reset(new pcl::PointCloud<pcl::PointXYZ>);
    VD.global_map->points.reserve(50000);
    VD.global_skel.reserve(1000);
    VD.global_vpts.reserve(5000);
    VD.updated_vertices.reserve(100);
    running = 1;

    // octree_ = std::make_shared<pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ>>(cfg_.map_voxel_size);
    octree_ = std::make_shared<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>>(cfg_.map_voxel_size);
}

bool ViewpointManager::viewpoint_run() {
    VD.gskel_size = VD.global_skel.size();
    RUN_STEP(fetch_updated_vertices);

    for (int i=0; i<static_cast<int>(VD.global_skel.size()); ++i) {
        if (i != VD.global_skel[i].vid) {
            std::cout << "PROBLEM!!!!" << std::endl;
        }
    }

    RUN_STEP(branch_extract);
    RUN_STEP(viewpoint_sampling);
    RUN_STEP(build_all_vpts);

    // std::cout << "Viewpoints size: " << VD.global_vpts.size() << std::endl;

    return running;
}

bool ViewpointManager::fetch_updated_vertices() {
    if (VD.gskel_size == 0) return 0;
    VD.updated_vertices.clear();
    for (const auto& v : VD.global_skel) {
        if (v.pos_update || v.type_update) {
            VD.updated_vertices.push_back(v.vid);
        }
    }

    // Rebuild graph cache
    const int N = VD.gskel_size;
    vid2idx_.clear();
    vid2idx_.reserve(N*2);
    idx2vid_.clear();
    idx2vid_.assign(N, -1);
    
    for (int i=0; i<N; ++i) {
        vid2idx_[VD.global_skel[i].vid] = i;
        idx2vid_[i] = VD.global_skel[i].vid;
    }
    
    VD.global_adj.assign(N, {});
    degree_.assign(N,0);
    is_endpoint_.assign(N,0);

    for (int i=0; i<N; ++i) {
        const auto& nbs = VD.global_skel[i].nb_ids;
        auto& out = VD.global_adj[i];
        out.reserve(nbs.size());
        for (int nb_vid : nbs) {
            auto it = vid2idx_.find(nb_vid);
            if (it != vid2idx_.end()) {
                out.push_back(it->second);
            }
        }
        degree_[i] = static_cast<int>(out.size());
    }
    for (int i=0; i<N; ++i) {
        int d = degree_[i];
        is_endpoint_[i] = (d == 1 || d > 2) ? 1 : 0;
    }

    octree_->setInputCloud(VD.global_map);
    octree_->addPointsFromInputCloud();
    // octree_->deleteTree();
    // octree_->addPointsFromInputCloud();

    return 1;
}

bool ViewpointManager::branch_extract() {
    const int N = VD.gskel_size;
    if (N == 0) return 0;
    const int branch_min_vers_ = 5;

    std::vector<int> updated_idxs;
    updated_idxs.reserve(VD.updated_vertices.size());
    for (int vid : VD.updated_vertices) {
        auto it = vid2idx_.find(vid);
        if (it != vid2idx_.end()) {
            updated_idxs.push_back(it->second);
        }
    }

    // const bool full_rebuild = VD.branches.empty() || updated_idxs.empty();
    const bool full_rebuild = VD.branches.empty();
    // const bool full_rebuild = true;

    if (full_rebuild) {
        VD.branches.clear();
        std::vector<char> allowed(N,1);
        std::unordered_set<std::pair<int,int>, PairHash> visited;
        visited.reserve(std::max(64, 2*N));

        for (int i=0; i<N; ++i) {
            if (!is_endpoint_[i]) continue; // start at endpoint
            for (int nb_i : VD.global_adj[i]) {
                auto br = walk_branch(i, nb_i, allowed, visited);
                if (!br.empty()) {
                    const bool enough_vers = (static_cast<int>(br.size()) - 1) >= branch_min_vers_;
                    if (enough_vers) VD.branches.emplace_back(std::move(br));
                }
            }
        }
        return 1;
    }

    // Incremental build
    std::vector<char> in_region(N,0);
    std::vector<int> stack;
    stack.reserve(256);

    auto push_idx = [&](int i) {
        if (i < 0 || i >= N) return;
        if (!in_region[i]) {
            in_region[i] = 1; // mark index as in region with updated vertex
            stack.push_back(i);
        }
    };

    // Seed the region with updated vertices and their neighbors
    for (int ui : updated_idxs) {
        push_idx(ui);
        for (int nb_i : VD.global_adj[ui]) {
            push_idx(nb_i);
        }
    }

    // DFS flood-fill the region and stopping at endpoints
    while (!stack.empty()) {
        int i = stack.back();
        stack.pop_back();
        if (is_endpoint_[i]) continue; // stop expanding the region at an endpoint (reached depth in search - trace back)
        for (int nb_i : VD.global_adj[i]) {
            push_idx(nb_i);
        }
    }

    bool any_in = false;
    for (char c : in_region) {
        if (c) {
            any_in = true;
            break;
        }
    }

    if (!any_in) return 1; // nothing to do...

    auto edge_key= [](int u, int v) {
        if (u > v) std::swap(u,v);
        return std::make_pair(u,v);
    };
    
    // Build the region edges
    std::unordered_set<std::pair<int,int>, PairHash> region_edges;
    region_edges.reserve(8 * (int)updated_idxs.size() + 64);
    for (int i=0; i<N; ++i) {
        if (!in_region[i]) continue; // focus on the region (skip non-update areas)
        for (int j : VD.global_adj[i]) {
            if (!in_region[j]) continue;
            if (j <= i) continue; // avoid duplicate edges (undirected...)
            region_edges.insert(edge_key(i,j));
        }
    }

    auto branch_hits_region = [&](const std::vector<int>& br_vids) -> bool {
        if (br_vids.size() < 2) return false;
        for (size_t k=1; k<br_vids.size(); ++k) {
            auto ita = vid2idx_.find(br_vids[k-1]);
            auto itb = vid2idx_.find(br_vids[k]);
            if (ita == vid2idx_.end() || itb == vid2idx_.end()) continue;
            if (region_edges.count(edge_key(ita->second, itb->second))) {
                return true;
            }
        }
        return false;
    };

    // Remove any previously stored branches that crosses region edges
    std::vector<std::vector<int>> kept;
    kept.reserve(VD.branches.size());
    for (auto& br : VD.branches) {
        if (!branch_hits_region(br)) {
            kept.emplace_back(std::move(br));
        }
    }
    VD.branches.swap(kept);

    // Reseed new branch walks from endpoints inside the update region
    std::unordered_set<std::pair<int,int>, PairHash> visited_local;
    visited_local.reserve(region_edges.size());
    for (int i=0; i<N; ++i) {
        if (!in_region[i]) continue;
        if (!is_endpoint_[i]) continue;
        for (int nb_i : VD.global_adj[i]) {
            // if (!in_region[nb_i]) continue;
            auto br = walk_branch(i, nb_i, in_region, visited_local);
            if (!br.empty()) {
                const bool enough_vers = (static_cast<int>(br.size()) - 1) >= branch_min_vers_;
                if (enough_vers) VD.branches.emplace_back(std::move(br));
            }
        }
    }
    return 1;
}

bool ViewpointManager::viewpoint_sampling() {
    for (int i=0; i<static_cast<int>(VD.gskel_size); ++i) {
        auto& vertex = VD.global_skel[i];
        if (vertex.vid != (int)i) std::cout << "WARNING: VID DOES NOT MATCH SKEL INDEX!" << std::endl;

        const bool need_resample = (vertex.type_update || vertex.pos_update || vertex.vpts.empty());
        if (!need_resample) continue;

        erase_handles_for_vertex(i); // clear global_vpt_handles for the vid

        std::vector<Viewpoint> new_vpts = generate_viewpoint(i);
        vertex.vpts = std::move(new_vpts);

        append_handles_for_vertex(i);

        vertex.type_update = false;
        vertex.pos_update = false;

    }
    return 1;
}

bool ViewpointManager::build_all_vpts() {
    VD.global_vpts_handles.clear();
    VD.global_vpts.clear();
    for (int vid=0; vid<(int)VD.global_skel.size(); ++vid) {
        auto& v = VD.global_skel[vid];
        for (int j=0; j<(int)v.vpts.size(); ++j) {
            VptHandle vphndl{vid, j};
            VD.global_vpts_handles.push_back(vphndl);
        }
    }
    return 1;
}

/* Helpers */
std::vector<Viewpoint> ViewpointManager::generate_viewpoint(int idx) {
    std::vector<Viewpoint> out;

    // that is the direction of the skeleton - n1hat and n2hat forms the orthonormal basis at the vertex
    Eigen::Vector3f that, n1hat, n2hat;
    build_local_frame(idx, that, n1hat, n2hat);
    
    auto& vertex = VD.global_skel[idx];
    const int type = vertex.type;
    const Eigen::Vector3f vertex_pos = vertex.position.getVector3fMap();

    std::vector<float> phis; // resulting azimuthal angles relative to skeleton direction
    if (type == 1) {
        const float fan_deg = 180.0f;
        const int K_fan = 6;
        Eigen::Vector2f dir_xy = -that.head<2>();
        dir_xy.normalize();
        const float phi0 = std::atan2(dir_xy.y(), dir_xy.x());
        const float half = deg2rad(fan_deg * 0.5f);
        for (int i=0; i<K_fan; ++i) {
            float t = float(i) / float(K_fan - 1);
            float phi = phi0 - half + t * (2.0f * half);
            phis.push_back(wrapPi(phi));
        }
    }
    else if (type == 2) {
        Eigen::Vector2f dir_xy = that.head<2>();
        dir_xy.normalize();
        const float phi0 = std::atan2(dir_xy.y(), dir_xy.x());
        phis.push_back(phi0 + M_PI_2);
        phis.push_back(phi0 - M_PI_2);
    }
    else if (type == 3) {
        // Joint -> Do nothing for now
    }
    else {
        // Default -> Do nothing
    }

    auto voxel_occupied = [&](const Eigen::Vector3f& q) -> bool {
        std::vector<int> idxs;
        octree_->voxelSearch(pcl::PointXYZ(q.x(), q.y(), q.z()), idxs);
        return !idxs.empty();
    };

    const float S = octree_->getResolution();
    const float MIN_SEP = cfg_.vpt_disp_dist;
    const int MAX_ATTEMPTS = 10;
    const float STEP = std::max(0.5f * S, 0.05f);

    // build viewpoints
    for (float phi : phis) {
        Eigen::Vector3f u(std::cos(phi), std::sin(phi), 0.0f);
        u.normalize();
        float dtfs = distance_to_free_space(vertex_pos, u);
        if (dtfs < 0.0f) {
            dtfs = 0.0f; // CHANGE THIS LATER... It should generally continue here 
        }

        float d = MIN_SEP + dtfs;
        Eigen::Vector3f vp_pos = vertex_pos + u * d;

        int tries = 0;
        while (tries < MAX_ATTEMPTS && (voxel_occupied(vp_pos) || d < MIN_SEP)) {
            vp_pos += u * STEP;
            d += STEP;
            ++tries;
        }

        if (voxel_occupied(vp_pos)) continue;

        float yaw = yaw_to_face(vp_pos, vertex_pos);
        Viewpoint vp;
        vp.position = vp_pos;
        vp.orientation = yaw_to_quat(yaw);
        vp.target_vid = vertex.vid;
        out.push_back(vp);
    }
    return out;
}

void ViewpointManager::build_local_frame(const int vid, Eigen::Vector3f& that, Eigen::Vector3f& n1hat, Eigen::Vector3f& n2hat) {
    const auto& v = VD.global_skel[vid];

    that  = Eigen::Vector3f::UnitX();
    n1hat = Eigen::Vector3f::UnitY();
    n2hat = Eigen::Vector3f::UnitZ();

    if (v.nb_ids.size() < 1) return;

    Eigen::MatrixXf M(3, std::max<size_t>(v.nb_ids.size(), 1));
    for (size_t i=0; i<v.nb_ids.size(); ++i) {
        const int nb_id = v.nb_ids[i];
        const auto& nb_v = VD.global_skel[nb_id];
        Eigen::Vector3f e = nb_v.position.getVector3fMap() - v.position.getVector3fMap();
        M.col(int(i)) = e;
    }
    if (v.nb_ids.size() == 1) {
        that = (M.col(0).norm() > 1e-6) ? M.col(0).normalized() : Eigen::Vector3f::UnitX();
    }
    else {
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeThinU);
        that = svd.matrixU().col(0);
    }

    if (that.norm() < 1e-6f) that = Eigen::Vector3f::UnitX();
    that.normalize();

    // If that is too close to world-up (~parallel) -> cross with world x-dir instead
    const Eigen::Vector3f up(0,0,1);
    n1hat = (std::abs(that.dot(up)) < 0.95f) ? (up.cross(that)).normalized() : (Eigen::Vector3f::UnitX().cross(that)).normalized();
    n2hat = that.cross(n1hat).normalized();
}

float ViewpointManager::distance_to_free_space(const Eigen::Vector3f& p_in, const Eigen::Vector3f dir_in) {
    if (!octree_) return -1.0f;
    const auto& cloud = octree_->getInputCloud();
    if (!cloud || cloud->empty()) return -1.0f;

    const float EPS = 1e-6f;
    const float MAX_DIST = 10.0f;
    const float S = octree_->getResolution();

    // Normalize direction so returned t is in meters
    Eigen::Vector3f dir = dir_in;
    float n = dir.norm();
    // if (n < EPS) return -1.0f;
    if (n < EPS) {
        std::cout << "Returned due to small norm!" << std::endl; 
        return -1.0f;
    }
    dir /= n;

    // Nudge off boundaries
    Eigen::Vector3f p = p_in + EPS * dir;

    // Helper: a voxel is "occupied" if voxelSearch finds any points in it
    auto is_occ = [&](const Eigen::Vector3f& q) -> bool {
        std::vector<int> idx;
        pcl::PointXYZ pq(q.x(), q.y(), q.z());
        octree_->voxelSearch(pq, idx);
        return !idx.empty();
    };

    // Collect ray-intersected voxel centers (ordered along the ray)
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::AlignedPointTVector centers;
    // max_voxels=0 means “no explicit cap”; we’ll clamp by MAX_DIST below
    // octree_->getIntersectedVoxelCenters(p, dir, centers, 0);
    Eigen::Vector3f ray_vec = dir * (MAX_DIST + 0.5f * S);
    octree_->getIntersectedVoxelCenters(p, ray_vec, centers, 0);

    bool phase_occ = is_occ(p);  // Are we starting inside occupied?

    for (const auto& c_pt : centers) {
        // signed distance to this voxel center along the ray
        Eigen::Vector3f c(c_pt.x, c_pt.y, c_pt.z);
        float along = (c - p).dot(dir);
        if (along < 0.0f) continue;                   // behind the start
        if (along > MAX_DIST + 0.5f * S) break;       // beyond search range

        // Bias a hair *forward* along the ray so the check samples inside that voxel
        bool nowOcc = is_occ(c + 1e-5f * dir);

        if (phase_occ && !nowOcc) {
            // OCC -> FREE transition: approximate boundary at half a cell before center
            float t_boundary = std::max(0.0f, along - 0.5f * S);
            std::cout << "RETURNED CORRECTLY!" << std::endl;
            return t_boundary;
        }
        if (!phase_occ && nowOcc) {
            // First entered occupied, now keep looking for the exit
            phase_occ = true;
        }
    }

    // No OCC->FREE transition within MAX_DIST
    return -1.0f;
}

std::vector<int> ViewpointManager::walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges) {
    auto edge_key = [](int u, int v) {
        if (u > v) std::swap(u,v);
        return std::make_pair(u,v);
    };

    auto edge_seen = [&](int a, int b) {
        return visited_edges.count(edge_key(a,b)) != 0;
    };
    
    auto mark_edge = [&](int a, int b) {
        visited_edges.insert(edge_key(a,b));
    };

    std::vector<int> out_vids;
    out_vids.reserve(32);

    if (start_idx < 0 || nb_idx < 0) return out_vids; // invalid
    if (!allowed[start_idx] || !allowed[nb_idx]) return out_vids; // not allowed
    if (edge_seen(start_idx, nb_idx)) return out_vids; // already seen

    int prev = start_idx;
    int curr = nb_idx;

    out_vids.push_back(idx2vid_[start_idx]);

    while (true) {
        out_vids.push_back(idx2vid_[curr]);
        mark_edge(prev, curr);

        if (is_endpoint_[curr] && curr != start_idx) break; // found endpoint
        int next_idx = -1;

        for (int nb_i : VD.global_adj[curr]) {
            if (nb_i == prev) continue;
            if (!allowed[nb_i]) continue;
            if (!edge_seen(curr, nb_i)) {
                next_idx = nb_i;
                break;
            }
        }

        // fallback: any nb neq prev
        if (next_idx == -1) {
            for (int nb_i : VD.global_adj[curr]) {
                if (nb_i != prev) {
                    next_idx = nb_i;
                    break;
                }
            }
        }

        if (next_idx == -1) break; // no next found!

        prev = curr;
        curr = next_idx;
    }
    
    if (out_vids.size() < 2) {
        out_vids.clear();
    }

    return out_vids;
}























// float ViewpointManager::distance_to_free_space(const Eigen::Vector3f& p_in, const Eigen::Vector3f dir_in) {
//     const float EPS = 1e-6f;
//     const float MAX_DIST_TO_FREE = 10.0f;
//     const float S = octree_->getResolution();

//     Eigen::Vector3f dir = dir_in;
//     const float n = dir.norm();
//     if (n < EPS) return -1.0f;
//     dir /= n;

//     Eigen::Vector3f p = p_in + (EPS * dir); // nudge off boundaries

//     auto occ_at_point = [&](const Eigen::Vector3f& q) -> bool {
//         return octree_->isVoxelOccupiedAtPoint(pcl::PointXYZ(q.x(), q.y(), q.z()));
//     };

//     bool occ = occ_at_point(p);
//     int phase = occ ? 1 : 0;

//     auto flo = [&](float x) { return std::floor(x / S); };
//     Eigen::Vector3i v((int)flo(p.x()), (int)flo(p.y()), (int)flo(p.z()));

//     const int sx = (dir.x() > 0.0) ? +1 : (dir.x() < 0.0 ? -1 : 0);
//     const int sy = (dir.y() > 0.0) ? +1 : (dir.y() < 0.0 ? -1 : 0);
//     const int sz = (dir.z() > 0.0) ? +1 : (dir.z() < 0.0 ? -1 : 0);
    
//     auto nextBoundary = [&](int axis, int vi, int s) -> float {
//         if (s > 0) return (vi + 1) * S;
//         if (s < 0) return vi * S;
//         return (dir[axis] >= 0.0f) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
//     };
//     auto sdiv = [&](float num, float den) -> float {
//         return (std::abs(den) < EPS) ? std::numeric_limits<float>::infinity() : (num / den);
//     };

//     float bx = nextBoundary(0, v.x(), sx);
//     float by = nextBoundary(1, v.y(), sy);
//     float bz = nextBoundary(2, v.z(), sz);

//     float tMaxX = sdiv(bx - p.x(), dir.x());
//     float tMaxY = sdiv(by - p.y(), dir.y());
//     float tMaxZ = sdiv(bz - p.z(), dir.z());

//     const float tDeltaX = (sx==0) ? std::numeric_limits<float>::infinity() : std::abs(S/dir.x());
//     const float tDeltaY = (sy==0) ? std::numeric_limits<float>::infinity() : std::abs(S/dir.y());
//     const float tDeltaZ = (sz==0) ? std::numeric_limits<float>::infinity() : std::abs(S/dir.z());

//     float t = 0.f;
//     const float TIE_EPS = 1e-9f;

//     while (t <= MAX_DIST_TO_FREE) {
//         float tNext = std::min(tMaxX, std::min(tMaxY, tMaxZ));
//         bool stepX = (tNext <= tMaxX + TIE_EPS);
//         bool stepY = (tNext <= tMaxY + TIE_EPS);
//         bool stepZ = (tNext <= tMaxZ + TIE_EPS);

//         if (stepX) { v.x() += sx; tMaxX += tDeltaX; }
//         if (stepY) { v.y() += sy; tMaxY += tDeltaY; }
//         if (stepZ) { v.z() += sz; tMaxZ += tDeltaZ; }

//         t = tNext;
//         if (t > MAX_DIST_TO_FREE) break;

//         // Sample a point just inside the newly entered voxel along the ray
//         Eigen::Vector3f q = p + (t + 1e-5f) * dir;
//         bool nowOcc = occ_at_point(q);

//         if (phase == 1 && !nowOcc) return t; // OCC → FREE
//         if (phase == 0 &&  nowOcc) phase = 1; // FREE → OCC
//     }

//     // No transition within range
//     std::cout << "Did not reach free space!" << std::endl;
//     return -1.0f;
// }
