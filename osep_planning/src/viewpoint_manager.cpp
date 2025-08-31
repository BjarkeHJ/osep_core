/* 

Main algorithm for the OSEP viewpoint manager

TODO: Incorporate 2d_local_costmap into the viewpoint sampling process.

*/

#include "viewpoint_manager.hpp"

ViewpointManager::ViewpointManager(const ViewpointConfig& cfg) : cfg_(cfg) {
    gmap.reset(new pcl::PointCloud<pcl::PointXYZ>);
    running = 1;
}

bool ViewpointManager::update_viewpoints(std::vector<Vertex>& gskel) {
    /* Main public function */
    running = sample_viewpoints(gskel);
    running = filter_viewpoints(gskel);
    return running;
}

bool ViewpointManager::sample_viewpoints(std::vector<Vertex>& gskel) {
    if (gskel.empty()) return 0;

    for (int i=0; i<static_cast<int>(gskel.size()); ++i) {
        Vertex& v = gskel[i];
        if (v.vid != i) std::cout << "WARNING: VID DOES NOT MATCH SKEL INDEX" << std::endl;
        
        const bool need_resample = v.type_update || v.pos_update || v.vpts.empty();
        if (!need_resample) continue;

        std::vector<Viewpoint> new_vpts = generate_viewpoints(gskel, v);
        v.vpts = std::move(new_vpts);
    }
    return 1;
}

bool ViewpointManager::filter_viewpoints(std::vector<Vertex>& gskel) {
    const int MAX_ATTEMPTS = 10;
    for (Vertex& v : gskel) {
        for (Viewpoint& vp : v.vpts) {
            if (is_not_safe_dist(vp.position)) {
                const Eigen::Vector3f dir(std::cos(vp.yaw), std::sin(vp.yaw), 0.0f);
                const float step = std::max(0.5f * static_cast<float>(octree_->getResolution()), 0.05f);
                bool fixed = false;
                for (int i=0; i<MAX_ATTEMPTS; ++i) {
                    Eigen::Vector3f trial = vp.position - dir*step*(i+1);
                    if (!is_not_safe_dist(trial)) {
                        vp.position = trial;
                        vp.updated = true;
                        fixed = true;
                        break;
                    }
                }
                if (!fixed) {
                    vp.invalid = true; // Change later to delted vp
                    continue;
                }
            }
        }
    }

    return 1;
}

std::vector<Viewpoint> ViewpointManager::generate_viewpoints(std::vector<Vertex>& gskel, Vertex& v) {
    std::vector<Viewpoint> out;
    Eigen::Vector3f that, n1hat, n2hat;
    build_local_frame(gskel, v, that, n1hat, n2hat);

    const int type = v.type;
    const Eigen::Vector3f v_pos = v.position.getVector3fMap();

    std::vector<float> phis;
    if (type == 1) {
        const float fan_deg = 180.0f;
        const int K_fan = 6;
        Eigen::Vector2f dir_xy = -that.head<2>();
        dir_xy.normalize();
        const float phi0 = std::atan2(dir_xy.y(), dir_xy.x());
        const float half = deg2rad(fan_deg * 0.5f);
        for (int i=0; i<K_fan; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(K_fan - 1);
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
        // Default -> Do nothing...
    }

    auto voxel_occupied = [&](const Eigen::Vector3f& p) -> bool {
        std::vector<int> ids;
        pcl::PointXYZ q(p.x(), p.y(), p.z());
        octree_->voxelSearch(q, ids);
        return !ids.empty();
    };

    const float S = octree_->getResolution();
    const float MIN_SEP = cfg_.vpt_safe_dist;
    const int MAX_ATTEMPTS = 10;
    const float STEP = std::max(0.5f * S, 0.05f);

    int pos_id = 0;
    for (float phi : phis) {
        Eigen::Vector3f u(std::cos(phi), std::sin(phi), 0.0f);
        u.normalize();
        float dtfs = distance_to_free_space(v_pos, u); // not currently working
        if (dtfs < 0.0f) {
            dtfs = 0.0f; // change later!
        }

        float d = MIN_SEP + dtfs;
        Eigen::Vector3f vp_pos = v_pos + u * d;

        int tries = 0;
        while (tries < MAX_ATTEMPTS && (voxel_occupied(vp_pos) || d < MIN_SEP)) {
            vp_pos += u * STEP;
            d += STEP;
            ++tries;
        }

        if (voxel_occupied(vp_pos)) continue;

        float yaw = yaw_to_face(vp_pos, v_pos);
        Viewpoint vp;

        uint32_t& seq = per_vertex_seg[v.vid];
        vp.vptid = make_vpt_handle(v.vid, seq++); // give viewpoint an unique identifier (pack vid and uniquie vpt id)

        vp.position = vp_pos;
        vp.yaw = yaw;
        vp.orientation = yaw_to_quat(yaw);
        vp.target_vid = v.vid;
        vp.target_vp_pos = pos_id;
        vp.updated = true;
        out.push_back(vp);
        ++pos_id;
    }
    return out;
}

void ViewpointManager::build_local_frame(std::vector<Vertex>& gskel, Vertex& v, Eigen::Vector3f& that, Eigen::Vector3f& n1hat, Eigen::Vector3f& n2hat) {
    that  = Eigen::Vector3f::UnitX();
    n1hat = Eigen::Vector3f::UnitY();
    n2hat = Eigen::Vector3f::UnitZ();

    if (v.nb_ids.size() < 1) return;

    Eigen::MatrixXf M(3, std::max<size_t>(v.nb_ids.size(), 1));
    for (size_t i=0; i<v.nb_ids.size(); ++i) {
        const int nb_id = v.nb_ids[i]; // nb vertex vid
        const int gskel_nb_id = gskel_vid2idx[nb_id]; // Fetch gskel vector index -> ensure correct nb vid
        const auto& nb_v = gskel[gskel_nb_id];

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









// std::vector<int> ViewpointManager::walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges) {
//     auto edge_key = [](int u, int v) {
//         if (u > v) std::swap(u,v);
//         return std::make_pair(u,v);
//     };

//     auto edge_seen = [&](int a, int b) {
//         return visited_edges.count(edge_key(a,b)) != 0;
//     };
    
//     auto mark_edge = [&](int a, int b) {
//         visited_edges.insert(edge_key(a,b));
//     };

//     std::vector<int> out_vids;
//     out_vids.reserve(32);

//     if (start_idx < 0 || nb_idx < 0) return out_vids; // invalid
//     if (!allowed[start_idx] || !allowed[nb_idx]) return out_vids; // not allowed
//     if (edge_seen(start_idx, nb_idx)) return out_vids; // already seen

//     int prev = start_idx;
//     int curr = nb_idx;

//     out_vids.push_back(idx2vid_[start_idx]);

//     while (true) {
//         out_vids.push_back(idx2vid_[curr]);
//         mark_edge(prev, curr);

//         if (is_endpoint_[curr] && curr != start_idx) break; // found endpoint
//         int next_idx = -1;

//         for (int nb_i : VD.gadj[curr]) {
//             if (nb_i == prev) continue;
//             if (!allowed[nb_i]) continue;
//             if (!edge_seen(curr, nb_i)) {
//                 next_idx = nb_i;
//                 break;
//             }
//         }

//         // fallback: any nb neq prev
//         if (next_idx == -1) {
//             for (int nb_i : VD.gadj[curr]) {
//                 if (nb_i != prev) {
//                     next_idx = nb_i;
//                     break;
//                 }
//             }
//         }

//         if (next_idx == -1) break; // no next found!

//         prev = curr;
//         curr = next_idx;
//     }
    
//     if (out_vids.size() < 2) {
//         out_vids.clear();
//     }

//     return out_vids;
// }





