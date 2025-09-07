/* 

Main algorithm for global incremental skeletonization
a topologically preserving representation of the structure

Note: If I want to utilize the KF filter after "frozen" is triggered 
I must change such that when the position (of Vertex) is changed it changes the KF state as well...

TODO: Downsample the skeleton vertices further to extract more meaningfull vertices and longer edges (smaller vertex set)
      Long branches have few vertices: linear fitting ish...

*/

#include <gskel.hpp>

GSkel::GSkel(const GSkelConfig& cfg) : cfg_(cfg) {
    /* Constructor - Init data structures etc... */
    GD.new_cands.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GD.global_vers_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GD.global_vers_cloud->points.reserve(10000);
    GD.sparse_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    running = 1;
}

bool GSkel::gskel_run() {
    // RUN_STEP(increment_skeleton);
    if (GD.new_cands->empty()) return 1;

    RUN_STEP(new_increment_skeleton);
    RUN_STEP(graph_adj);
    RUN_STEP(mst);
    RUN_STEP(vertex_merge);
    RUN_STEP(prune);
    RUN_STEP(smooth_vertex_positions);
    RUN_STEP(vid_manager);
    RUN_STEP(resample_skeleton); // try removing 

    running = 1;
    return running;
}

bool GSkel::new_increment_skeleton() {
    if (!GD.new_cands) return 0;

    auto& in_cloud = *GD.new_cands;
    if (in_cloud.empty()) {
        build_cloud_from_vertices(); // still refresh published cloud
        return true;
    }

    const float fuse_r  = cfg_.fuse_dist_th;
    const float fuse_r2 = fuse_r * fuse_r;

    // Tunables (derive from map voxel size if you have it)
    const float sig_t_meas = std::max(2.0f * cfg_.lkf_mn,  1e-3f); // along branch
    const float sig_n_meas = std::max(0.5f * cfg_.lkf_mn,  1e-4f); // across
    const float sig_t_proc = std::max(3.0f * cfg_.lkf_pn,  1e-4f);
    const float sig_n_proc = std::max(0.3f * cfg_.lkf_pn,  1e-5f);
    const float gate2      = 9.0f; // ~3-sigma ellipsoidal gate

    const int   min_conf_obs     = 3;   // need at least this many hits
    const float conf_norm_trace  = cfg_.fuse_conf_th; // threshold on normal-plane covariance
    const int   max_unconf_steps = cfg_.max_obs_wo_conf;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prelim_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    prelim_cloud->points.reserve(GD.prelim_vers.size());
    for (const auto& v : GD.prelim_vers) {
        prelim_cloud->points.push_back(v.position);
    }
    kd_tree_->setInputCloud(prelim_cloud);

    std::vector<int> idx_buff;
    idx_buff.reserve(32);
    std::vector<float> dist2_buff;
    dist2_buff.reserve(32);
    std::vector<pcl::PointXYZ> spawned_this_tick;
    spawned_this_tick.reserve(64);

    // For each incoming ROSA point -> Fuse into global (update existing / spawn new)
    for (const auto& pt : in_cloud.points) {
        if (!pcl::isFinite(pt)) continue;

        Eigen::Vector3f z = const_cast<pcl::PointXYZ&>(pt).getVector3fMap(); // measurement

        bool any_in_radius = false;
        bool frozen_in_radius = false;

        int chosen_idx = -1;
        float best_normal_md2 = std::numeric_limits<float>::infinity();

        if (!prelim_cloud->points.empty()) {
            idx_buff.clear();
            dist2_buff.clear();

            if (kd_tree_->radiusSearch(pt, fuse_r, idx_buff, dist2_buff) > 0) {
                any_in_radius = true;

                for (size_t k=0; k<idx_buff.size(); ++k) {
                    const int i = idx_buff[k];
                    if (i < 0 || i >= static_cast<int>(GD.prelim_vers.size())) continue;

                    auto& gv = GD.prelim_vers[i];
                    if (gv.frozen) {
                        frozen_in_radius = true;
                        continue;
                    }

                    // ensure local frame exists
                    LocalFrame lf;
                    {
                        std::vector<int> nn_idx;
                        std::vector<float> nn_d2;
                        nn_idx.reserve(16);
                        nn_d2.reserve(16);
                        kd_tree_->radiusSearch(gv.position, fuse_r, nn_idx, nn_d2);
                        lf = refineLocalFrameFromNeighbors(gv.kf.x, nn_idx, *prelim_cloud);
                    }

                    if (!lf.valid) {
                        lf = defaultFrame();
                    }

                    // ellipsoidal gate
                    if (!passGate(gv.kf.x, z, lf.U, sig_t_meas, sig_n_meas, gate2)) continue;

                    // prefer smallest normal residual (tolerant along tangent)
                    Eigen::Vector3f dl = lf.U.transpose() * (z - gv.kf.x);
                    float normal_md2 = (dl.y()*dl.y() + dl.z()*dl.z()) / (sig_n_meas*sig_n_meas);
                    if (normal_md2 < best_normal_md2) {
                        best_normal_md2 = normal_md2;
                        chosen_idx = i;
                    }
                }
            }
        }

        //prevent multiple spawns at the spot this tick
        auto near_spawned = [&]() {
            for (const auto& s : spawned_this_tick) {
                Eigen::Vector3f sv = const_cast<pcl::PointXYZ&>(s).getVector3fMap();
                if ((sv - z).squaredNorm() < fuse_r2) return true; 
            }
            return false;
        };

        // Update chosen preliminary vertex with anisotropic Q/R
        if (chosen_idx >= 0) {
            auto& gv = GD.prelim_vers[chosen_idx];

            // build a local frame at the updated state
            std::vector<int> nn_idx;
            std::vector<float> nn_d2;
            kd_tree_->radiusSearch(gv.position, fuse_r, nn_idx, nn_d2);
            LocalFrame lf = refineLocalFrameFromNeighbors(gv.kf.x, nn_idx, *prelim_cloud);
            if (!lf.valid) {
                lf = defaultFrame();
            }

            Eigen::Matrix3f Qw = makeAnisotropic(lf.U, sig_t_proc, sig_n_proc);
            Eigen::Matrix3f Rw = makeAnisotropic(lf.U, sig_t_meas, sig_n_meas);

            gv.kf.update(z, Qw, Rw);
            if (!gv.kf.x.allFinite()) {
                gv.marked_for_deletion = true;
                continue;
            }

            gv.position.getVector3fMap() = gv.kf.x;
            gv.obs_count++;
            gv.pos_update = true;

            // confidence check emphasizing normal-plane stability
            float normal_trace = gv.kf.P(1,1) + gv.kf.P(2,2);
            if (!gv.conf_check && gv.obs_count >= min_conf_obs && normal_trace < conf_norm_trace) {
                gv.conf_check = true;
                gv.just_approved = true; // approved this tick
                gv.frozen = true; // publish ready
                gv.unconf_check = 0;
            }
            else {
                gv.unconf_check++;
                if (gv.unconf_check > max_unconf_steps) {
                    gv.marked_for_deletion = true;
                }
            }
        }
        else {
            // no candidates passed gate
            // if near any occupied -> skip
            if (any_in_radius || frozen_in_radius) continue;
            if (near_spawned()) continue;

            Vertex nv;
            nv.vid = -1;
            nv.kf.initFrom(z, Eigen::Matrix3f::Identity()); // tune initial P?
            nv.position.getVector3fMap() = z;
            nv.obs_count = 1;
            nv.smooth_iters = cfg_.niter_smooth_vertex;
            nv.frozen = false;
            nv.conf_check = false;
            nv.unconf_check = 0;
            nv.pos_update = false;

            GD.prelim_vers.emplace_back(std::move(nv));
            spawned_this_tick.push_back(pt);
        }
    }

    // Garbarge collect: deleted marked vertices
    GD.prelim_vers.erase(
        std::remove_if(GD.prelim_vers.begin(), GD.prelim_vers.end(),
            [](const Vertex& v){ return v.marked_for_deletion || !pcl::isFinite(v.position); }),
        GD.prelim_vers.end() );

    // promote just approved
    GD.new_vers_indxs.clear();
    GD.global_vers.reserve(GD.global_vers.size() + GD.prelim_vers.size());
    
    for (auto& v : GD.prelim_vers) {
        if (!v.just_approved) continue;
        const float zg = v.position.z;
        if (zg <= cfg_.gnd_th) {
            v.just_approved = false;
            continue;
        }

        // assign stable vertex id
        v.vid = GD.next_vid++;
        GD.global_vers.emplace_back(v); // copy
        GD.new_vers_indxs.push_back(static_cast<int>(GD.global_vers.size() - 1)); //maybe vid instead - but should not matter as only related to this tick
        v.just_approved = false; // consume this flag
    }


    GD.gskel_size = GD.global_vers.size();
    // build_cloud_from_vertices();
    // rebuild_vid_index_map();
    return 1;
}

bool GSkel::increment_skeleton() {
    if (!GD.new_cands || GD.new_cands->points.empty()) return 0;

    // Reset update flags in global skeleton
    for (auto& vg : GD.global_vers) {
        vg.pos_update = false;
        vg.type_update = false;
    }

    // Reset flags
    for (auto& v : GD.prelim_vers) {
        v.just_approved = false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr prelim_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    prelim_cloud->points.reserve(GD.prelim_vers.size());
    for (const auto& v : GD.prelim_vers) {
        prelim_cloud->points.push_back(v.position);
    }

    if (!prelim_cloud->points.empty()) {
        kd_tree_->setInputCloud(prelim_cloud);
    }

    const float fuse_r2 = cfg_.fuse_dist_th * cfg_.fuse_dist_th;

    std::vector<int> idx_buff;
    std::vector<float> dist2_buff;
    idx_buff.reserve(16);
    dist2_buff.reserve(16);
    std::vector<pcl::PointXYZ> spawned_this_tick;
    spawned_this_tick.reserve(128);
    for (auto& pt : GD.new_cands->points) {
        Eigen::Vector3f ver = pt.getVector3fMap();
        if (!ver.allFinite()) continue;

        int chosen_idx = -1;
        float chosen_dist2 = std::numeric_limits<float>::max();
        bool any_in_radius = false;
        bool frozen_in_radius = false;
        if (!prelim_cloud->points.empty()) {
            idx_buff.clear();
            dist2_buff.clear();
            if (kd_tree_->radiusSearch(pt, cfg_.fuse_dist_th, idx_buff, dist2_buff) > 0) {
                any_in_radius = true;
                // pick nearest "non-frozen" candidate
                for (size_t k=0; k<idx_buff.size(); ++k) {
                    const int i = idx_buff[k];
                    auto& gver = GD.prelim_vers[i];
                    if (gver.frozen) {
                        frozen_in_radius = true;
                        continue;
                    }
                    if (dist2_buff[k] < chosen_dist2) {
                        chosen_dist2 = dist2_buff[k];
                        chosen_idx = i;
                    }
                }
            }
        }

        // To prevent spawning multiple at the same place this tick
        auto near_spawned = [&]() {
            for (const auto&s : spawned_this_tick) {
                Eigen::Vector3f sv = const_cast<pcl::PointXYZ&>(s).getVector3fMap();
                if ((sv - ver).squaredNorm() <= fuse_r2) return true;
            }
            return false;
        };

        if (chosen_idx >= 0 && chosen_dist2 <= fuse_r2) {
            auto& gver = GD.prelim_vers[chosen_idx];
            gver.kf.update(ver, Q, R);
            if (!gver.kf.x.allFinite()) {
                gver.marked_for_deletion = true;
                continue;
            }
            gver.position.getVector3fMap() = gver.kf.x; // overwrite position
            gver.obs_count++;

            const float trace = gver.kf.P.trace();
            if (!gver.conf_check && trace < cfg_.fuse_conf_th) {
                gver.conf_check = true;
                gver.just_approved = true;
                gver.frozen = true;
                gver.unconf_check = 0;
            }
            else {
                gver.unconf_check++;
            }

            if (gver.unconf_check > cfg_.max_obs_wo_conf) {
                gver.marked_for_deletion = true;
            }
        }
        else if (any_in_radius || frozen_in_radius) {
            // area occupied
            continue;
        }
        else if (!near_spawned()) {
            Vertex new_ver;
            Eigen::Matrix3f P0 = Eigen::Matrix3f::Identity();
            new_ver.kf.initFrom(ver, P0);
            new_ver.position.getVector3fMap() = ver;
            new_ver.obs_count++;
            new_ver.smooth_iters = cfg_.niter_smooth_vertex;
            GD.prelim_vers.emplace_back(std::move(new_ver));
            spawned_this_tick.push_back(pt);
        }
    }

    // Remove vertices marked for deletion
    GD.prelim_vers.erase(
        std::remove_if(
            GD.prelim_vers.begin(),
            GD.prelim_vers.end(),
            [](const Vertex& v) {
                return v.marked_for_deletion;
            }
        ),
        GD.prelim_vers.end()
    );

    // Pass confident vertices to the global skeleton
    GD.new_vers_indxs.clear();
    GD.global_vers.reserve(GD.global_vers.size() + GD.prelim_vers.size());
    for (auto& v : GD.prelim_vers) {
        if (!pcl::isFinite(v.position)) continue;

        if (v.just_approved && v.position.getVector3fMap().z() > cfg_.gnd_th) {
            GD.global_vers.emplace_back(v); // copy to global vertices
            GD.new_vers_indxs.push_back(GD.global_vers.size() - 1);
            v.just_approved = false;
        }
    }

    GD.gskel_size = GD.global_vers.size();

    if (GD.new_vers_indxs.size() == 0) {
        return 0; // No reason to proceed with the pipeline 
    }
    else return 1;
}


bool GSkel::graph_adj() {
    if (!GD.global_vers_cloud) return 0;

    build_cloud_from_vertices();
    std::vector<std::vector<int>> new_adj(GD.global_vers_cloud->points.size());
    kd_tree_->setInputCloud(GD.global_vers_cloud);
    const int K = 10;
    const float max_dist_th = 3.0f * cfg_.fuse_dist_th;

    std::vector<int> indices;
    std::vector<float> dist2;
    indices.reserve(K);
    dist2.reserve(K);
    for (size_t i=0; i<GD.gskel_size; ++i) {
        indices.clear();
        dist2.clear();
        const auto& vq = GD.global_vers[i];
        int n_nbs = kd_tree_->nearestKSearch(vq.position, K, indices, dist2);
        for (int j=1; j<n_nbs; ++j) {
            int nb_idx = indices[j];
            const auto& vnb = GD.global_vers[nb_idx];
            float dist_to_nb = (vq.position.getVector3fMap() - vnb.position.getVector3fMap()).norm();

            if (dist_to_nb > max_dist_th) continue;

            bool is_good_nb = true;
            for (int k=1; k<n_nbs; ++k) {
                if (k==j) continue;

                int other_nb_idx = indices[k];
                const auto& vnb_2 = GD.global_vers[other_nb_idx];
                float dist_nb_to_other = (vnb.position.getVector3fMap() - vnb_2.position.getVector3fMap()).norm();
                float dist_to_other = (vq.position.getVector3fMap() - vnb_2.position.getVector3fMap()).norm();

                if (dist_nb_to_other < dist_to_nb && dist_to_other < dist_to_nb) {
                    is_good_nb = false;
                    break;
                }
            }

            if (is_good_nb) {
                new_adj[i].push_back(nb_idx);
                new_adj[nb_idx].push_back(i);
            }
        } 
    }

    GD.global_adj = new_adj; // replace old adjencency
    return size_assert();
}

bool GSkel::mst() {
    if (GD.gskel_size == 0 || GD.global_adj.empty()) return 0;

    std::vector<Edge> mst_edges;
    for (size_t i=0; i<GD.gskel_size; ++i) {
        for (int nb : GD.global_adj[i]) {
            if (nb <= (int)i) continue; // avoid bi-directional check
            const Eigen::Vector3f& ver_i = GD.global_vers[i].position.getVector3fMap();
            const Eigen::Vector3f& ver_nb = GD.global_vers[nb].position.getVector3fMap();
            const float weight = (ver_i - ver_nb).norm();
            Edge new_edge;
            new_edge.u = i;
            new_edge.v = nb;
            new_edge.w = weight;
            mst_edges.push_back(new_edge);
        }
    }

    std::sort(mst_edges.begin(), mst_edges.end());
    UnionFind uf(GD.gskel_size);
    std::vector<std::vector<int>> mst_adj(GD.gskel_size);

    for (const auto& edge : mst_edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst_adj[edge.u].push_back(edge.v);
            mst_adj[edge.v].push_back(edge.u);
        }
    }

    GD.global_adj = std::move(mst_adj);
    return size_assert();
}

bool GSkel::vertex_merge() {
    int N_new = GD.new_vers_indxs.size();
    if (GD.gskel_size == 0 || N_new == 0) return 0;
    if (static_cast<float>(N_new) / static_cast<float>(GD.gskel_size) > 0.5) return 1; // dont prune in beginning...

    std::set<int> to_delete;

    for (int new_id : GD.new_vers_indxs) {
        if (new_id < 0 || new_id >= (int)GD.gskel_size) continue;
        if (to_delete.count(new_id)) continue;

        const auto& nbrs = GD.global_adj[new_id];
        for (int nb_id : nbrs) {
            if (nb_id < 0 || nb_id >= (int)GD.gskel_size) continue;
            if (new_id == nb_id || to_delete.count(nb_id)) continue;

            bool do_merge = false;
            // if (is_joint(new_id) && is_joint(nb_id)) {
            if (GD.global_adj[new_id].size() > 2 && GD.global_adj[nb_id].size() > 2) {
                do_merge = true;
            }

            const auto &Vi = GD.global_vers[new_id];
            const auto &Vj = GD.global_vers[nb_id];
            float dist = (Vi.position.getVector3fMap() - Vj.position.getVector3fMap()).norm();

            if (!do_merge && dist < 0.5f * cfg_.fuse_dist_th) {
                do_merge = true;
            }

            if (!do_merge) continue;

            merge_into(nb_id, new_id); // Keeps existing and deletes new_id (after merge)

            to_delete.insert(new_id);
            break;
        }
    }

    if (to_delete.empty()) return 1; // end with success - no need to merge

    for (auto it = to_delete.rbegin(); it != to_delete.crend(); ++it) {
        const int del = *it;
        if (del < 0 || del >= static_cast<int>(GD.global_vers.size())) continue;
        
        GD.global_vers.erase(GD.global_vers.begin() + del);
        GD.global_adj.erase(GD.global_adj.begin() + del);

        for (auto &nbrs : GD.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), del), nbrs.end());
            for (auto &v : nbrs) {
                if (v > del) --v;
            }
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }

        GD.new_vers_indxs.erase(std::remove(GD.new_vers_indxs.begin(), GD.new_vers_indxs.end(), del), GD.new_vers_indxs.end());
        for (auto &id : GD.new_vers_indxs) {
            if (id > del) --id;
        }
    }

    GD.gskel_size = GD.global_vers.size();
    return size_assert();
}

bool GSkel::prune() {
    int N_new = GD.new_vers_indxs.size();
    if (GD.gskel_size == 0 || N_new == 0) return 0;
    if (static_cast<float>(N_new) / static_cast<float>(GD.gskel_size) > 0.5) return 1; // dont prune in beginning...
    std::vector<int> to_delete;

    for (int v : GD.new_vers_indxs) {
        if (v < 0 || v >= static_cast<int>(GD.global_adj.size())) continue;
        if (GD.global_adj[v].size() != 1) continue; // not a leaf
        int nb = GD.global_adj[v][0];
        if (GD.global_adj[nb].size() >= 3) {
            to_delete.push_back(v);
        }
    }

    if (to_delete.empty()) return 1; // exit w. success

    std::sort(to_delete.rbegin(), to_delete.rend());
    for (int idx : to_delete) {
        GD.global_vers.erase(GD.global_vers.begin() + idx);
        GD.global_adj.erase(GD.global_adj.begin() + idx);
        for (auto &nbrs : GD.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), idx), nbrs.end());
            for (auto &v : nbrs) {
                if (v > idx) --v;
            }
        }

        GD.new_vers_indxs.erase(std::remove(GD.new_vers_indxs.begin(), GD.new_vers_indxs.end(), idx), GD.new_vers_indxs.end());
        for (auto &id : GD.new_vers_indxs) {
            if (id > idx) -- id;
        }
    }
    GD.gskel_size = GD.global_vers.size();
    return 1;
}

bool GSkel::smooth_vertex_positions() {
    if (GD.gskel_size == 0 || GD.global_adj.size() == 0) return 0;

    std::vector<pcl::PointXYZ> new_pos(GD.gskel_size);

    for (size_t i=0; i<GD.gskel_size; ++i) {
        auto &v = GD.global_vers[i];
        auto &nbrs = GD.global_adj[i];

        new_pos[i] = v.position;
        if (v.type == 1 || v.type == 3) continue;
        if (v.smooth_iters <= 0) continue;

        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        int cnt = 0;
        for (int j : nbrs) {
            if (j < 0 || j >= (int)GD.gskel_size) continue;
            const auto& p = GD.global_vers[j].position;
            sum += p.getVector3fMap();
            cnt++;
        }
        if (cnt == 0) continue; // nothing to average - no neighbors

        Eigen::Vector3f avg = sum / static_cast<float>(nbrs.size());
        new_pos[i].getVector3fMap() = (1.0f - cfg_.vertex_smooth_coef)*v.position.getVector3fMap() + cfg_.vertex_smooth_coef*avg;
        v.pos_update = true;
        --v.smooth_iters;
    }

    for (size_t i=0; i<GD.gskel_size; ++i) {
        GD.global_vers[i].position = new_pos[i];
    }

    return 1;
}

bool GSkel::vid_manager() {
    // If vertex in global_vers does not have a unique vid (new vertex) -> Assign one
    for (int idx : GD.new_vers_indxs) {
        if (idx < 0 || idx >= (int)GD.global_vers.size()) continue;
        auto &v = GD.global_vers[idx];
        if (v.vid < 0) {
            v.vid = GD.next_vid++;
        }
    }

    rebuild_vid_index_map();

    for (size_t idx=0; idx<GD.gskel_size; ++idx) {
        auto& v = GD.global_vers[idx];
        const auto& nbrs_idx = GD.global_adj[idx];
        v.nb_ids.clear();
        v.nb_ids.reserve(nbrs_idx.size());
        for (int nb_idx : nbrs_idx) {
            if (nb_idx < 0 || nb_idx >= static_cast<int>(GD.global_vers.size())) continue;
            const int nb_vid = GD.global_vers[nb_idx].vid;
            if (nb_vid >= 0) {
                v.nb_ids.push_back(nb_vid);
            }
        }
        
        // dedupe just in case
        std::sort(v.nb_ids.begin(), v.nb_ids.end());
        v.nb_ids.erase(std::unique(v.nb_ids.begin(), v.nb_ids.end()), v.nb_ids.end());
    }

    graph_decomp();
    build_cloud_from_vertices(); // To publish correct cloud 
    return 1;
}

bool GSkel::resample_skeleton() {
    if (GD.global_vers.empty() || GD.global_adj.empty()) return true;
    
    graph_decomp();

    const float fuse        = cfg_.fuse_dist_th;
    const float eps_rdp     = 0.75f * fuse;       // RDP tolerance
    const float dmin        = 5.0f * fuse;       // min spacing on curves
    const float dmax        = 8.0f * fuse;       // spacing on straights

    const float beta        = 3.0f;               // curvature sensitivity
    const float r_match_end = 2.00f * dmin;       // capture radius for endpoints (anchors)
    const float r_match_mid = 1.00f * dmin;       // capture radius for interior samples
    const float ema_alpha   = 0.30f;              // EMA smoothing for matched vertices

    if (!extract_branches()) return false; // fill GD.branches

    std::size_t nodes_total = GD.global_vers.size();
    std::size_t branches_cnt = GD.branches.size();
    std::size_t branch_nodes = 0;
    for (auto& b : GD.branches) branch_nodes += b.size();

    std::cout << "[resample] dense nodes=" << nodes_total
            << " branches=" << branches_cnt
            << " nodes_in_branches=" << branch_nodes << std::endl;

    if (GD.branches.empty()) {
        // no branches -> clear sparse view
        GD.sparse_vers.clear();
        GD.sparse_adj.clear();
        GD.sparse_cloud->clear();
        GD.sparse_vid2idx.clear();
        return true;
    }

    if (!GD.sparse_vers.empty()) {
        GD.sparse_cloud->points.clear();
        GD.sparse_cloud->points.reserve(GD.sparse_vers.size());
        for (const auto& v : GD.sparse_vers) {
            GD.sparse_cloud->points.push_back(v.position);
        }
        kd_tree_->setInputCloud(GD.sparse_cloud);
    }

    const bool had_prev_sparse = !GD.sparse_vers.empty();

    // bookkeeping: dont reuse the same old sparse vertex twice in one tick
    std::vector<char> prev_taken(GD.sparse_vers.size(), 0);

    auto reuse_or_spawn = [&](const Eigen::Vector3f& target, bool is_endpoint) -> SparseVertex {
        int best_idx = -1;
        float best_d2 = std::numeric_limits<float>::infinity();
        if (had_prev_sparse) {
            pcl::PointXYZ q;
            q.getVector3fMap() = target;
            std::vector<int> ids;
            std::vector<float> d2;
            const float r = is_endpoint ? r_match_end : r_match_mid;
            int found = kd_tree_->radiusSearch(q, r, ids, d2);
            for (int k=0; k<found; ++k) {
                const int sid = ids[k];
                if (prev_taken[sid]) continue;
                if (d2[k] < best_d2) {
                    best_d2 = d2[k];
                    best_idx = sid;
                }
            }
        }

        if (best_idx >= 0) {
            // reuse old sparse vertex and smooth position
            SparseVertex v = GD.sparse_vers[best_idx];
            Eigen::Vector3f cur = v.position.getVector3fMap();
            v.position.getVector3fMap() = ema_alpha * target + (1.0f - ema_alpha) * cur;
            v.pos_update = true; // mark sparse vertex position updated
            prev_taken[best_idx] = 1;
            return v; // retain the original SVID
        }

        // no match -> spawn a new sparse vertex
        SparseVertex v;
        v.svid = GD.next_svid++; // update svid 
        v.position.getVector3fMap() = target; // update position
        v.pos_update = true;
        v.type = 0;
        return v;
    };

    // Generate resampled polylines per dense branch
    std::vector<SparseVertex> new_sparse;
    std::vector<std::vector<int>> new_sadj; // adjacency in index space
    new_sparse.reserve(GD.sparse_vers.size() + GD.branches.size() * 8);

    auto ensure_slot = [&](int idx) {
        if (static_cast<int>(new_sadj.size()) <= idx) {
            new_sadj.resize(idx+1);
        }
    };

    for (const auto& br_idx : GD.branches) {
        if (br_idx.size() < 2) continue; 

        // build dense polyline
        std::vector<Eigen::Vector3f> poly;
        poly.reserve(br_idx.size());
        for (int gi : br_idx) {
            poly.push_back(GD.global_vers[gi].position.getVector3fMap());
        }

        // simplify + adaptive resampling
        // auto simp = rdp3D(poly, eps_rdp);
        auto simp = poly;
        auto samp = adaptive_resampling(simp, dmin, dmax, beta);
        if (samp.size() < 2) continue;

        // emit endpoints (anchors) first then interiors 
        SparseVertex v_head = reuse_or_spawn(samp.front(), /*is_endpoint=*/true);
        new_sparse.push_back(std::move(v_head));
        int prev_idx = static_cast<int>(new_sparse.size() - 1);
        ensure_slot(prev_idx);

        for (size_t t=1; t+1<samp.size(); ++t) {
            SparseVertex v_mid = reuse_or_spawn(samp[t], /*is_endpoint*/false);
            new_sparse.push_back(std::move(v_mid));
            int cur_idx = static_cast<int>(new_sparse.size() - 1);
            ensure_slot(cur_idx);
            // connect edge
            new_sadj[prev_idx].push_back(cur_idx);
            new_sadj[cur_idx].push_back(prev_idx);
            prev_idx = cur_idx;
        }

        // tail endpoint
        SparseVertex v_tail = reuse_or_spawn(samp.back(), /*is_endpoint*/true);
        new_sparse.push_back(std::move(v_tail));
        int tail_idx = static_cast<int>(new_sparse.size() - 1);
        ensure_slot(tail_idx);
        // connect last edge
        new_sadj[prev_idx].push_back(tail_idx);
        new_sadj[tail_idx].push_back(prev_idx);
    }

    // dedupe neighbors
    for (auto& nbs : new_sadj) {
        std::sort(nbs.begin(), nbs.end());
        nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
    }


    // set type and fill nb_ids with svids
    for (int i=0; i<static_cast<int>(new_sparse.size()); ++i) {
        int deg = static_cast<int>(new_sadj[i].size());
        int new_type = 0;
        if (deg == 1) new_type = 1;
        else if (deg == 2) new_type = 2;
        else if (deg > 2) new_type = 3;

        if (new_sparse[i].type != new_type) {
            new_sparse[i].type_update = true;
        }

        new_sparse[i].type = new_type;

        // nb svids
        new_sparse[i].nb_ids.clear();
        if (i < static_cast<int>(new_sadj.size())) {
            new_sparse[i].nb_ids.reserve(new_sadj[i].size());
            for (int j : new_sadj[i]) {
                if ( j >= 0 && j < static_cast<int>(new_sparse.size())) {
                    new_sparse[i].nb_ids.push_back(new_sparse[j].svid);                
                }
            }
            std::sort(new_sparse[i].nb_ids.begin(), new_sparse[i].nb_ids.end());
            new_sparse[i].nb_ids.erase(std::unique(new_sparse[i].nb_ids.begin(), new_sparse[i].nb_ids.end()), new_sparse[i].nb_ids.end()); // dedupe
        }
    }

    GD.sparse_vers.swap(new_sparse);
    GD.sparse_adj.swap(new_sadj);

    // rebuild sparse cloud
    GD.sparse_cloud->points.clear();
    GD.sparse_cloud->points.reserve(GD.sparse_vers.size());
    for (const auto& v : GD.sparse_vers) {
        GD.sparse_cloud->points.push_back(v.position);
    }
    GD.sparse_cloud->width = (uint32_t)GD.sparse_cloud->points.size();
    GD.sparse_cloud->height = 1;
    GD.sparse_cloud->is_dense = true;

    // vid map
    GD.sparse_vid2idx.clear();
    GD.sparse_vid2idx.reserve(GD.sparse_vers.size());
    for (int i=0; i<static_cast<int>(GD.sparse_vers.size()); ++i) {
        int svid = GD.sparse_vers[i].svid; // sparse vertex id
        if (svid >= 0) {
            GD.sparse_vid2idx[svid] = i;
        } 
    }

    std::cout << "[resample] sparse_vers=" << GD.sparse_vers.size()
          << " sparse_edges=" << std::accumulate(GD.sparse_adj.begin(), GD.sparse_adj.end(), 0,
               [](int s, const auto& v){ return s + (int)v.size(); }) / 2
          << std::endl;

    return 1;
}


/* Helpers */
void GSkel::build_cloud_from_vertices() {
    if (GD.global_vers.empty()) return;
    GD.global_vers_cloud->clear();
    for (const auto& v : GD.global_vers) {
        if (!pcl::isFinite(v.position)) {
            std::cout << "INVALID POSITION!!" << std::endl;
        }
        GD.global_vers_cloud->points.push_back(v.position);
    }

    GD.global_vers_cloud->width  = static_cast<uint32_t>(GD.global_vers_cloud->points.size());
    GD.global_vers_cloud->height = 1;
    GD.global_vers_cloud->is_dense = true;

    if (GD.global_vers.size() != GD.global_vers_cloud->points.size()) {
        std::cout << "NOT SAME SIZE???" << std::endl;
    }
}

void GSkel::rebuild_vid_index_map() {
    GD.vid2idx.clear();
    GD.vid2idx.reserve(GD.global_vers.size());
    for (int i=0; i<static_cast<int>(GD.global_vers.size()); ++i) {
        const int vid = GD.global_vers[i].vid;
        if (vid >= 0) {
            GD.vid2idx[vid] = i;
        }
    }
}

void GSkel::graph_decomp() {
    GD.joints.clear();
    GD.leafs.clear();

    const int N = GD.global_vers.size();
    for (int i=0; i<N; ++i) {
        auto &vg = GD.global_vers[i];
        int degree = GD.global_adj[i].size();
        int v_type = 0;

        switch (degree)
        {
        case 1:
            GD.leafs.push_back(i);
            v_type = 1;
            break;
        case 2:
            v_type = 2;
            break;
        default:
            if (degree > 2) {
                GD.joints.push_back(i);
                v_type = 3;
            }
            break;
        }

        // update type
        if (vg.type != v_type) vg.type_update = true;
        vg.type = v_type;
    }
}

void GSkel::merge_into(int keep, int del) {
    auto& Vi = GD.global_vers[keep];
    auto& Vj = GD.global_vers[del];

    int tot = Vi.obs_count + Vj.obs_count;
    if (tot == 0) tot = 1;
    Vi.position.getVector3fMap() = (Vi.position.getVector3fMap() * Vi.obs_count + Vj.position.getVector3fMap() * Vj.obs_count) / static_cast<float>(tot);
    Vi.obs_count = tot;
    Vi.pos_update = true; // position updated

    // Remap neighbors 
    for (int nb : GD.global_adj[del]) {
        if (nb == keep) continue;
        auto& keep_nbs = GD.global_adj[keep];
        if (std::find(keep_nbs.begin(), keep_nbs.end(), nb) == keep_nbs.end()) {
            keep_nbs.push_back(nb);
        }

        // Remap neighbor's neighbors
        auto &nbs_nb = GD.global_adj[nb];
        std::replace(nbs_nb.begin(), nbs_nb.end(), del, keep);
    }
}

bool GSkel::size_assert() {
    const int A = static_cast<int>(GD.global_vers.size());
    const int B = static_cast<int>(GD.global_adj.size());
    const int C = static_cast<int>(GD.gskel_size);
    const bool ok = (A == B) && (B == C);
    return ok;
}

std::vector<Eigen::Vector3f> GSkel::adaptive_resampling(std::vector<Eigen::Vector3f>& poly, float dmin, float dmax, float beta) {
    const int n = static_cast<int>(poly.size());
    if (n == 0) return {};
    if (n == 1) return poly;

    auto curvature3 = [](const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
        Eigen::Vector3f ab = b - a;
        Eigen::Vector3f bc = c - b;
        Eigen::Vector3f ac = c - a;
        float la = ab.norm();
        float lb = bc.norm();
        float lc = ac.norm();
        float denom = la * lb * lc;
        if (denom < 1e-6f) return 0.f;
        float area2 = ab.cross(bc).norm(); // twice tiangle area
        return area2 / denom; // ~1/R
    };

    auto target_d = [&](int i) -> float {
        if (i <= 0 || i>=n-1) return dmax; // endpoints
        float k = curvature3(poly[i-1], poly[i], poly[i+1]);
        float d = dmax / (1.0f + beta * k);
        return std::clamp(d, dmin, dmax);
    };

    std::vector<Eigen::Vector3f> out;
    out.reserve(n);
    out.push_back(poly.front());
    float acc = 0.0f;

    for (int i=1; i<n; ++i) {
        Eigen::Vector3f seg = poly[i] - poly[i-1];
        float segL = seg.norm();
        if (segL < 1e-6f) continue;

        float local = target_d(i-1);
        acc += segL;

        while (acc >= local) {
            float over = acc - local;
            float t = (segL - over) / segL;
            out.push_back(poly[i-1] + t * seg);
            acc = over;
            local = target_d(i-1);
        }
    }

    if ((out.back() - poly.back()).norm() > 1e-3f) {
        out.push_back(poly.back());
    }

    return out;
}

bool GSkel::extract_branches() {    
    const int N = static_cast<int>(GD.global_adj.size());
    GD.branches.clear();
    if (N == 0) return true;

    std::vector<int> deg(N, 0);
    for (int i=0; i<N; ++i) {
        // deg[i] = GD.global_vers[i].type;
        deg[i] = static_cast<int>(GD.global_adj[i].size());
    }


    auto ek = [](int a, int b)->uint64_t {
        if (a>b) {
            std::swap(a,b);
        } 
        return (uint64_t(a)<<32) | uint32_t(b); 
    };
    
    std::unordered_set<uint64_t> used_edge;
    used_edge.reserve(N * 4);

    auto walk = [&](int s, int nb) -> std::vector<int> {
        std::vector<int> path;
        path.reserve(32);

        // seed edge
        used_edge.insert(ek(s, nb));
        int prev = s;
        int cur  = nb;

        path.push_back(s);
        path.push_back(cur);

        // follow degree-2 chain until endpoint (deg!=2) or we hit an already-used edge
        while (deg[cur] == 2) {
            int nxt = -1;
            for (int w : GD.global_adj[cur]) {
                if (w != prev) { nxt = w; break; }
            }
            if (nxt < 0) break;
            auto key = ek(cur, nxt);
            if (used_edge.count(key)) break;
            used_edge.insert(key);

            prev = cur;
            cur  = nxt;
            path.push_back(cur);
        }
        return path;
    };

    for (int i = 0; i < N; ++i) {
        if (deg[i] == 0) continue;
        if (deg[i] != 2) {
            for (int nb : GD.global_adj[i]) {
                auto key = ek(i, nb);
                if (used_edge.count(key)) continue;
                auto path = walk(i, nb);
                if ((int)path.size() >= 2) {
                    GD.branches.push_back(std::move(path));
                }
            }
        }
    }

    // Keep very short branches (>=2) so resampling has endpoints to latch onto
    GD.branches.erase(
        std::remove_if(GD.branches.begin(), GD.branches.end(),
                       [](const std::vector<int>& b){ return (int)b.size() < 2; }),
        GD.branches.end()
    );
    return true;


    // std::vector<char> used(N, 0);
    // auto walk = [&](int s, int prev) -> std::vector<int> {
    //     std::vector<int> path;
    //     path.reserve(64);
    //     int cur = s;
    //     int par = prev;
    //     while (true) {
    //         path.push_back(cur);
    //         used[cur] =  1;
    //         if (cur != s && deg[cur] != 2) break; // reached leaf / joint - Will include the last leaf/joint in branch
    //         int nxt = -1;
    //         for (int nb : GD.global_adj[cur]) {
    //             if (nb != par) {
    //                 nxt = nb;
    //                 break;
    //             }
    //         }
    //         if (nxt < 0 || used[nxt]) break; // dead end or already used
    //         par = cur;
    //         cur = nxt;
    //     }
    //     return path;
    // };

    // // start walks at leafs and junctions 
    // for (int i=0; i<N; ++i) {
    //     if (deg[i] != 2) {
    //         for (int nb : GD.global_adj[i]) {
    //             if (!used[i]) {
    //                 GD.branches.push_back(walk(i, nb)); // each branch includes the junction / leaf itself
    //             }
    //         }
    //     }
    // }

    // GD.branches.erase(
    //     std::remove_if(GD.branches.begin(), GD.branches.end(),
    //         [&](const std::vector<int>& b){ return (int)b.size() < std::max(2, cfg_.min_branch_length); }),
    //     GD.branches.end()
    // );
    
    // return true;
}

void GSkel::rdp_impl(const std::vector<Eigen::Vector3f>& P, int a, int b, float eps, std::vector<int>& keep) {
    if (b <= a+1) return;
    
    auto pointSegDist = [](const Eigen::Vector3f& x, const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> float {
        /* ab spans the segment - x lies between */
        Eigen::Vector3f ab = b - a;
        Eigen::Vector3f ax = x - a;
        float lab2 = ab.squaredNorm();
        if (lab2 < 1e-6f) return (x - a).norm();
        float s = ax.dot(ab) / lab2;
        s = std::max(0.f, std::min(1.0f, s));
        Eigen::Vector3f p = a + s * ab;
        return (x - p).norm();
    };
    
    float maxd = -1.f; 
    int idx = -1;
    
    for (int i=a+1; i<b; ++i) {
        float d = pointSegDist(P[i], P[a], P[b]);
        if (d > maxd) { maxd = d; idx = i; }
    }

    if (maxd > eps) {
        rdp_impl(P, a, idx, eps, keep);
        keep.push_back(idx);
        rdp_impl(P, idx, b, eps, keep);
    }
}

std::vector<Eigen::Vector3f> GSkel::rdp3D(const std::vector<Eigen::Vector3f>& poly, float eps) {
    const int n = (int)poly.size();
    if (n <= 2) return poly;
    std::vector<int> keep; keep.reserve(n);
    keep.push_back(0);

    rdp_impl(poly, 0, n-1, eps, keep);
    keep.push_back(n-1);
    std::sort(keep.begin(), keep.end());
    std::vector<Eigen::Vector3f> out; out.reserve(keep.size());
    for (int i : keep) out.push_back(poly[i]);
    return out;
}



/* If sparse skel larger than threshold -> prune small branches! */

/* Joint merging into common junction point! (radius based) */