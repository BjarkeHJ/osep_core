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
    running = 1;
}

bool GSkel::gskel_run() {
    // auto ts = std::chrono::high_resolution_clock::now();

    RUN_STEP(increment_skeleton);
    RUN_STEP(graph_adj);
    RUN_STEP(mst);
    RUN_STEP(vertex_merge);
    RUN_STEP(prune);
    RUN_STEP(smooth_vertex_positions);
    RUN_STEP(vid_manager);

    // Keep small fusing distance -> Downsample to get sparser skeleton???

    running = 1;

    // auto te = std::chrono::high_resolution_clock::now();
    // auto telaps = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count();
    // std::cout << "[GSKEL] Time Elapsed: " << telaps << " ms" << std::endl;
    return running;
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

bool GSkel::downsample() {
    if (GD.global_vers.empty()) return 0;

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

