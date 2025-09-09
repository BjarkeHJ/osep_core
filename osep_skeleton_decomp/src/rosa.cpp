/*

Main algorithm for local ROSA Point computation

----- Notes -----
Consider normalization of point cloud data prior to processing (for ease of parameter tuning)
OBS: de-normalize before returning...

Consider downsampling prior to normal estimation (keeping many points ~1000)

TODO: REMOVE ANY VERTICES THAT ARE FAR AWAY FROM POINT CLOUD

TODO: For normal estimation and neigborhood use radius instead of KNN

*/

#include <rosa.hpp>

Rosa::Rosa(const RosaConfig& cfg) : cfg_(cfg) {
    RD.orig_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    RD.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    RD.nrms_.reset(new pcl::PointCloud<pcl::Normal>);
    RD.pts_->points.reserve(cfg.max_points);
    RD.nrms_->points.reserve(cfg.max_points);

    RD.surf_nbs.reserve(cfg.max_points);

    tmp_pt_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    tmp_pt_->points.reserve(10000);
    tmp_nrm_.reset(new pcl::PointCloud<pcl::Normal>);
    tmp_nrm_->points.reserve(10000);
    tmp_pn_.reset(new pcl::PointCloud<pcl::PointNormal>);
    tmp_pn_->points.reserve(10000);
    tmp_pn_ds_.reset(new pcl::PointCloud<pcl::PointNormal>);
    tmp_pn_ds_->points.reserve(cfg_.max_points);

    running = 1;
}

bool Rosa::rosa_run() {
    if (RD.orig_->points.empty()) return 0; // No points received...
    // auto ts = std::chrono::high_resolution_clock::now();
    
    RUN_STEP(preprocess);
    RUN_STEP(rosa_init);
    RUN_STEP(similarity_neighbor_extraction);
    RUN_STEP(drosa);
    RUN_STEP(dcrosa);
    RUN_STEP(vertex_sampling);
    RUN_STEP(vertex_smooth);

    // auto te = std::chrono::high_resolution_clock::now();
    // auto telaps = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count();
    // std::cout << "[ROSA] Time Elapsed: " << telaps << " ms" << std::endl;
    return running;
}

bool Rosa::preprocess() {
    RD.pts_->clear();
    RD.nrms_->clear();
    RD.surf_nbs.clear();
    RD.simi_nbs.clear();

    /* Distance Filtering */
    const float r2 = cfg_.pts_dist_lim * cfg_.pts_dist_lim;
    tmp_pt_->clear();
    tmp_pt_->points.resize(RD.orig_->points.size());
    size_t n = 0;

    const auto& src = RD.orig_->points;
    auto& dst = tmp_pt_->points;

    for (size_t i = 0; i < src.size(); ++i) {
        const auto& p = src[i];
        const float d2 = p.x * p.x + p.y * p.y + p.z * p.z;
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && d2 <= r2 && d2 > 0.1f*0.1f) {
            dst[n++] = p;
        }
    }
    dst.resize(n);
    RD.orig_->swap(*tmp_pt_);
    RD.orig_->width  = static_cast<uint32_t>(RD.orig_->points.size());
    RD.orig_->height = 1;
    RD.orig_->is_dense = true;
    RD.pcd_size_ = RD.orig_->points.size();

    /* CONSIDER DOWNSAMPLING HERE (Perhaps max points can be pushed up then?)*/

    /* Normal Estimation */
    if (static_cast<int>(RD.pcd_size_) < cfg_.min_points) {
        tmp_pt_->clear();
        tmp_nrm_->clear();
        return false;
    }

    ne_.setInputCloud(RD.orig_);
    ne_.setViewPoint(0.0, 0.0, 0.0);
    ne_.setSearchMethod(kd_tree_);
    ne_.setKSearch(cfg_.ne_knn);
    ne_.compute(*tmp_nrm_);

    /* Downsampling */
    tmp_pn_->clear();
    tmp_pn_->points.resize(RD.orig_->points.size());
    for (size_t i = 0; i < tmp_pn_->points.size(); ++i) {
        auto& q = tmp_pn_->points[i];
        const auto& p = RD.orig_->points[i];
        const auto& n = tmp_nrm_->points[i];
        q.x = p.x;  q.y = p.y;  q.z = p.z;
        q.normal_x = n.normal_x;
        q.normal_y = n.normal_y;
        q.normal_z = n.normal_z;
    }
    tmp_pn_->width  = static_cast<uint32_t>(tmp_pn_->points.size());
    tmp_pn_->height = 1;
    
    float leaf = estimate_leaf_from_bbox(*RD.orig_, cfg_.max_points);

    const int   max_iters = 8;
    const float tol       = 0.05f;
    const float leaf_min  = 1e-4f;
    const float leaf_max  = 1e4f;
    size_t N = 0;
    for (int iter = 0; iter < max_iters; ++iter) {
        vgf_.setInputCloud(tmp_pn_);
        vgf_.setLeafSize(leaf, leaf, leaf);
        vgf_.filter(*tmp_pn_ds_);
        
        N = tmp_pn_ds_->points.size();
        if (N == 0) {
            leaf *= 0.5f;
            if (leaf < leaf_min) { leaf = leaf_min; break; }
            continue;
        }

        const float ratio_out = static_cast<float>(N) / static_cast<float>(cfg_.max_points);
        if (std::fabs(ratio_out - 1.0f) <= tol) {
            break;
        }

        leaf *= std::cbrt(ratio_out);
        if (!std::isfinite(leaf)) leaf = 0.05f;
        if (leaf < leaf_min)  leaf = leaf_min;
        if (leaf > leaf_max)  leaf = leaf_max;
    }

    leaf_size = leaf;

    const size_t M = tmp_pn_ds_->points.size();
    RD.pts_->clear();
    RD.pts_->resize(M);
    RD.nrms_->clear();
    RD.nrms_->resize(M);

    for (size_t i = 0; i < tmp_pn_ds_->points.size(); ++i) {
        const auto& q = tmp_pn_ds_->points[i];
        RD.pts_->points[i] = pcl::PointXYZ(q.x, q.y, q.z);
        Eigen::Vector3f nn(q.normal_x, q.normal_y, q.normal_z);
        float L = nn.norm();
        if (L > 1e-6f) nn /= L; else nn = Eigen::Vector3f::Zero();
        RD.nrms_->points[i].normal_x = nn.x();
        RD.nrms_->points[i].normal_y = nn.y();
        RD.nrms_->points[i].normal_z = nn.z();
    }

    // TODO: Density filtering here!
    
    RD.pts_->width  = static_cast<uint32_t>(tmp_pn_ds_->points.size());
    RD.pts_->height = 1;
    RD.pts_->is_dense = true;
    RD.nrms_->width  = static_cast<uint32_t>(tmp_pn_ds_->points.size());
    RD.nrms_->height = 1;
    RD.nrms_->is_dense = true;
    
    tmp_pt_->clear();
    tmp_nrm_->clear();
    tmp_pn_->clear();
    tmp_pn_ds_->clear();

    RD.pcd_size_ = RD.pts_->points.size();
    return true;
}

bool Rosa::rosa_init() {
    Eigen::Vector3f nv;
    Eigen::Matrix3f M;
    pset.resize(RD.pcd_size_, 3);
    vset.resize(RD.pcd_size_, 3);
    vvar.resize(RD.pcd_size_, 1);
    for (size_t i=0; i<RD.pcd_size_; ++i) {
        const auto m = RD.pts_->points[i].getVector3fMap();
        pset.row(i) = m.transpose();
        nv(0) = RD.nrms_->points[i].normal_x;
        nv(1) = RD.nrms_->points[i].normal_y;
        nv(2) = RD.nrms_->points[i].normal_z;
        M = create_orthonormal_frame(nv);
        vset.row(i) = M.col(0);
    }
    return 1;
}

bool Rosa::similarity_neighbor_extraction() {
    RD.simi_nbs.clear();
    RD.simi_nbs.resize(RD.pcd_size_);

    if (RD.pts_->empty()) return 0;

    kd_tree_->setInputCloud(RD.pts_);
    std::vector<int> nb_indxs;
    std::vector<float> nb_dists;
    nb_indxs.reserve(32);
    nb_dists.reserve(32);
    const float radius_r = 10.0f * leaf_size;
    const float th_sim = 0.1f * leaf_size;

    for (size_t i=0; i<RD.pcd_size_; ++i) {
        nb_indxs.clear();
        nb_dists.clear();

        const pcl::PointXYZ& P1 = RD.pts_->points[i];
        const pcl::Normal& N1 = RD.nrms_->points[i];

        if (!std::isfinite(N1.normal_x) || !std::isfinite(N1.normal_y) || !std::isfinite(N1.normal_z)) {
            RD.simi_nbs[i].clear();
            continue;
        }

        const Eigen::Map<const Eigen::Vector3f> p1(&P1.x);
        Eigen::Vector3f v1(N1.normal_x, N1.normal_y, N1.normal_z);
        v1.normalize();

        kd_tree_->radiusSearch(P1, radius_r, nb_indxs, nb_dists);

        std::vector<int> temp_neighs;
        temp_neighs.reserve(nb_indxs.size());

        for (int idx : nb_indxs) {
            if (idx == (int)i) continue;
            const pcl::PointXYZ& P2 = RD.pts_->points[idx];
            const pcl::Normal& N2 = RD.nrms_->points[idx];
            
            if (!std::isfinite(N2.normal_x) || !std::isfinite(N2.normal_y) || !std::isfinite(N2.normal_z)) {
                continue;
            }

            const Eigen::Map<const Eigen::Vector3f> p2(&P2.x);
            Eigen::Vector3f v2(N2.normal_x, N2.normal_y, N2.normal_z);
            v2.normalize();
            const float w = similarity_metric(p1, v1, p2, v2, radius_r); // can take 6th arg scale (default 5.0)
            if (w > th_sim) {
                temp_neighs.push_back(idx);
            }
        }
        RD.simi_nbs[i].swap(temp_neighs);
    }
    return 1;
}

bool Rosa::drosa() {
    RD.surf_nbs.clear();
    RD.surf_nbs.resize(RD.pcd_size_);
    std::vector<int> knn_idxs(cfg_.nb_knn);
    std::vector<float> knn_dists(cfg_.nb_knn);
    kd_tree_->setInputCloud(RD.pts_);
    for (size_t i=0; i<RD.pcd_size_; ++i) {
        kd_tree_->nearestKSearch(i, cfg_.nb_knn, knn_idxs, knn_dists);
        RD.surf_nbs[i] = knn_idxs;
    }   

    /* ROSA POINT ORIENTATION*/
    Eigen::Vector3f p, v, new_v;
    std::vector<int> active;
    
    for (int it=0; it<cfg_.niter_drosa; ++it) {
        Eigen::MatrixXf vnew = Eigen::MatrixXf::Zero(RD.pcd_size_, 3);

        for (size_t pidx=0; pidx<RD.pcd_size_; ++pidx) {
            p = pset.row(pidx);
            v = vset.row(pidx);
            active = compute_active_samples(pidx, p, v); // Extract point cloud slice according to the plane equation from v
            
            if (!active.empty()) {
                new_v = compute_symmetrynormal(active);
                vnew.row(pidx) = new_v.transpose();
                vvar(pidx, 0) = symmnormal_variance(new_v, active);
            }
            else {
                vnew.row(pidx) = vset.row(pidx);
                // vvar(pidx, 0) = 0.0f;
                vvar(pidx, 0) = 1e+3f;
            }
        }
        vset = vnew; // Overwrite 
        
        constexpr float eps = 1e-6f;
        vvar = (vvar.array().square().square() + eps).inverse(); // ~1/(vvar⁴ + eps)

        // Smoothing...
        for (size_t p=0; p<RD.pcd_size_; ++p) {
            const auto& snb = RD.surf_nbs[p];
            if (snb.empty()) continue;
            
            // Compute weighted covariance among the surface neighbors and solve for the principal axis
            // M = sum_i (w_i * v_i * w_i^T) = V^T diag(w) V
            Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
            for (int j : snb) {
                if (j == (int)p) continue;
                const Eigen::Vector3f v = vset.row(j).transpose();
                const float w = vvar(j,0);
                M.noalias() += w * (v * v.transpose());
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(M);
            Eigen::Vector3f principal = es.eigenvectors().col(2);
            principal.normalize();
            vset.row(p) = principal.transpose();
        }
    }
    
    /* ROSA POINT POSITION */
    // float PLANAR_TH = 0.1;
    // Enable planar fallback for flat surfaces
    float PLANAR_TH = 0.2f; // tune as needed (0.05-0.15 typical)
    std::vector<int> good_ids;
    std::vector<int> poor_ids;
    std::vector<Eigen::Vector3f> goodCenters;
    good_ids.reserve(RD.pcd_size_);
    poor_ids.reserve(RD.pcd_size_);
    goodCenters.reserve(RD.pcd_size_);

    for (size_t pidx=0; pidx<RD.pcd_size_; ++pidx) {
        p = pset.row(pidx);
        v = vset.row(pidx);
        active = compute_active_samples(pidx, p, v);

        if (active.empty()) {
            poor_ids.push_back((int)pidx);
            continue;
        }

        Eigen::Vector3f evals = cov_eigs_from_normals(active);
        const float eps = 1e-6f;
        float planar_score = (evals(1) - evals(0)) / std::max(evals(2), eps);

        Eigen::Vector3f center_f;
        if (planar_score < PLANAR_TH) {
            // std::cout << "PLANAR PCA BLEND!" << std::endl;
            Eigen::Vector3f mean_p = Eigen::Vector3f::Zero();
            Eigen::Vector3f mean_n = Eigen::Vector3f::Zero();
            for (int i : active) {
                mean_p += RD.pts_->points[i].getVector3fMap();
                const auto v = RD.nrms_->points[i].getNormalVector3fMap();
                const float s2 = v.squaredNorm();
                const float inv_len = (s2 > 1e-20f) ? (1.0f / std::sqrt(s2)) : 0.0f;
                mean_n += v * inv_len;
            }
            const float inv_m = 1.0f / static_cast<float>(active.size());
            mean_p *= inv_m;
            mean_n *= inv_m;
            Eigen::Vector3f fallback_center = mean_p - mean_n * cfg_.max_proj_range;
            Eigen::Vector3f proj_center = closest_projection_point(active);
            float blend = 0.5f; // 0.0 = only projection, 1.0 = only fallback
            center_f = blend * fallback_center + (1.0f - blend) * proj_center;
        } else {
            center_f = closest_projection_point(active);
        }

        const bool finite = center_f.allFinite();
        const bool in_range = (center_f - p).norm() < cfg_.max_proj_range;
        if (finite && in_range) {
            pset.row(pidx) = center_f;
            good_ids.push_back((int)pidx);
            goodCenters.push_back(center_f);
        }
        else {
            poor_ids.push_back(pidx);
        }
    }

    if (poor_ids.empty()) return 1; // exit succesfully...

    pcl::PointCloud<pcl::PointXYZ>::Ptr goodSrc(new pcl::PointCloud<pcl::PointXYZ>);
    goodSrc->reserve(good_ids.size());
    for (int gid : good_ids) {
        goodSrc->push_back(RD.pts_->points[gid]);
    }

    kd_tree_->setInputCloud(goodSrc);
    std::vector<int> pair_id(1);
    std::vector<float> pair_dist(1);
    for (int poor_id : poor_ids) {
        const pcl::PointXYZ& q = RD.pts_->points[poor_id];
        if (kd_tree_->nearestKSearch(q, 1, pair_id, pair_dist) > 0) {
            const int idx = pair_id[0];
            pset.row(poor_id) = goodCenters[idx].transpose();
        }
    }

    return 1;
}

bool Rosa::dcrosa() {
    constexpr float CONF_TH = 0.7f;
    constexpr int MIN_NB = 3;
    if (RD.pcd_size_ == 0) return false;
    Eigen::MatrixXf newpset(RD.pcd_size_, 3);

    constexpr float CONVERGENCE_TH = 1e-5f;
    constexpr float TRIM_RATIO = 0.4f; // 30% farthest neighbors trimmed
    constexpr float MAX_MOVE = 0.01f; // clamp max movement per iteration (units)
    constexpr float GAUSS_SIGMA = 0.4f; // Gaussian sigma for weighting (units)
    for (int it = 0; it < cfg_.niter_dcrosa; ++it) {
        bool any_change = false;
        float max_delta = 0.0f;
        // Pass 1: Robust neighborhood averaging (weighted trimmed mean, Gaussian weights, clamped move)
        for (size_t i = 0; i < RD.pcd_size_; ++i) {
            const auto& nb = RD.simi_nbs[i];
            if (nb.empty()) {
                newpset.row(i) = pset.row(i);
                continue;
            }
            // Collect neighbor points and distances
            std::vector<std::pair<float, Eigen::Vector3f>> dists;
            const Eigen::Vector3f pi = pset.row(i).transpose();
            for (int j : nb) {
                Eigen::Vector3f pj = pset.row(j).transpose();
                float d = (pj - pi).norm();
                dists.emplace_back(d, pj);
            }
            // Sort by distance
            std::sort(dists.begin(), dists.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            // Trim farthest neighbors
            size_t trim = static_cast<size_t>(TRIM_RATIO * dists.size());
            size_t keep = dists.size() - trim;
            if (keep < 1) keep = 1;
            // Weighted mean (Gaussian)
            double wsum = 0.0;
            Eigen::Vector3d acc = Eigen::Vector3d::Zero();
            for (size_t k = 0; k < keep; ++k) {
                double w = std::exp(-0.5 * (dists[k].first * dists[k].first) / (GAUSS_SIGMA * GAUSS_SIGMA));
                acc += w * dists[k].second.cast<double>();
                wsum += w;
            }
            Eigen::Vector3d avg = (wsum > 0.0) ? (acc / wsum) : Eigen::Vector3d(pi.x(), pi.y(), pi.z());
            Eigen::Vector3f avgf = avg.cast<float>();
            Eigen::Vector3f move = avgf - pi;
            float move_norm = move.norm();
            if (move_norm > MAX_MOVE) move = move * (MAX_MOVE / move_norm);
            Eigen::Vector3f newp = pi + move;
            float delta = move.norm();
            if (delta > CONVERGENCE_TH) any_change = true;
            if (delta > max_delta) max_delta = delta;
            newpset.row(i) = newp.transpose();
        }
        pset.swap(newpset);

        // Pass 2: Local line fit refinement
        for (size_t i = 0; i < RD.pcd_size_; ++i) {
            const auto& nb = RD.simi_nbs[i];
            newpset.row(i) = pset.row(i); // default
            if (static_cast<int>(nb.size()) < MIN_NB) continue;

            Eigen::Vector3f mean, dir;
            float conf;
            if (local_line_fit(nb, mean, dir, conf, MIN_NB) && conf > CONF_TH) {
                const Eigen::Vector3f x = pset.row(i).transpose();
                const float t = dir.dot(x - mean);
                Eigen::Vector3f refined = mean + t * dir;
                float delta = (refined - x).norm();
                if (delta > CONVERGENCE_TH) any_change = true;
                if (delta > max_delta) max_delta = delta;
                newpset.row(i) = refined.transpose();
            }
        }
        pset.swap(newpset);

        // Early exit if nothing changed or converged
        if (!any_change || max_delta < CONVERGENCE_TH) break;
    }
    return true;
}

bool Rosa::vertex_sampling() {
    if (!pset_cloud) {
        pset_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }

    if (RD.pcd_size_ == 0) {
        RD.skelver.resize(0,3);
        return 0;
    }

    pset_cloud->points.clear();
    pset_cloud->width = 0;
    pset_cloud->height = 1;
    pset_cloud->is_dense = true;

    for (size_t i=0; i<RD.pcd_size_; ++i) {
        const float x = pset(i,0);
        const float y = pset(i,1);
        const float z = pset(i,2);
        const bool finite = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
        if (!finite) continue;
        pset_cloud->points.emplace_back(x,y,z);
    }
    pset_cloud->width = pset_cloud->points.size();
    RD.pcd_size_ = pset_cloud->points.size();

    const float R = static_cast<float>(leaf_size);
    std::vector<char> covered(RD.pcd_size_, 0);
    std::vector<int> corresp(RD.pcd_size_, -1);
    std::vector<float> mind2(RD.pcd_size_, std::numeric_limits<float>::infinity());

    RD.skelver.resize(0,3);
    kd_tree_->setInputCloud(pset_cloud);
    
    std::vector<int> indxs;
    std::vector<float> d2s;
    int num_covered = RD.pcd_size_;

    for (size_t start=0; start<RD.pcd_size_ && num_covered>0; ) {
        int seed = -1;
        for (size_t i=0; i<RD.pcd_size_; ++i) {
            if (!covered[i]) {
                seed = i;
                start = i+1;
                break;
            }
        }
        if (seed == -1) break; // all covered

        const int vid = static_cast<int>(RD.skelver.rows());
        RD.skelver.conservativeResize(vid + 1, 3);
        RD.skelver.row(vid) = pset.row(seed);

        indxs.clear();
        d2s.clear();
        const pcl::PointXYZ& q = pset_cloud->points[seed];
        const int found = kd_tree_->radiusSearch(q, R, indxs, d2s);

        if (found <= 0) {
            covered[seed] = 1;
            num_covered -= 1;
            mind2[seed] = 0.0f;
            corresp[seed] = vid;
            continue;
        }

        for (int j=0; j<found; ++j) {
            const int idx = indxs[j];
            const float d2 = d2s[j];
            if (d2 < mind2[idx]) {
                mind2[idx] = d2;
                corresp[idx] = vid;
            }
            if (!covered[idx]) {
                covered[idx] = 1;
                num_covered -= 1;
            }
        }
    }
    
    // EXPORT CORRESP IF NEEDED DOWNSTREAM... (RD.corresp or something...)

    /* Recentering */

    // Inverse correspondence map
    std::vector<std::vector<int>> vertex_to_pts(RD.skelver.rows());
    for (size_t i=0; i<RD.pcd_size_; ++i) {
        int vid = corresp[i];
        if (vid >= 0) {
            vertex_to_pts[vid].push_back(i);
        }
    }

    for (int i=0; i<RD.skelver.rows(); ++i) {
        auto &idxs = vertex_to_pts[i];
        if (idxs.empty()) continue;

        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        for (int idx : idxs) {
            const auto& p = RD.pts_->points[idx];
            sum += p.getVector3fMap();
        }

        Eigen::Vector3f eucl_center = sum / static_cast<float>(idxs.size());
        Eigen::Vector3f current = RD.skelver.row(i);
        RD.skelver.row(i) = (cfg_.alpha_recenter * current + (1.0 - cfg_.alpha_recenter) * eucl_center).transpose();
    }
    return 1;
}

bool Rosa::vertex_smooth() {
    int V = static_cast<int>(RD.skelver.rows());
    if (V == 0) return 0;

    tmp_pt_->points.clear();
    tmp_pt_->width = 0;
    tmp_pt_->height = 1;
    tmp_pt_->is_dense = true;
    
    for (int i=0; i<V; ++i) {
        const float x = RD.skelver(i,0);
        const float y = RD.skelver(i,1);
        const float z = RD.skelver(i,2);
        const bool finite = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
        if (!finite) continue;
        tmp_pt_->points.emplace_back(x,y,z);
    }

    V = static_cast<int>(tmp_pt_->points.size());
    kd_tree_->setInputCloud(tmp_pt_);
    
    std::vector<std::vector<int>> neighbors(V);
    {
        std::vector<int> idx;
        std::vector<float> dist2;
        idx.reserve(64);
        dist2.reserve(64);

        for (int i=0; i<V; ++i) {
            idx.clear();
            dist2.clear();
            pcl::PointXYZ& qp = tmp_pt_->points[i];
            if (kd_tree_->radiusSearch(qp, cfg_.radius_smooth, idx, dist2) > 0) {
                if (idx.empty() || idx[0] != i) {
                    idx.insert(idx.begin(), i);
                }
                neighbors[i].assign(idx.begin(), idx.end());
            }
            else {
                neighbors[i] = { i }; // fallback to self...
            }
        }
    }

    Eigen::MatrixXf cur = RD.skelver;
    Eigen::MatrixXf next = RD.skelver;

    for (int it=0; it<cfg_.niter_smooth; ++it) {
        for (int i=0; i<V; ++i) {
            const auto& nbr = neighbors[i];
            const int n = static_cast<int>(nbr.size());
            if (n <= 1) {
                next.row(i) = cur.row(i);
                continue;
            }

            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (int j : nbr) {
                mean += cur.row(j).transpose();
            }
            mean /= static_cast<float>(n);

            Eigen::MatrixXf C = Eigen::Matrix3f::Zero();
            for (int j : nbr) {
                Eigen::Vector3f d = cur.row(j).transpose() - mean;
                C.noalias() += d * d.transpose();
            }
            C /= static_cast<float>(n);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
            const Eigen::Vector3f principal_dir = es.eigenvectors().col(2);

            // Project
            const Eigen::Vector3f p = cur.row(i).transpose();
            const Eigen::Vector3f delta = p - mean;
            const Eigen::Vector3f proj = mean + principal_dir * (principal_dir.dot(delta));

            // Fuse/Blend
            const Eigen::Vector3f fused = cfg_.alpha_recenter * p + (1.0f - cfg_.alpha_recenter) * proj;
            next.row(i) = fused.transpose();
        }

        cur.swap(next);
    }
    RD.skelver = cur;
    return 1;
}


/* HELPER FUNCTIONS */

float Rosa::similarity_metric(const Eigen::Vector3f& p1, const Eigen::Vector3f& v1, const Eigen::Vector3f& p2, const Eigen::Vector3f& v2, const float range_r, const float scale) {
    const Eigen::Vector3f d = p1 - p2;
    const float dot12_raw = v1.dot(v2);
    const float dot12 = std::max(0.0f, std::min(1.0f, dot12_raw));
    if (dot12 == 0.0f) return 0.0f; // antiparallel or orthogonal

    const float proj1 = d.dot(v1);
    const float proj2 = -d.dot(v2);

    float dist1 = (d + scale * proj1 * v1).norm() / range_r;
    float dist2 = (d - scale * proj2 * v2).norm() / range_r;

    auto bump = [](float t) {
        if (t>1.0f) return 0.0f;
        const float t2 = t * t;
        const float t3 = t2 * t;
        return 2.0f * t3 - 3.0f * t2 + 1.0f;
    };

    const float k1 = bump(dist1);
    if (k1 == 0.0f) return 0.0f;
    const float k2 = bump(dist2);
    if (k2 == 0.0f) return 0.0f;

    const float align2 = dot12 * dot12;
    return std::min(k1,k2) * align2;
}

std::vector<int> Rosa::compute_active_samples(const int seed, const Eigen::Vector3f& p, const Eigen::Vector3f& n) {
    std::vector<char> state(RD.pcd_size_, 0); // 0=unknown, 1=queued, 2=visited
    std::vector<int> out;
    std::vector<int> stack;

    auto in_slab = [&](int i)->bool {
        const auto& q = RD.pts_->points[i];
        float d = n[0]*(p[0]-q.x) + n[1]*(p[1]-q.y) + n[2]*(p[2]-q.z);
        return std::abs(d) < leaf_size;
    };

    if (!in_slab(seed)) return out;

    state[seed] = 1;
    stack.push_back(seed);

    while(!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        state[cur] = 2;
        out.push_back(cur);

        for (int nb : RD.simi_nbs[cur]) {
            if (state[nb] == 0 && in_slab(nb)) {
                state[nb] = 1;
                stack.push_back(nb);
            }
        }
    }
    return out;
}

Eigen::Vector3f Rosa::compute_symmetrynormal(const std::vector<int>& idxs) {
    const int m = (int)idxs.size();
    if (m==0) return Eigen::Vector3f::UnitX();

    Eigen::Vector3f sum_v = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    for (int idx : idxs) {
        const auto v = RD.nrms_->points[idx].getNormalVector3fMap();
        float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20) ? 1.0f / std::sqrt(s2) : 0.0f;
        const Eigen::Vector3f vn = inv_len * v;
        sum_v += vn;
        sum_outer.noalias() += vn * vn.transpose();
    }

    const float inv_m = 1.0f / static_cast<float>(m);
    const Eigen::Vector3f mu = sum_v * inv_m;
    Eigen::Matrix3f M = sum_outer * inv_m - (mu * mu.transpose()); //covariance
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(M);
    return es.eigenvectors().col(0);
}   

float Rosa::symmnormal_variance(Eigen::Vector3f& symm_nrm, std::vector<int>& idxs) {
    const int m = (int)idxs.size();
    if (m==0) return 0.0f;

    float sum_a = 0.0f;
    float sum_a2 = 0.0f;

    for (int idx : idxs) {
        const auto v = RD.nrms_->points[idx].getNormalVector3fMap();
        const float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20f) ? 1.0f / std::sqrt(s2) : 0.0f;
        
        const float a = v.dot(symm_nrm) * inv_len;
        sum_a += a;
        sum_a2 += a * a;
    }

    const float inv_m = 1.0f / static_cast<float>(m);
    const float mu = sum_a * inv_m;
    float var = sum_a2 * inv_m - mu * mu;
    if (m > 1) {
        var *= static_cast<float>(m) / static_cast<float>(m-1);
    }
    return var;
}

Eigen::Vector3f Rosa::cov_eigs_from_normals(const std::vector<int>& idxs) {
    Eigen::Vector3f sum_v = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    for (int i : idxs) {
        const auto v = RD.nrms_->points[i].getNormalVector3fMap();
        const float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20) ? 1.0f / std::sqrt(s2) : 0.0f;
        const Eigen::Vector3f vn = v * inv_len;
        sum_v += vn;
        sum_outer.noalias() += vn * vn.transpose();
    }

    const float inv_m = 1.0f / std::max<size_t>(1, idxs.size());
    const Eigen::Vector3f mu = sum_v * inv_m;
    const Eigen::Matrix3f C = sum_outer * inv_m - mu * mu.transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    return es.eigenvalues(); // returns eigenvalues in ascending order...
}

Eigen::Vector3f Rosa::closest_projection_point(const std::vector<int>& idxs) {
    if (idxs.empty()) return Eigen::Vector3f::Constant(1e8);

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    Eigen::Vector3f B = Eigen::Vector3f::Zero();

    for (int i : idxs) {
        const auto p = RD.pts_->points[i].getVector3fMap();
        const auto v = RD.nrms_->points[i].getNormalVector3fMap();
        const float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20f) ? (1.0 / std::sqrt(s2)) : 0.0f;
        const Eigen::Vector3f vn = v * inv_len;
        
        const Eigen::Matrix3f P = Eigen::Matrix3f::Identity() - (vn * vn.transpose());
        M.noalias() += P;
        B.noalias() += P * p;
    }

    const float tr = M.trace();
    M.diagonal().array() += 1e-6f * std::max(tr, 1e-6f);

    Eigen::Vector3f X;
    Eigen::LLT<Eigen::Matrix3f> llt(M);
    if (llt.info() == Eigen::Success) {
        X = llt.solve(B);
    }
    else {
        Eigen::LDLT<Eigen::Matrix3f> ldlt(M);
        if (ldlt.info() == Eigen::Success) {
            X = ldlt.solve(B);
        }
        else {
            X = Eigen::Vector3f::Constant(1e8f);
        }
    }
    return X;
}

bool Rosa::local_line_fit(const std::vector<int>& nb, Eigen::Vector3f& mean_out, Eigen::Vector3f& dir_out, float& conf_out, const int min_nb) {
    const int m = static_cast<int>(nb.size());
    const float EPS = 1e-6f;
    if (m < min_nb) {
        mean_out.setZero();
        dir_out = Eigen::Vector3f::UnitX();
        conf_out = 0.0f;
        return false;
    }

    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (int j : nb) {
        mean += pset.row(j).transpose();
    }
    mean /= static_cast<float>(m);
    
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (int j : nb) {
        Eigen::Vector3f q = pset.row(j).transpose() - mean;
        C.noalias() += q * q.transpose(); 
    }
    C /= static_cast<float>(m);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    if (es.info() != Eigen::Success) {
        mean_out = mean;
        dir_out = Eigen::Vector3f::UnitX();
        conf_out = 0.0f;
        return false;
    }

    const Eigen::Vector3f eval = es.eigenvalues();
    const Eigen::Matrix3f evec = es.eigenvectors();

    const float denom = std::max(EPS, eval.sum());
    mean_out = mean;
    dir_out = evec.col(2); // principal direction (largest eigenvalue)
    const float nrm = dir_out.norm();

    if (nrm > EPS) {
        dir_out /= nrm;
    }
    else {
        dir_out = Eigen::Vector3f::UnitX();
    }

    conf_out = eval(2) / denom;
    return true;
} 
