#ifndef RAYCAST_HPP_
#define RAYCAST_HPP_

#include <vector>
#include <unordered_set>
#include <stdlib.h>
#include <stdint.h>
#include <Eigen/Core>

struct VoxelKeyHasher {
    size_t operator()(const uint64_t& k) const noexcept { return std::hash<uint64_t>{}(k); }
};

using VoxelSeenSet = std::unordered_set<uint64_t, VoxelKeyHasher>;

struct CoverageState {
    float voxel_size;
    VoxelSeenSet seen;
};

struct RaySet {
    std::vector<Eigen::Vector3f> rays_cam;
    float max_range = 20.0;
};

struct GainStats {
    int new_surface = 0;
    int overlap_surface = 0;
};

#endif