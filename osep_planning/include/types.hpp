#ifndef TYPES_HPP_
#define TYPES_HPP_

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

#endif