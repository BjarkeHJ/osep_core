#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include "types.hpp"
#include <iostream>
#include <chrono>
#include <unordered_set>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)


struct PlannerConfig {
    int test;
};

struct PlannerData {
    size_t gskel_size;
    std::vector<Vertex> gskel;
    std::unordered_map<int,int> gskel_vid2idx;

    Viewpoint start;
    Viewpoint end;
};


class PathPlanner {
public:
    PathPlanner(const PlannerConfig& cfg);
    bool planner_run();
    void update_skeleton(const std::vector<Vertex>& verts);

    std::vector<Vertex>& output_skeleton() { return PD.gskel; }
    
private:
    /* Functions */

    bool generate_path();

    /* Helper */
    void merge_viewpoints(Vertex& vcur, const Vertex& vin);

    static inline bool approx_eq(float a, float b, float eps=1e-6f){ return std::fabs(a-b) <= eps; }
    static inline bool approx_eq3(const Eigen::Vector3f& a,const Eigen::Vector3f& b,float eps2=1e-8f){ return (a - b).squaredNorm() <= eps2; }


    /* Params */
    PlannerConfig cfg_;
    bool running;

    /* Data */
    PlannerData PD;
};


#endif