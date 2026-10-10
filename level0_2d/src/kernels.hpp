#pragma once
#include <cstddef>
#include <vector>

#include "grid.hpp"
#include "params.hpp"

namespace blueball {

inline size_t out_idx(int k, int ny, int nx, int j, int i) {
    return (static_cast<size_t>(k) * ny + j) * nx + i;
}

void apply_bc(Grid& g, const Params& p);
void advection_rhs(const Grid& g, const Params& p, std::vector<double>& out);
void diffusion_rhs(const Grid& g, const Params& p, std::vector<double>& out);
void react_step(Grid& g, const Params& p, double dt_half);
void apply_ignition(Grid& g, const Params& p, double t);

struct Diagnostics {
    double Tmax;
    double Yf_min;
    double Yo_min;
    double w_max;
    bool has_nan;
};

Diagnostics compute_diagnostics(const Grid& g, const Params& p);

}
