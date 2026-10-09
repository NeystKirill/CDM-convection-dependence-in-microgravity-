#pragma once
#include <vector>

#include "grid.hpp"
#include "kernels.hpp"
#include "params.hpp"

namespace blueball {

class Solver {
public:
    explicit Solver(Params params);

    double step();
    void run(double t_end, long max_steps = -1);

    const Grid& grid() const { return grid_; }
    const Params& params() const { return params_; }
    double time() const { return t_; }
    long steps() const { return n_steps_; }
    Diagnostics diagnostics() const { return compute_diagnostics(grid_, params_); }

private:
    Params params_;
    Grid grid_;
    Grid grid1_;
    double t_ = 0.0;
    long n_steps_ = 0;

    std::vector<double> rhs_buf_;
    std::vector<double> adv_buf_;
    std::vector<double> diff_buf_;

    void rhs(const Grid& g, std::vector<double>& out);
};

}
