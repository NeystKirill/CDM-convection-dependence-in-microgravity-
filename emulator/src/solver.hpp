#pragma once
#include <vector>

#include "grid.hpp"
#include "kernels.hpp"
#include "params.hpp"

namespace ff1d {

class Solver {
public:
    explicit Solver(Params params);

    double step();
    void run(double t_end, long max_steps = -1);

    const Grid& grid() const { return grid_; }
    const Params& params() const { return params_; }
    double time() const { return t_; }
    long steps() const { return n_steps_; }

    std::vector<double> density() const;

    const std::vector<double>& T() const { return T_; }
    const std::vector<double>& Yf() const { return Yf_; }
    const std::vector<double>& Yo() const { return Yo_; }

    double T_max() const;
    double r_f() const;
    double chi_rad() const;
    double eta() const;
    const QoiTracker& tracker() const { return tracker_; }

private:
    Params params_;
    Grid grid_;
    std::vector<double> T_, Yf_, Yo_;
    std::vector<double> rho_prev_;
    double t_ = 0.0;
    long n_steps_ = 0;
    double dt_ = 0.0;
    bool has_dt_ = false;
    QoiTracker tracker_;
};

}
