#include "solver.hpp"
#include "formulas.hpp"

namespace blueball {

Solver::Solver(Params params)
    : params_(params), grid_(params_), grid1_(params_),
      rhs_buf_(3 * static_cast<size_t>(params_.ny) * params_.nx, 0.0),
      adv_buf_(rhs_buf_.size(), 0.0),
      diff_buf_(rhs_buf_.size(), 0.0)
{
    for (int j = 0; j < grid_.ny; ++j) {
        const int gj = grid_.ng + j;
        for (int i = 0; i < grid_.nx; ++i) {
            const int gi = grid_.ng + i;
            grid_.at(T, gj, gi) = params_.T_inf;
            grid_.at(F, gj, gi) = 0.0;
            grid_.at(O, gj, gi) = params_.Yo_inf;
        }
    }
    apply_bc(grid_, params_);
}

void Solver::rhs(const Grid& g, std::vector<double>& out) {
    advection_rhs(g, params_, adv_buf_);
    diffusion_rhs(g, params_, diff_buf_);

    const int ny = g.ny, nx = g.nx, ng = g.ng;
    for (size_t n = 0; n < out.size(); ++n)
        out[n] = adv_buf_[n] + diff_buf_[n];

    for (int j = 0; j < ny; ++j) {
        const int gj = ng + j;
        for (int i = 0; i < nx; ++i) {
            const int gi = ng + i;
            double Tf = g.at(T, gj, gi);
            double qrad = radiative_loss(Tf, params_.T_inf, params_.a_p);
            out[out_idx(T, ny, nx, j, i)] -= qrad / (params_.rho * params_.cp);
        }
    }
}

double Solver::step() {
    const double dt = params_.dt;
    const int ny = grid_.ny, nx = grid_.nx, ng = grid_.ng;

    react_step(grid_, params_, dt / 2.0);
    apply_bc(grid_, params_);

    rhs(grid_, rhs_buf_);
    grid1_.q = grid_.q;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < ny; ++j) {
            const int gj = ng + j;
            for (int i = 0; i < nx; ++i) {
                const int gi = ng + i;
                grid1_.at(k, gj, gi) = grid_.at(k, gj, gi) + dt * rhs_buf_[out_idx(k, ny, nx, j, i)];
            }
        }
    }
    apply_bc(grid1_, params_);

    rhs(grid1_, rhs_buf_);
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < ny; ++j) {
            const int gj = ng + j;
            for (int i = 0; i < nx; ++i) {
                const int gi = ng + i;
                double qv = grid_.at(k, gj, gi);
                double q1v = grid1_.at(k, gj, gi);
                grid_.at(k, gj, gi) = 0.5 * qv + 0.5 * (q1v + dt * rhs_buf_[out_idx(k, ny, nx, j, i)]);
            }
        }
    }
    apply_bc(grid_, params_);

    apply_ignition(grid_, params_, t_);

    react_step(grid_, params_, dt / 2.0);
    apply_bc(grid_, params_);

    t_ += dt;
    ++n_steps_;
    return dt;
}

void Solver::run(double t_end, long max_steps) {
    while (t_ < t_end) {
        if (max_steps >= 0 && n_steps_ >= max_steps) break;
        step();
    }
}

}
