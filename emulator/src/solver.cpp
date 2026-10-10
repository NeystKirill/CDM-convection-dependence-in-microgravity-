#include <algorithm>
#include <cmath>

#include "formulas.hpp"
#include "solver.hpp"

namespace ff1d {

Solver::Solver(Params params)
    : params_(params), grid_(params.Rb, params.Rout, params.n, params.stretch),
      T_(params.n, params.T_inf), Yf_(params.n, 0.0), Yo_(params.n, params.Yo_inf)
{
    rho_prev_ = density();
}

std::vector<double> Solver::density() const {
    const int n = params_.n;
    std::vector<double> rho(n);
    for (int i = 0; i < n; ++i) {
        double W = mean_molar_mass(Yf_[i], Yo_[i], params_.Mf, params_.Mo, params_.Mp);
        rho[i] = gas_density(params_.p, T_[i], W);
    }
    return rho;
}

double Solver::step() {
    const Params& p = params_;
    const Grid& g = grid_;
    const int n = p.n;

    if (t_ >= p.t_end) return 0.0;

    std::vector<double> rho = density();

    std::vector<double> lam(n), Df(n), Do(n), alpha(n);
    for (int i = 0; i < n; ++i) {
        lam[i] = conductivity(T_[i], p.lam_ref, 300.0, p.lam_exponent);
        alpha[i] = lam[i] / (rho[i] * p.cp);
        Df[i] = species_diffusivity(lam[i], rho[i], p.cp, p.Le_fuel);
        Do[i] = species_diffusivity(lam[i], rho[i], p.cp, p.Le_o2);
    }

    std::vector<double> drho_dt(n, 0.0);
    if (has_dt_) {
        for (int i = 0; i < n; ++i) drho_dt[i] = (rho[i] - rho_prev_[i]) / dt_;
    }
    std::vector<double> u = compute_velocity(g, p, rho, drho_dt);

    std::vector<double> dr(n);
    for (int i = 0; i < n; ++i) dr[i] = g.r_face[i + 1] - g.r_face[i];
    double dt_adv = 1e300;
    for (int i = 0; i < n; ++i) {
        double u_pos = std::max(std::abs(u[i]), 1e-12);
        dt_adv = std::min(dt_adv, p.cfl * dr[i] / u_pos);
    }
    double dt = std::min(dt_adv, p.t_end - t_);
    dt_ = dt;
    has_dt_ = true;

    react_step(T_, Yf_, Yo_, p, dt / 2.0);

    std::vector<double> T_adv(n), Yf_adv(n), Yo_adv(n);
    {
        std::vector<double> dT = upwind_term(g, T_, u, p.Tb);
        std::vector<double> dYf = upwind_term(g, Yf_, u, p.YF_inj);
        std::vector<double> dYo = upwind_term(g, Yo_, u, 0.0);
        for (int i = 0; i < n; ++i) {
            T_adv[i] = T_[i] + dt * dT[i];
            Yf_adv[i] = Yf_[i] + dt * dYf[i];
            Yo_adv[i] = Yo_[i] + dt * dYo[i];
        }
    }

    std::vector<double> q_rad, rad_slope;
    radiative_loss_linearized(T_, Yf_, Yo_, p, q_rad, rad_slope);
    std::vector<double> diag_extra(n), rhs_extra(n);
    for (int i = 0; i < n; ++i) {
        double rad_const = q_rad[i] - rad_slope[i] * T_[i];
        diag_extra[i] = dt * rad_slope[i] / (rho[i] * p.cp);
        rhs_extra[i] = -dt * rad_const / (rho[i] * p.cp);
    }

    T_ = implicit_diffuse(g, T_adv, alpha, p.Tb, alpha[0], p.T_inf, alpha[n - 1], dt, &diag_extra, &rhs_extra);
    Yf_ = implicit_diffuse(g, Yf_adv, Df, p.YF_inj, Df[0], 0.0, Df[n - 1], dt);
    Yo_ = implicit_diffuse(g, Yo_adv, Do, 0.0, Do[0], p.Yo_inf, Do[n - 1], dt);
    for (int i = 0; i < n; ++i) {
        Yf_[i] = clip01(Yf_[i]);
        Yo_[i] = clip01(Yo_[i]);
    }

    apply_ignition(T_, g, p, t_);

    react_step(T_, Yf_, Yo_, p, dt / 2.0);

    tracker_.update(t_ + dt, dt, T_, Yf_, Yo_, rho, g, p);
    rho_prev_ = rho;
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

double Solver::T_max() const {
    return *std::max_element(T_.begin(), T_.end());
}

double Solver::r_f() const {
    std::vector<double> rho = density();
    return flame_radius(T_, Yf_, Yo_, rho, grid_, params_);
}

double Solver::chi_rad() const {
    std::vector<double> rho = density();
    return radiative_fraction(T_, Yf_, Yo_, rho, grid_, params_);
}

double Solver::eta() const {
    return tracker_.completeness(Yf_, grid_);
}

}
