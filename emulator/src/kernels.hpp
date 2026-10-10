#pragma once
#include <optional>
#include <string>
#include <vector>

#include "grid.hpp"
#include "params.hpp"

namespace ff1d {

std::vector<double> thomas_solve(std::vector<double> sub, std::vector<double> diag,
                                  std::vector<double> sup, std::vector<double> rhs);

void diffusion_coeffs(const Grid& g, const std::vector<double>& gamma,
                       double gamma_rb, double gamma_out,
                       std::vector<double>& c_left, std::vector<double>& c_right);

std::vector<double> implicit_diffuse(const Grid& g, const std::vector<double>& phi_rhs,
                                      const std::vector<double>& gamma,
                                      double phi_rb, double gamma_rb,
                                      double phi_out, double gamma_out, double dt,
                                      const std::vector<double>* diag_extra = nullptr,
                                      const std::vector<double>* rhs_extra = nullptr);

std::vector<double> compute_velocity(const Grid& g, const Params& p,
                                      const std::vector<double>& rho,
                                      const std::vector<double>& drho_dt);

std::vector<double> upwind_term(const Grid& g, const std::vector<double>& phi,
                                 const std::vector<double>& u, double phi_rb);

void react_step(std::vector<double>& T, std::vector<double>& Yf, std::vector<double>& Yo,
                 const Params& p, double dt_half);

void apply_ignition(std::vector<double>& T, const Grid& g, const Params& p, double t);

std::vector<double> heat_release(const std::vector<double>& T, const std::vector<double>& Yf,
                                  const std::vector<double>& Yo, const std::vector<double>& rho,
                                  const Params& p);

std::vector<double> radiative_loss_field(const std::vector<double>& T, const std::vector<double>& Yf,
                                          const std::vector<double>& Yo, const Params& p);

void radiative_loss_linearized(const std::vector<double>& T, const std::vector<double>& Yf,
                                const std::vector<double>& Yo, const Params& p,
                                std::vector<double>& q_rad, std::vector<double>& slope);

double flame_radius(const std::vector<double>& T, const std::vector<double>& Yf,
                     const std::vector<double>& Yo, const std::vector<double>& rho,
                     const Grid& g, const Params& p);

double radiative_fraction(const std::vector<double>& T, const std::vector<double>& Yf,
                           const std::vector<double>& Yo, const std::vector<double>& rho,
                           const Grid& g, const Params& p);

struct QoiTracker {
    double peak_heat_release = 0.0;
    bool ever_ignited = false;
    std::optional<double> t_ext;
    double fuel_injected = 0.0;

    void update(double t, double dt, const std::vector<double>& T, const std::vector<double>& Yf,
                const std::vector<double>& Yo, const std::vector<double>& rho,
                const Grid& g, const Params& p);

    double completeness(const std::vector<double>& Yf, const Grid& g) const;
    std::string regime(double chi_rad) const;
};

}
