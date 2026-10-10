#include <algorithm>
#include <cmath>

#include "formulas.hpp"
#include "kernels.hpp"

namespace ff1d {

namespace {

double absorption_coeff(double Yf, double Yo, const Params& p) {
    double Yp = clip01(1.0 - Yf - Yo);
    double p_atm = p.p / 101325.0;
    return p.rad_mult * p.a_p_ref * p_atm * Yp;
}

}

std::vector<double> heat_release(const std::vector<double>& T, const std::vector<double>& Yf,
                                  const std::vector<double>& Yo, const std::vector<double>& rho,
                                  const Params& p) {
    const int n = static_cast<int>(T.size());
    std::vector<double> q(n);
    for (int i = 0; i < n; ++i) {
        double omega = reaction_rate(T[i], Yf[i], Yo[i], p.B, p.Ta);
        q[i] = rho[i] * p.cp * p.Qcp * omega;
    }
    return q;
}

std::vector<double> radiative_loss_field(const std::vector<double>& T, const std::vector<double>& Yf,
                                          const std::vector<double>& Yo, const Params& p) {
    const int n = static_cast<int>(T.size());
    std::vector<double> q(n, 0.0);
    if (!p.rad_enabled) return q;
    for (int i = 0; i < n; ++i) {
        double a_p = absorption_coeff(Yf[i], Yo[i], p);
        q[i] = radiative_loss(T[i], p.T_inf, a_p);
    }
    return q;
}

void radiative_loss_linearized(const std::vector<double>& T, const std::vector<double>& Yf,
                                const std::vector<double>& Yo, const Params& p,
                                std::vector<double>& q_rad, std::vector<double>& slope) {
    const int n = static_cast<int>(T.size());
    q_rad.assign(n, 0.0);
    slope.assign(n, 0.0);
    if (!p.rad_enabled) return;
    for (int i = 0; i < n; ++i) {
        double a_p = absorption_coeff(Yf[i], Yo[i], p);
        q_rad[i] = radiative_loss(T[i], p.T_inf, a_p);
        slope[i] = 16.0 * SIGMA_SB * a_p * T[i] * T[i] * T[i];
    }
}

double flame_radius(const std::vector<double>& T, const std::vector<double>& Yf,
                     const std::vector<double>& Yo, const std::vector<double>& rho,
                     const Grid& g, const Params& p) {
    std::vector<double> q = heat_release(T, Yf, Yo, rho, p);
    int idx = static_cast<int>(std::max_element(q.begin(), q.end()) - q.begin());
    return g.r_c[idx];
}

double radiative_fraction(const std::vector<double>& T, const std::vector<double>& Yf,
                           const std::vector<double>& Yo, const std::vector<double>& rho,
                           const Grid& g, const Params& p) {
    std::vector<double> q_chem = heat_release(T, Yf, Yo, rho, p);
    std::vector<double> q_rad = radiative_loss_field(T, Yf, Yo, p);

    double total_chem = 0.0, total_rad = 0.0;
    for (int i = 0; i < g.n; ++i) {
        total_chem += q_chem[i] * g.volume[i];
        total_rad += q_rad[i] * g.volume[i];
    }
    if (total_chem <= 0.0) return 0.0;
    return total_rad / total_chem;
}

void QoiTracker::update(double t, double dt, const std::vector<double>& T, const std::vector<double>& Yf,
                         const std::vector<double>& Yo, const std::vector<double>& rho,
                         const Grid& g, const Params& p) {
    std::vector<double> q_chem = heat_release(T, Yf, Yo, rho, p);
    double total_chem = 0.0;
    for (int i = 0; i < g.n; ++i) total_chem += q_chem[i] * g.volume[i];
    peak_heat_release = std::max(peak_heat_release, total_chem);
    fuel_injected += p.mdot * p.YF_inj * dt;

    double T_max = *std::max_element(T.begin(), T.end());
    if (T_max >= 1000.0) ever_ignited = true;

    if (!t_ext.has_value() && ever_ignited && peak_heat_release > 0.0
        && total_chem < 0.01 * peak_heat_release) {
        t_ext = t;
    }
}

double QoiTracker::completeness(const std::vector<double>& Yf, const Grid& g) const {
    if (fuel_injected <= 0.0) return 0.0;
    double remaining = 0.0;
    for (int i = 0; i < g.n; ++i) remaining += Yf[i] * g.volume[i];
    return clip01(1.0 - remaining / fuel_injected);
}

std::string QoiTracker::regime(double chi_rad) const {
    if (!ever_ignited) return "no_ignition";
    if (!t_ext.has_value()) return "steady";
    if (chi_rad > 0.3) return "radiative_extinction";
    return "kinetic_extinction";
}

}
