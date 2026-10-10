#include <cmath>

#include "kernels.hpp"

namespace ff1d {

std::vector<double> compute_velocity(const Grid& g, const Params& p,
                                      const std::vector<double>& rho,
                                      const std::vector<double>& drho_dt) {
    const int n = g.n;
    std::vector<double> mdot_face(n + 1);
    mdot_face[0] = p.mdot;
    double accum = 0.0;
    for (int i = 0; i < n; ++i) {
        accum += drho_dt[i] * g.volume[i];
        mdot_face[i + 1] = p.mdot - accum;
    }

    std::vector<double> u(n);
    for (int i = 0; i < n; ++i) {
        double mdot_c = 0.5 * (mdot_face[i] + mdot_face[i + 1]);
        u[i] = mdot_c / (rho[i] * 4.0 * M_PI * g.r_c[i] * g.r_c[i]);
    }
    return u;
}

}
