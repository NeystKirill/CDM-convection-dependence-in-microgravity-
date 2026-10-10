#include <algorithm>
#include <cmath>

#include "kernels.hpp"

namespace blueball {

constexpr int N_ITER = 30;

void react_step(Grid& g, const Params& p, double dt_half) {
    const int ny = g.ny, nx = g.nx, ng = g.ng;

    for (int j = 0; j < ny; ++j) {
        const int gj = ng + j;
        for (int i = 0; i < nx; ++i) {
            const int gi = ng + i;

            double T0 = g.at(T, gj, gi);
            double Yf0 = g.at(F, gj, gi);
            double Yo0 = g.at(O, gj, gi);

            double xi_max = std::max(std::min(Yf0, Yo0 / p.s), 0.0);
            double xi = 0.0;

            for (int it = 0; it < N_ITER; ++it) {
                double Tc = T0 + p.Qcp * xi;
                double Yfc = Yf0 - xi;
                double Yoc = Yo0 - p.s * xi;
                double expo = std::exp(-p.Ta / Tc);
                double omega = p.B * Yfc * Yoc * expo;

                double residual = xi - dt_half * omega;
                double domega_dxi = p.B * expo * (-Yoc - p.s * Yfc
                                     + Yfc * Yoc * p.Qcp * p.Ta / (Tc * Tc));
                double jac = 1.0 - dt_half * domega_dxi;
                if (std::abs(jac) < 1e-30) jac = 1e-30;

                double xi_new = std::clamp(xi - residual / jac, 0.0, xi_max);
                bool converged = std::abs(xi_new - xi) < 1e-14;
                xi = xi_new;
                if (converged) break;
            }

            g.at(T, gj, gi) = T0 + p.Qcp * xi;
            g.at(F, gj, gi) = Yf0 - xi;
            g.at(O, gj, gi) = Yo0 - p.s * xi;
        }
    }
}

}
