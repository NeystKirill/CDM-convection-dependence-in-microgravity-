#include <algorithm>
#include <cmath>

#include "kernels.hpp"

namespace ff1d {

void react_step(std::vector<double>& T, std::vector<double>& Yf, std::vector<double>& Yo,
                 const Params& p, double dt_half) {
    const int n = static_cast<int>(T.size());

    for (int i = 0; i < n; ++i) {
        double T0 = T[i], Yf0 = Yf[i], Yo0 = Yo[i];

        double xi_max = Yo0 / p.s;
        if (Yf0 < xi_max) xi_max = Yf0;
        if (xi_max < 0.0) xi_max = 0.0;
        double xi = 0.0;

        for (int it = 0; it < p.n_iter; ++it) {
            double Tc = T0 + p.Qcp * xi;
            double Yfc = Yf0 - xi;
            double Yoc = Yo0 - p.s * xi;
            double expo = std::exp(-p.Ta / Tc);
            double omega = p.B * Yfc * Yoc * expo;

            double residual = xi - dt_half * omega;
            double domega_dxi = p.B * expo * (-Yoc - p.s * Yfc + Yfc * Yoc * p.Qcp * p.Ta / (Tc * Tc));
            double jac = 1.0 - dt_half * domega_dxi;
            if (std::abs(jac) < 1e-30) jac = 1e-30;

            xi = xi - residual / jac;
            if (xi < 0.0) xi = 0.0;
            else if (xi > xi_max) xi = xi_max;
        }

        T[i] = T0 + p.Qcp * xi;
        Yf[i] = Yf0 - xi;
        Yo[i] = Yo0 - p.s * xi;
    }
}

}
