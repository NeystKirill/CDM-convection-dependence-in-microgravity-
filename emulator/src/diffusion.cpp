#include "kernels.hpp"

namespace ff1d {

std::vector<double> thomas_solve(std::vector<double> sub, std::vector<double> diag,
                                  std::vector<double> sup, std::vector<double> rhs) {
    const int n = static_cast<int>(diag.size());
    for (int i = 1; i < n; ++i) {
        double w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    std::vector<double> x(n);
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i];
    }
    return x;
}

void diffusion_coeffs(const Grid& g, const std::vector<double>& gamma,
                       double gamma_rb, double gamma_out,
                       std::vector<double>& c_left, std::vector<double>& c_right) {
    const int n = g.n;
    c_left.assign(n, 0.0);
    c_right.assign(n, 0.0);

    for (int i = 1; i < n; ++i) {
        double gamma_face = 0.5 * (gamma[i - 1] + gamma[i]);
        double dr = g.r_c[i] - g.r_c[i - 1];
        double coef = g.area[i] * gamma_face / dr;
        c_left[i] = coef / g.volume[i];
        c_right[i - 1] = coef / g.volume[i - 1];
    }

    double dr0 = g.r_c[0] - g.r_face[0];
    double coef0 = g.area[0] * gamma_rb / dr0;
    c_left[0] = coef0 / g.volume[0];

    double drn = g.r_face[n] - g.r_c[n - 1];
    double coefn = g.area[n] * gamma_out / drn;
    c_right[n - 1] = coefn / g.volume[n - 1];
}

std::vector<double> implicit_diffuse(const Grid& g, const std::vector<double>& phi_rhs,
                                      const std::vector<double>& gamma,
                                      double phi_rb, double gamma_rb,
                                      double phi_out, double gamma_out, double dt,
                                      const std::vector<double>* diag_extra,
                                      const std::vector<double>* rhs_extra) {
    const int n = g.n;
    std::vector<double> c_left, c_right;
    diffusion_coeffs(g, gamma, gamma_rb, gamma_out, c_left, c_right);

    std::vector<double> sub(n), diag(n), sup(n), rhs(phi_rhs);
    for (int i = 0; i < n; ++i) {
        sub[i] = -dt * c_left[i];
        sup[i] = -dt * c_right[i];
        diag[i] = 1.0 + dt * (c_left[i] + c_right[i]);
    }
    rhs[0] += dt * c_left[0] * phi_rb;
    rhs[n - 1] += dt * c_right[n - 1] * phi_out;

    if (diag_extra) {
        for (int i = 0; i < n; ++i) diag[i] += (*diag_extra)[i];
    }
    if (rhs_extra) {
        for (int i = 0; i < n; ++i) rhs[i] += (*rhs_extra)[i];
    }

    sub[0] = 0.0;
    sup[n - 1] = 0.0;

    return thomas_solve(sub, diag, sup, rhs);
}

}
