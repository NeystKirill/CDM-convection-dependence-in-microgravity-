#include "kernels.hpp"

namespace ff1d {

std::vector<double> upwind_term(const Grid& g, const std::vector<double>& phi,
                                 const std::vector<double>& u, double phi_rb) {
    const int n = g.n;
    std::vector<double> grad(n);

    double dr0 = g.r_c[0] - g.r_face[0];
    grad[0] = (phi[0] - phi_rb) / dr0;

    for (int i = 1; i < n; ++i) {
        double dr = g.r_c[i] - g.r_c[i - 1];
        grad[i] = (phi[i] - phi[i - 1]) / dr;
    }

    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = -u[i] * grad[i];
    return out;
}

}
