#include "kernels.hpp"

namespace blueball {

void diffusion_rhs(const Grid& g, const Params& p, std::vector<double>& out) {
    const int ny = g.ny, nx = g.nx, ng = g.ng;
    const double dx2 = g.dx * g.dx;
    const double dy2 = g.dy * g.dy;

    for (int k = 0; k < 3; ++k) {
        double coef = (k == T) ? p.alpha : p.D;
        for (int j = 0; j < ny; ++j) {
            const int gj = ng + j;
            for (int i = 0; i < nx; ++i) {
                const int gi = ng + i;

                double center = g.at(k, gj, gi);
                double left = g.at(k, gj, gi - 1);
                double right = g.at(k, gj, gi + 1);
                double down = g.at(k, gj - 1, gi);
                double up = g.at(k, gj + 1, gi);

                double lap = (left - 2.0 * center + right) / dx2
                           + (down - 2.0 * center + up) / dy2;

                out[out_idx(k, ny, nx, j, i)] = coef * lap;
            }
        }
    }
}

}
