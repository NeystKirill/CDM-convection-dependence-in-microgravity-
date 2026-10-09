#include "formulas.hpp"
#include "kernels.hpp"

namespace blueball {

void advection_rhs(const Grid& g, const Params& p, std::vector<double>& out) {
    const int ny = g.ny, nx = g.nx, ng = g.ng;

    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < ny; ++j) {
            const int gj = ng + j;
            for (int i = 0; i < nx; ++i) {
                const int gi = ng + i;

                double q_im2 = g.at(k, gj, gi - 2);
                double q_im1 = g.at(k, gj, gi - 1);
                double q_i = g.at(k, gj, gi);
                double q_ip1 = g.at(k, gj, gi + 1);

                double slope_i = minmod(q_i - q_im1, q_ip1 - q_i);
                double slope_im1 = minmod(q_im1 - q_im2, q_i - q_im1);

                double face_ip12 = q_i + 0.5 * slope_i;
                double face_im12 = q_im1 + 0.5 * slope_im1;

                out[out_idx(k, ny, nx, j, i)] = -p.U * (face_ip12 - face_im12) / g.dx;
            }
        }
    }
}

}
