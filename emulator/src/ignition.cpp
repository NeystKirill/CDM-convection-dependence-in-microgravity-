#include <algorithm>

#include "kernels.hpp"

namespace ff1d {

void apply_ignition(std::vector<double>& T, const Grid& g, const Params& p, double t) {
    if (t >= p.tign) return;

    double kernel_edge = g.r_face[0] + p.ignition_kernel_Rb * p.Rb;
    for (int i = 0; i < g.n; ++i) {
        if (g.r_c[i] <= kernel_edge) {
            T[i] = std::max(T[i], p.Tign);
        }
    }
}

}
