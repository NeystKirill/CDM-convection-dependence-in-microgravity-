#include <algorithm>

#include "kernels.hpp"

namespace blueball {

void apply_ignition(Grid& g, const Params& p, double t) {
    if (t >= p.ign_t) return;

    for (int j = 0; j < g.ny; ++j) {
        const int gj = g.ng + j;
        if (g.y[gj] > p.ign_y1) continue;

        for (int i = 0; i < g.nx; ++i) {
            const int gi = g.ng + i;
            double xi = g.x[gi];
            if (xi < p.ign_x0 || xi > p.ign_x1) continue;

            double& Tf = g.at(T, gj, gi);
            Tf = std::max(Tf, p.ign_T);
        }
    }
}

}
