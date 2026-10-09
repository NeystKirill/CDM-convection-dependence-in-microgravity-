#include <algorithm>
#include <cmath>
#include <limits>

#include "formulas.hpp"
#include "kernels.hpp"

namespace blueball {

Diagnostics compute_diagnostics(const Grid& g, const Params& p) {
    double Tmax = -std::numeric_limits<double>::infinity();
    double Yf_min = std::numeric_limits<double>::infinity();
    double Yo_min = std::numeric_limits<double>::infinity();
    double w_max = -std::numeric_limits<double>::infinity();

    for (int j = 0; j < g.ny; ++j) {
        const int gj = g.ng + j;
        for (int i = 0; i < g.nx; ++i) {
            const int gi = g.ng + i;
            double Tf = g.at(T, gj, gi);
            double Yf = g.at(F, gj, gi);
            double Yo = g.at(O, gj, gi);

            Tmax = std::max(Tmax, Tf);
            Yf_min = std::min(Yf_min, Yf);
            Yo_min = std::min(Yo_min, Yo);
            w_max = std::max(w_max, reaction_rate(Tf, Yf, Yo, p.B, p.Ta));
        }
    }

    bool has_nan = false;
    for (double v : g.q) {
        if (!std::isfinite(v)) {
            has_nan = true;
            break;
        }
    }

    return Diagnostics{Tmax, Yf_min, Yo_min, w_max, has_nan};
}

}
