#include "kernels.hpp"

namespace blueball {

void apply_bc(Grid& g, const Params& p) {
    const int ng = g.ng;
    const int NI = g.ni();
    const int NJ = g.nj();

    for (int j = 0; j < NJ; ++j) {
        for (int i = 0; i < ng; ++i) {
            g.at(T, j, i) = p.T_inf;
            g.at(F, j, i) = 0.0;
            g.at(O, j, i) = p.Yo_inf;
        }
    }

    for (int j = 0; j < NJ; ++j) {
        for (int i = NI - ng; i < NI; ++i) {
            for (int k = 0; k < 3; ++k)
                g.at(k, j, i) = g.at(k, j, NI - ng - 1);
        }
    }

    for (int j = NJ - ng; j < NJ; ++j) {
        for (int i = 0; i < NI; ++i) {
            for (int k = 0; k < 3; ++k)
                g.at(k, j, i) = g.at(k, NJ - ng - 1, i);
        }
    }

    for (int j = 0; j < ng; ++j) {
        for (int i = 0; i < NI; ++i) {
            double xi = g.x[i];
            bool on_fuel = (xi >= p.fuel_x0) && (xi <= p.fuel_x1);
            if (on_fuel) {
                g.at(T, j, i) = p.Ts;
                g.at(F, j, i) = p.Yf_s;
                g.at(O, j, i) = 0.0;
            } else {
                g.at(T, j, i) = g.at(T, ng, i);
                g.at(F, j, i) = g.at(F, ng, i);
                g.at(O, j, i) = g.at(O, ng, i);
            }
        }
    }
}

}
