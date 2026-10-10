#pragma once
#include <cmath>
#include <vector>

namespace ff1d {

struct Grid {
    int n;
    std::vector<double> r_face;
    std::vector<double> r_c;
    std::vector<double> volume;
    std::vector<double> area;

    Grid(double Rb, double Rout, int n_, double stretch)
        : n(n_), r_face(n_ + 1), r_c(n_), volume(n_), area(n_ + 1)
    {
        const double ratio = Rout / Rb;
        for (int i = 0; i <= n; ++i) {
            double x = static_cast<double>(i) / n;
            r_face[i] = Rb * std::pow(ratio, std::pow(x, stretch));
        }
        for (int i = 0; i < n; ++i) {
            r_c[i] = 0.5 * (r_face[i] + r_face[i + 1]);
            volume[i] = (4.0 / 3.0) * M_PI * (r_face[i + 1] * r_face[i + 1] * r_face[i + 1]
                                               - r_face[i] * r_face[i] * r_face[i]);
        }
        for (int i = 0; i <= n; ++i) {
            area[i] = 4.0 * M_PI * r_face[i] * r_face[i];
        }
    }
};

}
