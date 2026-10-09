#pragma once
#include <cstddef>
#include <vector>

#include "params.hpp"

namespace blueball {

constexpr int GHOST = 2;
enum Field { T = 0, F = 1, O = 2 };

struct Grid {
    int nx, ny, ng;
    double dx, dy;
    std::vector<double> q;
    std::vector<double> x, y;

    explicit Grid(const Params& p)
        : nx(p.nx), ny(p.ny), ng(GHOST), dx(p.dx), dy(p.dy),
          q(3 * static_cast<size_t>(ny + 2 * ng) * (nx + 2 * ng), 0.0),
          x(nx + 2 * ng), y(ny + 2 * ng)
    {
        for (int i = 0; i < nx + 2 * ng; ++i) x[i] = (i - ng + 0.5) * dx;
        for (int j = 0; j < ny + 2 * ng; ++j) y[j] = (j - ng + 0.5) * dy;
    }

    int ni() const { return nx + 2 * ng; }
    int nj() const { return ny + 2 * ng; }

    size_t idx(int k, int j, int i) const {
        return (static_cast<size_t>(k) * nj() + j) * ni() + i;
    }

    double& at(int k, int j, int i) { return q[idx(k, j, i)]; }
    double at(int k, int j, int i) const { return q[idx(k, j, i)]; }
};

}
