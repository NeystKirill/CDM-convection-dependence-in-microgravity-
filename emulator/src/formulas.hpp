#pragma once
#include <cmath>

namespace blueball {

constexpr double SIGMA_SB = 5.670374419e-8;

inline double reaction_rate(double T, double Yf, double Yo, double B, double Ta) {
    return B * Yf * Yo * std::exp(-Ta / T);
}

inline double radiative_loss(double T, double T_inf, double a_p) {
    double T2 = T * T;
    double Tinf2 = T_inf * T_inf;
    return 4.0 * SIGMA_SB * a_p * (T2 * T2 - Tinf2 * Tinf2);
}

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

}
