#pragma once
#include <cmath>

namespace ff1d {

constexpr double SIGMA_SB = 5.670374419e-8;
constexpr double R_GAS = 8.314462618;

inline double mean_molar_mass(double Yf, double Yo, double Mf, double Mo, double Mp) {
    double Yp = 1.0 - Yf - Yo;
    double inv_w = Yf / Mf + Yo / Mo + Yp / Mp;
    return 1.0 / inv_w;
}

inline double gas_density(double p, double T, double W) {
    return p * W / (R_GAS * T);
}

inline double conductivity(double T, double lam_ref, double T_ref = 300.0, double exponent = 0.7) {
    return lam_ref * std::pow(T / T_ref, exponent);
}

inline double species_diffusivity(double lam, double rho, double cp, double lewis) {
    double alpha = lam / (rho * cp);
    return alpha / lewis;
}

inline double reaction_rate(double T, double Yf, double Yo, double B, double Ta) {
    return B * Yf * Yo * std::exp(-Ta / T);
}

inline double radiative_loss(double T, double T_inf, double a_p) {
    double T2 = T * T, Tinf2 = T_inf * T_inf;
    return 4.0 * SIGMA_SB * a_p * (T2 * T2 - Tinf2 * Tinf2);
}

inline double clip01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

}
