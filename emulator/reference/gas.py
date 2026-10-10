import numpy as np

R_GAS = 8.314462618


def mean_molar_mass(Yf, Yo, Mf, Mo, Mp):
    Yp = 1.0 - Yf - Yo
    inv_w = Yf / Mf + Yo / Mo + Yp / Mp
    return 1.0 / inv_w


def density(p, T, W):
    return p * W / (R_GAS * T)


def conductivity(T, lam_ref, T_ref=300.0, exponent=0.7):
    return lam_ref * (T / T_ref) ** exponent


def species_diffusivity(lam, rho, cp, lewis):
    alpha = lam / (rho * cp)
    return alpha / lewis


def mole_to_mass_fraction(x, M, M_other):
    return x * M / (x * M + (1.0 - x) * M_other)
