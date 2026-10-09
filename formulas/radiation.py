import numpy as np

SIGMA_SB = 5.670374419e-8


def radiative_loss(T, T_inf, a_p):
    return 4.0 * SIGMA_SB * a_p * (T**4 - T_inf**4)
