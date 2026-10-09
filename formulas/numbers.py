import numpy as np


def peclet(U, L, alpha):
    return U * L / alpha


def reaction_time_scale(B, Yf_ref, Yo_ref, Ta, Tf):
    rate = B * Yf_ref * Yo_ref * np.exp(-Ta / Tf)
    return 1.0 / rate


def damkohler(flow_time, reaction_time):
    return flow_time / reaction_time


def zeldovich(Ta, Tf, T_inf):
    return Ta * (Tf - T_inf) / Tf**2
