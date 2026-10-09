import numpy as np


def reaction_rate(T, Yf, Yo, B, Ta):
    return B * Yf * Yo * np.exp(-Ta / T)
