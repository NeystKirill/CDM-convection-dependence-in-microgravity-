import numpy as np


def minmod(a, b):
    same_sign = (a * b) > 0
    return np.where(same_sign, np.where(np.abs(a) < np.abs(b), a, b), 0.0)
