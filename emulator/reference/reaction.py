import numpy as np
from numba import njit


@njit(cache=True)
def _react_core(T, Yf, Yo, B, Ta, Qcp, s, dt_half, n_iter):
    n = T.shape[0]
    T_out = np.empty(n)
    Yf_out = np.empty(n)
    Yo_out = np.empty(n)

    for i in range(n):
        T0, Yf0, Yo0 = T[i], Yf[i], Yo[i]
        xi_max = Yo0 / s
        if Yf0 < xi_max:
            xi_max = Yf0
        if xi_max < 0.0:
            xi_max = 0.0
        xi = 0.0

        for _ in range(n_iter):
            Tc = T0 + Qcp * xi
            Yfc = Yf0 - xi
            Yoc = Yo0 - s * xi
            expo = np.exp(-Ta / Tc)
            omega = B * Yfc * Yoc * expo

            residual = xi - dt_half * omega
            domega_dxi = B * expo * (-Yoc - s * Yfc + Yfc * Yoc * Qcp * Ta / Tc**2)
            jac = 1.0 - dt_half * domega_dxi
            if abs(jac) < 1e-30:
                jac = 1e-30

            xi = xi - residual / jac
            if xi < 0.0:
                xi = 0.0
            elif xi > xi_max:
                xi = xi_max

        T_out[i] = T0 + Qcp * xi
        Yf_out[i] = Yf0 - xi
        Yo_out[i] = Yo0 - s * xi

    return T_out, Yf_out, Yo_out


def react_step(T, Yf, Yo, params, dt_half):
    return _react_core(T, Yf, Yo, params.B, params.Ta, params.Qcp, params.s, dt_half, params.n_iter)


def reaction_rate(T, Yf, Yo, params):
    return params.B * Yf * Yo * np.exp(-params.Ta / T)
