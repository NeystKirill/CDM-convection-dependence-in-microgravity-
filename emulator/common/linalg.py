import numpy as np
from numba import njit


@njit(cache=True)
def thomas_solve(sub, diag, sup, rhs):
    n = diag.shape[0]
    for i in range(1, n):
        w = sub[i] / diag[i - 1]
        diag[i] -= w * sup[i - 1]
        rhs[i] -= w * rhs[i - 1]

    x = np.empty(n)
    x[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i]
    return x


def line_diffusion_coeffs(pos_face, pos_c, face_area, volume, gamma, gamma_lo, gamma_hi):
    n = volume.shape[0]
    c_left = np.zeros(n)
    c_right = np.zeros(n)

    gamma_face = 0.5 * (gamma[:-1] + gamma[1:])
    d = pos_c[1:] - pos_c[:-1]
    coef_mid = face_area[1:-1] * gamma_face / d

    c_left[1:] = coef_mid / volume[1:]
    c_right[:-1] = coef_mid / volume[:-1]

    d0 = pos_c[0] - pos_face[0]
    coef0 = face_area[0] * gamma_lo / d0
    c_left[0] = coef0 / volume[0]

    dn = pos_face[-1] - pos_c[-1]
    coefn = face_area[-1] * gamma_hi / dn
    c_right[-1] = coefn / volume[-1]

    return c_left, c_right


def implicit_diffuse_line(
    pos_face, pos_c, face_area, volume, gamma, phi,
    phi_lo, gamma_lo, phi_hi, gamma_hi, dt, diag_extra=None, rhs_extra=None,
):
    c_left, c_right = line_diffusion_coeffs(pos_face, pos_c, face_area, volume, gamma, gamma_lo, gamma_hi)

    sub = -dt * c_left
    sup = -dt * c_right
    diag = 1.0 + dt * (c_left + c_right)
    rhs = phi.copy()
    rhs[0] += dt * c_left[0] * phi_lo
    rhs[-1] += dt * c_right[-1] * phi_hi

    if diag_extra is not None:
        diag = diag + diag_extra
    if rhs_extra is not None:
        rhs = rhs + rhs_extra

    sub[0] = 0.0
    sup[-1] = 0.0

    return thomas_solve(sub, diag, sup, rhs)
