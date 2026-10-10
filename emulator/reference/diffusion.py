import numpy as np

from emulator.common.linalg import thomas_solve


def diffusion_coeffs(grid, gamma, gamma_rb, gamma_out):
    n = grid.n
    c_left = np.zeros(n)
    c_right = np.zeros(n)

    gamma_face = 0.5 * (gamma[:-1] + gamma[1:])
    dr = grid.r_c[1:] - grid.r_c[:-1]
    coef_mid = grid.area[1:-1] * gamma_face / dr

    c_left[1:] = coef_mid / grid.volume[1:]
    c_right[:-1] = coef_mid / grid.volume[:-1]

    dr0 = grid.r_c[0] - grid.r_face[0]
    coef0 = grid.area[0] * gamma_rb / dr0
    c_left[0] = coef0 / grid.volume[0]

    drn = grid.r_face[-1] - grid.r_c[-1]
    coefn = grid.area[-1] * gamma_out / drn
    c_right[-1] = coefn / grid.volume[-1]

    return c_left, c_right


def implicit_diffuse(grid, phi_rhs, gamma, phi_rb, gamma_rb, phi_out, gamma_out, dt, diag_extra=None, rhs_extra=None):
    c_left, c_right = diffusion_coeffs(grid, gamma, gamma_rb, gamma_out)

    sub = -dt * c_left
    sup = -dt * c_right
    diag = 1.0 + dt * (c_left + c_right)
    rhs = phi_rhs.copy()
    rhs[0] += dt * c_left[0] * phi_rb
    rhs[-1] += dt * c_right[-1] * phi_out

    if diag_extra is not None:
        diag = diag + diag_extra
    if rhs_extra is not None:
        rhs = rhs + rhs_extra

    sub[0] = 0.0
    sup[-1] = 0.0

    return thomas_solve(sub, diag, sup, rhs)
