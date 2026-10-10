import numpy as np


def flux_divergence(grid, phi, gamma, phi_rb, gamma_rb, phi_out, gamma_out):
    face_val = np.empty(grid.n + 1)

    gamma_face = 0.5 * (gamma[:-1] + gamma[1:])
    dphi = phi[1:] - phi[:-1]
    dr = grid.r_c[1:] - grid.r_c[:-1]
    face_val[1:-1] = grid.area[1:-1] * gamma_face * dphi / dr

    dr0 = grid.r_c[0] - grid.r_face[0]
    face_val[0] = grid.area[0] * gamma_rb * (phi[0] - phi_rb) / dr0

    drn = grid.r_face[-1] - grid.r_c[-1]
    face_val[-1] = grid.area[-1] * gamma_out * (phi_out - phi[-1]) / drn

    return (face_val[1:] - face_val[:-1]) / grid.volume
