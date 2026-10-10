import numpy as np


def face_mass_flow(grid, mdot, drho_dt):
    accum = np.cumsum(drho_dt * grid.volume)
    return mdot - np.concatenate(([0.0], accum))


def cell_velocity(grid, mdot_face, rho):
    mdot_c = 0.5 * (mdot_face[:-1] + mdot_face[1:])
    return mdot_c / (rho * 4.0 * np.pi * grid.r_c**2)
