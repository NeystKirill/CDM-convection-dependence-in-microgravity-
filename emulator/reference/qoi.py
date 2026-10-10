import numpy as np

from formulas import radiative_loss, SIGMA_SB
from .reaction import reaction_rate

IGNITED_T = 1000.0
EXTINCT_FRACTION = 0.01


def heat_release(T, Yf, Yo, rho, params):
    omega = reaction_rate(T, Yf, Yo, params)
    return rho * params.cp * params.Qcp * omega


def absorption_coeff(Yf, Yo, params):
    Yp = 1.0 - Yf - Yo
    p_atm = params.p / 101325.0
    return params.rad_mult * params.a_p_ref * p_atm * np.clip(Yp, 0.0, 1.0)


def radiative_loss_field(T, Yf, Yo, params):
    if not params.rad_enabled:
        return np.zeros_like(T)
    a_p = absorption_coeff(Yf, Yo, params)
    return radiative_loss(T, params.T_inf, a_p)


def radiative_loss_linearized(T, Yf, Yo, params):
    if not params.rad_enabled:
        zeros = np.zeros_like(T)
        return zeros, zeros
    a_p = absorption_coeff(Yf, Yo, params)
    q_rad = radiative_loss(T, params.T_inf, a_p)
    slope = 16.0 * SIGMA_SB * a_p * T**3
    return q_rad, slope


def flame_radius(T, Yf, Yo, rho, grid, params):
    q = heat_release(T, Yf, Yo, rho, params)
    return float(grid.r_c[np.argmax(q)])


def radiative_fraction(T, Yf, Yo, rho, grid, params):
    q_chem = heat_release(T, Yf, Yo, rho, params)
    q_rad = radiative_loss_field(T, Yf, Yo, params)
    total_chem = float(np.sum(q_chem * grid.volume))
    total_rad = float(np.sum(q_rad * grid.volume))
    if total_chem <= 0.0:
        return 0.0
    return total_rad / total_chem


class QoiTracker:
    def __init__(self):
        self.peak_heat_release = 0.0
        self.ever_ignited = False
        self.t_ext = None
        self.fuel_injected = 0.0

    def update(self, t, dt, T, Yf, Yo, rho, grid, params):
        q_chem = heat_release(T, Yf, Yo, rho, params)
        total_chem = float(np.sum(q_chem * grid.volume))
        self.peak_heat_release = max(self.peak_heat_release, total_chem)
        self.fuel_injected += params.mdot * params.YF_inj * dt

        if np.max(T) >= IGNITED_T:
            self.ever_ignited = True

        if (
            self.t_ext is None
            and self.ever_ignited
            and self.peak_heat_release > 0.0
            and total_chem < EXTINCT_FRACTION * self.peak_heat_release
        ):
            self.t_ext = t

    def completeness(self, Yf, grid):
        if self.fuel_injected <= 0.0:
            return 0.0
        remaining = float(np.sum(Yf * grid.volume))
        return float(np.clip(1.0 - remaining / self.fuel_injected, 0.0, 1.0))

    def regime(self, chi_rad):
        if not self.ever_ignited:
            return "no_ignition"
        if self.t_ext is None:
            return "steady"
        if chi_rad > 0.3:
            return "radiative_extinction"
        return "kinetic_extinction"
