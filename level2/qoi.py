import numpy as np

from emulator.reference.reaction import reaction_rate
from emulator.reference.qoi import absorption_coeff
from formulas import radiative_loss, damkohler, reaction_time_scale

IGNITED_T = 1000.0
EXTINCT_FRACTION = 0.01
KINETIC_DA_THRESHOLD = 1.0


def heat_release(T, Yf, Yo, rho, params):
    omega = reaction_rate(T, Yf, Yo, params)
    return rho * params.cp * params.Qcp * omega


def radiative_loss_field(T, Yf, Yo, params):
    if not params.rad_enabled:
        return np.zeros_like(T)
    a_p = absorption_coeff(Yf, Yo, params)
    return radiative_loss(T, params.T_inf, a_p)


def flame_location(T, Yf, Yo, rho, grid, params):
    q = heat_release(T, Yf, Yo, rho, params)
    j, i = np.unravel_index(np.argmax(q), q.shape)
    return j, i


def radiative_fraction(T, Yf, Yo, rho, grid, params):
    q_chem = heat_release(T, Yf, Yo, rho, params)
    q_rad = radiative_loss_field(T, Yf, Yo, params)
    total_chem = float(np.sum(q_chem * grid.volume))
    total_rad = float(np.sum(q_rad * grid.volume))
    if total_chem <= 0.0:
        return 0.0
    return total_rad / total_chem


def damkohler_number(T, Yf, Yo, rho, u_r, u_z, grid, params):
    j, i = flame_location(T, Yf, Yo, rho, grid, params)
    u_char = max(float(np.max(np.abs(u_z))), float(np.max(np.abs(u_r))), 1e-3)
    flow_time = params.wick_r / u_char
    reaction_time = float(reaction_time_scale(params.B, params.Yf_s, params.Yo_inf, params.Ta, T[j, i]))
    return float(damkohler(flow_time, reaction_time))


class QoiTracker:
    def __init__(self):
        self.peak_heat_release = 0.0
        self.peak_fuel_mass = 0.0
        self.ever_ignited = False
        self.t_ext = None

    def update(self, t, T, Yf, Yo, rho, grid, params):
        q_chem = heat_release(T, Yf, Yo, rho, params)
        total_chem = float(np.sum(q_chem * grid.volume))
        self.peak_heat_release = max(self.peak_heat_release, total_chem)

        total_fuel = float(np.sum(Yf * grid.volume))
        self.peak_fuel_mass = max(self.peak_fuel_mass, total_fuel)

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
        if self.peak_fuel_mass <= 0.0:
            return 0.0
        remaining = float(np.sum(Yf * grid.volume))
        return float(np.clip(1.0 - remaining / self.peak_fuel_mass, 0.0, 1.0))

    def regime(self, chi_rad, da):
        if not self.ever_ignited:
            return "no_ignition"
        if self.t_ext is None:
            return "steady"
        if da < KINETIC_DA_THRESHOLD:
            return "kinetic_extinction"
        return "radiative_extinction"
