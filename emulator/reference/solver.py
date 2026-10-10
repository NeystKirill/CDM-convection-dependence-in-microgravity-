import numpy as np

from .grid import Grid
from .gas import mean_molar_mass, density, conductivity, species_diffusivity
from .velocity import face_mass_flow, cell_velocity
from .diffusion import flux_divergence
from .advection import upwind_term
from .reaction import react_step
from .ignition import apply_ignition
from .qoi import QoiTracker, heat_release, radiative_loss_field, flame_radius, radiative_fraction


class Solver:
    def __init__(self, params):
        self.params = params
        self.grid = Grid(params.Rb, params.Rout, params.n, params.stretch)
        self.t = 0.0
        self.n_steps = 0

        self.T = np.full(params.n, params.T_inf)
        self.Yf = np.zeros(params.n)
        self.Yo = np.full(params.n, params.Yo_inf)

        self.rho_prev = self._density()
        self.qoi_tracker = QoiTracker()

    def _density(self):
        p = self.params
        W = mean_molar_mass(self.Yf, self.Yo, p.Mf, p.Mo, p.Mp)
        return density(p.p, self.T, W)

    def _transport_coeffs(self, rho):
        p = self.params
        lam = conductivity(self.T, p.lam_ref, exponent=p.lam_exponent)
        Df = species_diffusivity(lam, rho, p.cp, p.Le_fuel)
        Do = species_diffusivity(lam, rho, p.cp, p.Le_o2)
        return lam, Df, Do

    def _velocity(self, rho):
        p = self.params
        drho_dt = (rho - self.rho_prev) / self.dt if hasattr(self, "dt") else np.zeros_like(rho)
        mdot_face = face_mass_flow(self.grid, p.mdot, drho_dt)
        return cell_velocity(self.grid, mdot_face, rho)

    def _adaptive_dt(self, rho, lam, Df, Do, u):
        p = self.params
        dr = np.diff(self.grid.r_face)
        alpha = lam / (rho * p.cp)
        diff_rate = np.maximum.reduce([alpha, Df, Do]) / dr**2
        dt_diff = p.fo / np.max(diff_rate)
        u_pos = np.maximum(np.abs(u), 1e-12)
        dt_adv = p.cfl * np.min(dr / u_pos)
        return min(dt_diff, dt_adv, p.t_end - self.t)

    def step(self):
        p, g = self.params, self.grid

        rho = self._density()
        lam, Df, Do = self._transport_coeffs(rho)
        u = self._velocity(rho)
        self.dt = self._adaptive_dt(rho, lam, Df, Do, u)
        dt = self.dt

        self.T, self.Yf, self.Yo = react_step(self.T, self.Yf, self.Yo, p, dt / 2)

        alpha = lam / (rho * p.cp)
        dT = upwind_term(g, self.T, u, p.Tb) + flux_divergence(
            g, self.T, alpha, p.Tb, alpha[0], p.T_inf, alpha[-1]
        )
        qrad = radiative_loss_field(self.T, self.Yf, self.Yo, p)
        dT -= qrad / (rho * p.cp)

        dYf = upwind_term(g, self.Yf, u, p.YF_inj) + flux_divergence(
            g, self.Yf, Df, p.YF_inj, Df[0], 0.0, Df[-1]
        )
        dYo = upwind_term(g, self.Yo, u, 0.0) + flux_divergence(
            g, self.Yo, Do, 0.0, Do[0], p.Yo_inf, Do[-1]
        )

        self.T = self.T + dt * dT
        self.Yf = np.clip(self.Yf + dt * dYf, 0.0, 1.0)
        self.Yo = np.clip(self.Yo + dt * dYo, 0.0, 1.0)

        self.T = apply_ignition(self.T, g, p, self.t)

        self.T, self.Yf, self.Yo = react_step(self.T, self.Yf, self.Yo, p, dt / 2)

        self.qoi_tracker.update(self.t + dt, dt, self.T, self.Yf, self.Yo, rho, g, p)
        self.rho_prev = rho
        self.t += dt
        self.n_steps += 1
        return dt

    def run(self, t_end=None, max_steps=None, on_step=None):
        t_end = self.params.t_end if t_end is None else t_end
        while self.t < t_end:
            if max_steps is not None and self.n_steps >= max_steps:
                break
            self.step()
            if on_step is not None:
                on_step(self)
        return self

    def state(self):
        return {
            "r": self.grid.r_c.copy(),
            "T": self.T.copy(),
            "Yf": self.Yf.copy(),
            "Yo": self.Yo.copy(),
            "rho": self._density(),
        }

    def qoi(self):
        rho = self._density()
        chi_rad = radiative_fraction(self.T, self.Yf, self.Yo, rho, self.grid, self.params)
        return {
            "t": self.t,
            "r_f": flame_radius(self.T, self.Yf, self.Yo, rho, self.grid, self.params),
            "T_max": float(np.max(self.T)),
            "chi_rad": chi_rad,
            "eta": self.qoi_tracker.completeness(self.Yf, self.grid),
            "t_ext": self.qoi_tracker.t_ext,
            "regime": self.qoi_tracker.regime(chi_rad),
        }
