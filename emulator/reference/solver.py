import numpy as np

from .grid import Grid
from .gas import mean_molar_mass, density, conductivity, species_diffusivity
from .velocity import face_mass_flow, cell_velocity
from .diffusion import implicit_diffuse
from .advection import upwind_term
from .reaction import react_step
from .ignition import apply_ignition
from .qoi import QoiTracker, radiative_loss_linearized, flame_radius, radiative_fraction


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

    def _adaptive_dt(self, u):
        p = self.params
        dr = np.diff(self.grid.r_face)
        u_pos = np.maximum(np.abs(u), 1e-12)
        dt_adv = p.cfl * np.min(dr / u_pos)
        return min(dt_adv, p.t_end - self.t)

    def step(self):
        p, g = self.params, self.grid

        if self.t >= p.t_end:
            return 0.0

        rho = self._density()
        lam, Df, Do = self._transport_coeffs(rho)
        u = self._velocity(rho)
        self.dt = self._adaptive_dt(u)
        dt = self.dt

        self.T, self.Yf, self.Yo = react_step(self.T, self.Yf, self.Yo, p, dt / 2)

        alpha = lam / (rho * p.cp)
        T_adv = self.T + dt * upwind_term(g, self.T, u, p.Tb)
        Yf_adv = self.Yf + dt * upwind_term(g, self.Yf, u, p.YF_inj)
        Yo_adv = self.Yo + dt * upwind_term(g, self.Yo, u, 0.0)

        q_rad, rad_slope = radiative_loss_linearized(self.T, self.Yf, self.Yo, p)
        rad_const = q_rad - rad_slope * self.T
        diag_extra = dt * rad_slope / (rho * p.cp)
        rhs_extra = -dt * rad_const / (rho * p.cp)

        self.T = implicit_diffuse(g, T_adv, alpha, p.Tb, alpha[0], p.T_inf, alpha[-1], dt, diag_extra, rhs_extra)
        self.Yf = np.clip(implicit_diffuse(g, Yf_adv, Df, p.YF_inj, Df[0], 0.0, Df[-1], dt), 0.0, 1.0)
        self.Yo = np.clip(implicit_diffuse(g, Yo_adv, Do, 0.0, Do[0], p.Yo_inf, Do[-1], dt), 0.0, 1.0)

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
