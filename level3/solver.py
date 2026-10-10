import numpy as np

from emulator.reference.gas import mean_molar_mass, density, conductivity, species_diffusivity
from emulator.reference.qoi import radiative_loss_linearized
from .grid import Grid
from .momentum import PressureSolver, momentum_step
from .diffusion import diffuse3d
from .advection import upwind_term
from .reaction import react_step
from .ignition import apply_ignition
from .qoi import QoiTracker, radiative_fraction, damkohler_number, flame_location
from .params import G0


class Solver:
    def __init__(self, params):
        self.params = params
        self.grid = Grid(params.Lx, params.Ly, params.Lz, params.nx, params.ny, params.nz)
        self.pressure_solver = PressureSolver(self.grid)

        radial = (self.grid.X_c[0] - params.wick_x0) ** 2 + (self.grid.Y_c[0] - params.wick_y0) ** 2
        self.wick_mask = radial <= params.wick_r ** 2

        self.t = 0.0
        self.n_steps = 0
        self.dt = None
        self.div_max = 0.0

        shape = (params.nz, params.ny, params.nx)
        self.T = np.full(shape, params.T_inf)
        self.Yf = np.zeros(shape)
        self.Yo = np.full(shape, params.Yo_inf)
        self.u_x = np.zeros(shape)
        self.u_y = np.zeros(shape)
        self.u_z = np.full(shape, params.U_ext)

        W_inf = mean_molar_mass(0.0, params.Yo_inf, params.Mf, params.Mo, params.Mp)
        self.rho0 = density(params.p, params.T_inf, W_inf)
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
        alpha = lam / (rho * p.cp)
        nu = p.Pr * alpha
        return lam, Df, Do, alpha, nu

    def _adaptive_dt(self):
        p = self.params
        d_min = min(self.grid.dx, self.grid.dy, self.grid.dz)

        u_mag = max(
            float(np.max(np.abs(self.u_x))), float(np.max(np.abs(self.u_y))),
            float(np.max(np.abs(self.u_z))), 1e-6,
        )
        dt_adv = p.cfl * d_min / u_mag

        a_buoy = max(p.g_frac * G0, 1e-6)
        dt_buoy = p.cfl * np.sqrt(2.0 * self.grid.dz / a_buoy)

        alpha_ref = p.lam_ref / (self.rho0 * p.cp)
        dt_diff = p.cfl * d_min ** 2 / alpha_ref

        return min(dt_adv, dt_buoy, dt_diff, p.t_end - self.t)

    def step(self):
        p, g = self.params, self.grid

        if self.t >= p.t_end:
            return 0.0

        rho = self._density()
        lam, Df, Do, alpha, nu = self._transport_coeffs(rho)
        self.dt = self._adaptive_dt()
        dt = self.dt

        self.T, self.Yf, self.Yo = react_step(self.T, self.Yf, self.Yo, p, dt / 2)

        self.u_x, self.u_y, self.u_z, self.div_max = momentum_step(
            g, self.pressure_solver, self.u_x, self.u_y, self.u_z, rho, self.rho0, self.rho0,
            nu, p.g_frac, G0, p.U_ext, dt,
        )

        q_rad, rad_slope = radiative_loss_linearized(self.T, self.Yf, self.Yo, p)
        rad_const = q_rad - rad_slope * self.T
        diag_extra = dt * rad_slope / (rho * p.cp)
        rhs_extra = -dt * rad_const / (rho * p.cp)

        T_adv = self.T + dt * upwind_term(g, self.T, self.u_x, self.u_y, self.u_z)
        Yf_adv = self.Yf + dt * upwind_term(g, self.Yf, self.u_x, self.u_y, self.u_z)
        Yo_adv = self.Yo + dt * upwind_term(g, self.Yo, self.u_x, self.u_y, self.u_z)

        self.T = diffuse3d(
            g, T_adv, alpha, dt,
            x_lo_val=p.T_inf, x_hi_val=p.T_inf,
            y_lo_val=p.T_inf, y_hi_val=p.T_inf,
            z_lo_val=p.Ts, z_lo_active=self.wick_mask,
            z_hi_val=p.T_inf, z_hi_active=True,
            diag_extra=diag_extra, rhs_extra=rhs_extra,
        )
        self.Yf = np.clip(diffuse3d(
            g, Yf_adv, Df, dt,
            x_lo_val=0.0, x_hi_val=0.0,
            y_lo_val=0.0, y_hi_val=0.0,
            z_lo_val=p.Yf_s, z_lo_active=self.wick_mask,
            z_hi_val=0.0, z_hi_active=True,
        ), 0.0, 1.0)
        self.Yo = np.clip(diffuse3d(
            g, Yo_adv, Do, dt,
            x_lo_val=p.Yo_inf, x_hi_val=p.Yo_inf,
            y_lo_val=p.Yo_inf, y_hi_val=p.Yo_inf,
            z_lo_val=0.0, z_lo_active=self.wick_mask,
            z_hi_val=p.Yo_inf, z_hi_active=True,
        ), 0.0, 1.0)

        self.T = apply_ignition(self.T, g, p, self.t)

        self.T, self.Yf, self.Yo = react_step(self.T, self.Yf, self.Yo, p, dt / 2)

        self.qoi_tracker.update(self.t + dt, self.T, self.Yf, self.Yo, rho, g, p)
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
            "x": self.grid.x_c.copy(),
            "y": self.grid.y_c.copy(),
            "z": self.grid.z_c.copy(),
            "T": self.T.copy(),
            "Yf": self.Yf.copy(),
            "Yo": self.Yo.copy(),
            "rho": self._density(),
            "u_x": self.u_x.copy(),
            "u_y": self.u_y.copy(),
            "u_z": self.u_z.copy(),
        }

    def qoi(self):
        rho = self._density()
        chi_rad = radiative_fraction(self.T, self.Yf, self.Yo, rho, self.grid, self.params)
        da = damkohler_number(self.T, self.Yf, self.Yo, rho, self.u_x, self.u_y, self.u_z, self.grid, self.params)
        k, j, i = flame_location(self.T, self.Yf, self.Yo, rho, self.grid, self.params)
        return {
            "t": self.t,
            "T_max": float(np.max(self.T)),
            "x_f": float(self.grid.x_c[i]),
            "y_f": float(self.grid.y_c[j]),
            "z_f": float(self.grid.z_c[k]),
            "chi_rad": chi_rad,
            "da": da,
            "eta": self.qoi_tracker.completeness(self.Yf, self.grid),
            "t_ext": self.qoi_tracker.t_ext,
            "regime": self.qoi_tracker.regime(chi_rad, da),
        }
