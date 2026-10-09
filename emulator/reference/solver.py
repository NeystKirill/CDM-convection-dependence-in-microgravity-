from .grid import Grid, T as IDX_T, F as IDX_F, O as IDX_O
from .boundary import apply_bc
from .rhs import rhs
from .reaction import react_step
from .ignition import apply_ignition
from .diagnostics import diagnostics


class Solver:
    def __init__(self, params):
        self.params = params
        self.grid = Grid(params)
        self.t = 0.0
        self.n_steps = 0

        j, i = self.grid.interior
        self.grid.q[IDX_T, j, i] = params.T_inf
        self.grid.q[IDX_F, j, i] = 0.0
        self.grid.q[IDX_O, j, i] = params.Yo_inf
        apply_bc(self.grid, params)

    def step(self):
        g, p, dt = self.grid, self.params, self.params.dt
        j, i = g.interior

        react_step(g, p, dt / 2)
        apply_bc(g, p)

        k1 = rhs(g, p)
        q1 = g.q.copy()
        q1[:, j, i] = g.q[:, j, i] + dt * k1
        g1 = g.with_q(q1)
        apply_bc(g1, p)

        k2 = rhs(g1, p)
        qnew = g.q.copy()
        qnew[:, j, i] = 0.5 * g.q[:, j, i] + 0.5 * (q1[:, j, i] + dt * k2)
        g.q = qnew
        apply_bc(g, p)

        apply_ignition(g, p, self.t)

        react_step(g, p, dt / 2)
        apply_bc(g, p)

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

    def fields(self):
        j, i = self.grid.interior
        return {
            "T": self.grid.q[IDX_T, j, i].copy(),
            "Yf": self.grid.q[IDX_F, j, i].copy(),
            "Yo": self.grid.q[IDX_O, j, i].copy(),
        }

    def diagnostics(self):
        return diagnostics(self.grid, self.params)
