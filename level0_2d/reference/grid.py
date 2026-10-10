import copy

import numpy as np

GHOST = 2
T, F, O = 0, 1, 2


class Grid:
    def __init__(self, params):
        self.nx = params.nx
        self.ny = params.ny
        self.dx = params.dx
        self.dy = params.dy
        self.ng = GHOST

        self.q = np.zeros((3, self.ny + 2 * self.ng, self.nx + 2 * self.ng))

        i = np.arange(-self.ng, self.nx + self.ng)
        j = np.arange(-self.ng, self.ny + self.ng)
        self.x = (i + 0.5) * self.dx
        self.y = (j + 0.5) * self.dy

    @property
    def interior(self):
        ng = self.ng
        return slice(ng, ng + self.ny), slice(ng, ng + self.nx)

    def interior_field(self, k):
        j, i = self.interior
        return self.q[k, j, i]

    def with_q(self, q):
        g = copy.copy(self)
        g.q = q
        return g
