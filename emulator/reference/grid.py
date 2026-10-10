import numpy as np


class Grid:
    def __init__(self, Rb, Rout, n, stretch=1.0):
        x = np.linspace(0.0, 1.0, n + 1)
        self.r_face = Rb * (Rout / Rb) ** (x**stretch)
        self.r_c = 0.5 * (self.r_face[:-1] + self.r_face[1:])
        self.volume = (4.0 / 3.0) * np.pi * (self.r_face[1:] ** 3 - self.r_face[:-1] ** 3)
        self.area = 4.0 * np.pi * self.r_face**2
        self.n = n
