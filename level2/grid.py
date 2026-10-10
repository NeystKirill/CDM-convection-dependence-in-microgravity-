import numpy as np


class Grid:
    def __init__(self, Lr, Lz, nr, nz):
        self.nr = nr
        self.nz = nz

        self.r_face = np.linspace(0.0, Lr, nr + 1)
        self.z_face = np.linspace(0.0, Lz, nz + 1)
        self.r_c = 0.5 * (self.r_face[:-1] + self.r_face[1:])
        self.z_c = 0.5 * (self.z_face[:-1] + self.z_face[1:])
        self.dr = np.diff(self.r_face)
        self.dz = np.diff(self.z_face)

        ring_area = np.pi * (self.r_face[1:] ** 2 - self.r_face[:-1] ** 2)
        self.volume = ring_area[None, :] * self.dz[:, None]
        self.r_face_area = 2.0 * np.pi * self.r_face[None, :] * self.dz[:, None]
        self.z_face_area = np.broadcast_to(ring_area[None, :], (nz + 1, nr)).copy()

        self.R_c, self.Z_c = np.meshgrid(self.r_c, self.z_c)
