import numpy as np


class Grid:
    def __init__(self, Lx, Ly, Lz, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.x_face = np.linspace(0.0, Lx, nx + 1)
        self.y_face = np.linspace(0.0, Ly, ny + 1)
        self.z_face = np.linspace(0.0, Lz, nz + 1)
        self.x_c = 0.5 * (self.x_face[:-1] + self.x_face[1:])
        self.y_c = 0.5 * (self.y_face[:-1] + self.y_face[1:])
        self.z_c = 0.5 * (self.z_face[:-1] + self.z_face[1:])

        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz

        self.volume = self.dx * self.dy * self.dz
        self.x_face_area = self.dy * self.dz
        self.y_face_area = self.dx * self.dz
        self.z_face_area = self.dx * self.dy

        shape = (nz, ny, nx)
        self.X_c = np.broadcast_to(self.x_c[None, None, :], shape)
        self.Y_c = np.broadcast_to(self.y_c[None, :, None], shape)
        self.Z_c = np.broadcast_to(self.z_c[:, None, None], shape)
