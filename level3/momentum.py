import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from emulator.common.linalg import line_diffusion_coeffs
from .advection import upwind_term
from .diffusion import diffuse3d


class PressureSolver:
    def __init__(self, grid):
        self.grid = grid
        nz, ny, nx = grid.nz, grid.ny, grid.nx

        face_area_x = np.full(nx + 1, grid.x_face_area)
        face_area_y = np.full(ny + 1, grid.y_face_area)
        face_area_z = np.full(nz + 1, grid.z_face_area)
        vol_x = np.full(nx, grid.volume)
        vol_y = np.full(ny, grid.volume)
        vol_z = np.full(nz, grid.volume)
        ones_x, ones_y, ones_z = np.ones(nx), np.ones(ny), np.ones(nz)

        cx_left = np.empty((nz, ny, nx))
        cx_right = np.empty((nz, ny, nx))
        for k in range(nz):
            for j in range(ny):
                cl, cr = line_diffusion_coeffs(grid.x_face, grid.x_c, face_area_x, vol_x, ones_x, 0.0, 0.0)
                cx_left[k, j, :] = cl
                cx_right[k, j, :] = cr

        cy_left = np.empty((nz, ny, nx))
        cy_right = np.empty((nz, ny, nx))
        for k in range(nz):
            for i in range(nx):
                cl, cr = line_diffusion_coeffs(grid.y_face, grid.y_c, face_area_y, vol_y, ones_y, 0.0, 0.0)
                cy_left[k, :, i] = cl
                cy_right[k, :, i] = cr

        cz_left = np.empty((nz, ny, nx))
        cz_right = np.empty((nz, ny, nx))
        for j in range(ny):
            for i in range(nx):
                cl, cr = line_diffusion_coeffs(grid.z_face, grid.z_c, face_area_z, vol_z, ones_z, 0.0, 1.0)
                cz_left[:, j, i] = cl
                cz_right[:, j, i] = cr

        n = nz * ny * nx

        def idx(k, j, i):
            return (k * ny + j) * nx + i

        rows, cols, vals = [], [], []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    p = idx(k, j, i)
                    diag_val = (
                        cx_left[k, j, i] + cx_right[k, j, i]
                        + cy_left[k, j, i] + cy_right[k, j, i]
                        + cz_left[k, j, i] + cz_right[k, j, i]
                    )
                    rows.append(p)
                    cols.append(p)
                    vals.append(diag_val)
                    if i > 0:
                        rows.append(p); cols.append(idx(k, j, i - 1)); vals.append(-cx_left[k, j, i])
                    if i < nx - 1:
                        rows.append(p); cols.append(idx(k, j, i + 1)); vals.append(-cx_right[k, j, i])
                    if j > 0:
                        rows.append(p); cols.append(idx(k, j - 1, i)); vals.append(-cy_left[k, j, i])
                    if j < ny - 1:
                        rows.append(p); cols.append(idx(k, j + 1, i)); vals.append(-cy_right[k, j, i])
                    if k > 0:
                        rows.append(p); cols.append(idx(k - 1, j, i)); vals.append(-cz_left[k, j, i])
                    if k < nz - 1:
                        rows.append(p); cols.append(idx(k + 1, j, i)); vals.append(-cz_right[k, j, i])

        A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self._lu = spla.splu(A)

    def solve(self, source):
        phi = self._lu.solve(source.ravel())
        return phi.reshape(self.grid.nz, self.grid.ny, self.grid.nx)


def face_velocities(grid, u_x, u_y, u_z, bottom_uz):
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    ux_face = np.zeros((nz, ny, nx + 1))
    ux_face[:, :, 1:-1] = 0.5 * (u_x[:, :, :-1] + u_x[:, :, 1:])

    uy_face = np.zeros((nz, ny + 1, nx))
    uy_face[:, 1:-1, :] = 0.5 * (u_y[:, :-1, :] + u_y[:, 1:, :])

    uz_face = np.zeros((nz + 1, ny, nx))
    uz_face[1:-1, :, :] = 0.5 * (u_z[:-1, :, :] + u_z[1:, :, :])
    uz_face[0, :, :] = bottom_uz
    uz_face[-1, :, :] = u_z[-1, :, :]

    return ux_face, uy_face, uz_face


def divergence_from_faces(grid, ux_face, uy_face, uz_face):
    flux_x = grid.x_face_area * ux_face
    flux_y = grid.y_face_area * uy_face
    flux_z = grid.z_face_area * uz_face
    return (
        flux_x[:, :, 1:] - flux_x[:, :, :-1]
        + flux_y[:, 1:, :] - flux_y[:, :-1, :]
        + flux_z[1:, :, :] - flux_z[:-1, :, :]
    ) / grid.volume


def pressure_face_gradient(grid, phi):
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    gx_face = np.zeros((nz, ny, nx + 1))
    gx_face[:, :, 1:-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / grid.dx

    gy_face = np.zeros((nz, ny + 1, nx))
    gy_face[:, 1:-1, :] = (phi[:, 1:, :] - phi[:, :-1, :]) / grid.dy

    gz_face = np.zeros((nz + 1, ny, nx))
    gz_face[1:-1, :, :] = (phi[1:, :, :] - phi[:-1, :, :]) / grid.dz
    gz_face[-1, :, :] = (0.0 - phi[-1, :, :]) / (0.5 * grid.dz)

    return gx_face, gy_face, gz_face


def face_to_center(ux_face, uy_face, uz_face):
    u_x = 0.5 * (ux_face[:, :, :-1] + ux_face[:, :, 1:])
    u_y = 0.5 * (uy_face[:, :-1, :] + uy_face[:, 1:, :])
    u_z = 0.5 * (uz_face[:-1, :, :] + uz_face[1:, :, :])
    return u_x, u_y, u_z


def momentum_step(grid, pressure_solver, u_x, u_y, u_z, rho, rho_inf, rho0, nu, g_frac, g0, U_ext, dt):
    adv_x = upwind_term(grid, u_x, u_x, u_y, u_z)
    adv_y = upwind_term(grid, u_y, u_x, u_y, u_z)
    adv_z = upwind_term(grid, u_z, u_x, u_y, u_z)
    buoy_z = g_frac * g0 * (rho_inf - rho) / rho0

    u_x_star = diffuse3d(grid, u_x + dt * adv_x, nu, dt, z_hi_active=False)
    u_y_star = diffuse3d(grid, u_y + dt * adv_y, nu, dt, z_hi_active=False)
    u_z_star = diffuse3d(grid, u_z + dt * (adv_z + buoy_z), nu, dt, z_lo_val=U_ext, z_hi_active=False)

    ux_face, uy_face, uz_face = face_velocities(grid, u_x_star, u_y_star, u_z_star, U_ext)
    div_star = divergence_from_faces(grid, ux_face, uy_face, uz_face)
    phi = pressure_solver.solve(-(rho0 / dt) * div_star)
    gx_face, gy_face, gz_face = pressure_face_gradient(grid, phi)

    ux_face_new = ux_face - dt / rho0 * gx_face
    uy_face_new = uy_face - dt / rho0 * gy_face
    uz_face_new = uz_face - dt / rho0 * gz_face

    div_new = divergence_from_faces(grid, ux_face_new, uy_face_new, uz_face_new)
    u_x_new, u_y_new, u_z_new = face_to_center(ux_face_new, uy_face_new, uz_face_new)
    return u_x_new, u_y_new, u_z_new, float(np.max(np.abs(div_new)))
