import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from emulator.common.linalg import line_diffusion_coeffs
from .advection import upwind_term
from .diffusion import diffuse2d


class PressureSolver:
    def __init__(self, grid):
        self.grid = grid
        nz, nr = grid.nz, grid.nr
        ones_r = np.ones(nr)
        ones_z = np.ones(nz)

        c_r_left = np.empty((nz, nr))
        c_r_right = np.empty((nz, nr))
        for j in range(nz):
            cl, cr = line_diffusion_coeffs(
                grid.r_face, grid.r_c, grid.r_face_area[j, :], grid.volume[j, :], ones_r, 0.0, 0.0
            )
            c_r_left[j, :] = cl
            c_r_right[j, :] = cr

        c_z_left = np.empty((nz, nr))
        c_z_right = np.empty((nz, nr))
        for i in range(nr):
            cl, cr = line_diffusion_coeffs(
                grid.z_face, grid.z_c, grid.z_face_area[:, i], grid.volume[:, i], ones_z, 0.0, 1.0
            )
            c_z_left[:, i] = cl
            c_z_right[:, i] = cr

        n = nz * nr
        rows, cols, vals = [], [], []
        for j in range(nz):
            for i in range(nr):
                k = j * nr + i
                diag_val = c_r_left[j, i] + c_r_right[j, i] + c_z_left[j, i] + c_z_right[j, i]
                rows.append(k)
                cols.append(k)
                vals.append(diag_val)
                if i > 0:
                    rows.append(k)
                    cols.append(k - 1)
                    vals.append(-c_r_left[j, i])
                if i < nr - 1:
                    rows.append(k)
                    cols.append(k + 1)
                    vals.append(-c_r_right[j, i])
                if j > 0:
                    rows.append(k)
                    cols.append(k - nr)
                    vals.append(-c_z_left[j, i])
                if j < nz - 1:
                    rows.append(k)
                    cols.append(k + nr)
                    vals.append(-c_z_right[j, i])

        A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self._lu = spla.splu(A)

    def solve(self, source):
        phi = self._lu.solve(source.ravel())
        return phi.reshape(self.grid.nz, self.grid.nr)


def face_velocities(grid, u_r, u_z, bottom_uz):
    nz, nr = grid.nz, grid.nr

    ur_face = np.zeros((nz, nr + 1))
    ur_face[:, 1:-1] = 0.5 * (u_r[:, :-1] + u_r[:, 1:])

    uz_face = np.zeros((nz + 1, nr))
    uz_face[1:-1, :] = 0.5 * (u_z[:-1, :] + u_z[1:, :])
    uz_face[0, :] = bottom_uz
    uz_face[-1, :] = u_z[-1, :]

    return ur_face, uz_face


def divergence_from_faces(grid, ur_face, uz_face):
    flux_r = grid.r_face_area * ur_face
    flux_z = grid.z_face_area * uz_face
    return (flux_r[:, 1:] - flux_r[:, :-1] + flux_z[1:, :] - flux_z[:-1, :]) / grid.volume


def divergence(grid, u_r, u_z, bottom_uz):
    ur_face, uz_face = face_velocities(grid, u_r, u_z, bottom_uz)
    return divergence_from_faces(grid, ur_face, uz_face)


def pressure_face_gradient(grid, phi):
    nz, nr = grid.nz, grid.nr

    gr_face = np.zeros((nz, nr + 1))
    gr_face[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / (grid.r_c[None, 1:] - grid.r_c[None, :-1])

    gz_face = np.zeros((nz + 1, nr))
    gz_face[1:-1, :] = (phi[1:, :] - phi[:-1, :]) / (grid.z_c[1:, None] - grid.z_c[:-1, None])
    gz_face[-1, :] = (0.0 - phi[-1, :]) / (grid.z_face[-1] - grid.z_c[-1])

    return gr_face, gz_face


def face_to_center(ur_face, uz_face):
    u_r = 0.5 * (ur_face[:, :-1] + ur_face[:, 1:])
    u_z = 0.5 * (uz_face[:-1, :] + uz_face[1:, :])
    return u_r, u_z


def momentum_step(grid, pressure_solver, u_r, u_z, rho, rho_inf, rho0, nu, g_frac, g0, U_ext, dt):
    adv_r = upwind_term(grid, u_r, u_r, u_z)
    adv_z = upwind_term(grid, u_z, u_r, u_z)
    buoy_z = g_frac * g0 * (rho_inf - rho) / rho0

    u_r_star = diffuse2d(
        grid, u_r + dt * adv_r, nu, dt,
        r_hi_val=0.0, r_hi_active=True,
        z_lo_val=0.0, z_lo_active=True,
        z_hi_val=0.0, z_hi_active=False,
    )
    u_z_star = diffuse2d(
        grid, u_z + dt * (adv_z + buoy_z), nu, dt,
        r_hi_val=0.0, r_hi_active=True,
        z_lo_val=U_ext, z_lo_active=True,
        z_hi_val=0.0, z_hi_active=False,
    )

    ur_face, uz_face = face_velocities(grid, u_r_star, u_z_star, U_ext)
    div_star = divergence_from_faces(grid, ur_face, uz_face)
    phi = pressure_solver.solve(-(rho0 / dt) * div_star)
    gr_face, gz_face = pressure_face_gradient(grid, phi)

    ur_face_new = ur_face - dt / rho0 * gr_face
    uz_face_new = uz_face - dt / rho0 * gz_face

    div_new = divergence_from_faces(grid, ur_face_new, uz_face_new)
    u_r_new, u_z_new = face_to_center(ur_face_new, uz_face_new)
    return u_r_new, u_z_new, float(np.max(np.abs(div_new)))
