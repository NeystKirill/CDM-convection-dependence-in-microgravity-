import numpy as np

from emulator.common.linalg import implicit_diffuse_line


def diffuse3d(
    grid, phi, gamma, dt,
    x_lo_val=0.0, x_hi_val=0.0,
    y_lo_val=0.0, y_hi_val=0.0,
    z_lo_val=0.0, z_lo_active=True,
    z_hi_val=0.0, z_hi_active=True,
    diag_extra=None, rhs_extra=None,
):
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z_lo_val = np.broadcast_to(z_lo_val, (ny, nx))
    z_lo_active = np.broadcast_to(z_lo_active, (ny, nx))
    z_hi_val = np.broadcast_to(z_hi_val, (ny, nx))
    z_hi_active = np.broadcast_to(z_hi_active, (ny, nx))

    face_area_x = np.full(nx + 1, grid.x_face_area)
    face_area_y = np.full(ny + 1, grid.y_face_area)
    face_area_z = np.full(nz + 1, grid.z_face_area)
    vol_x = np.full(nx, grid.volume)
    vol_y = np.full(ny, grid.volume)
    vol_z = np.full(nz, grid.volume)

    after_x = np.empty_like(phi)
    for k in range(nz):
        for j in range(ny):
            after_x[k, j, :] = implicit_diffuse_line(
                grid.x_face, grid.x_c, face_area_x, vol_x, gamma[k, j, :],
                phi[k, j, :], x_lo_val, gamma[k, j, 0], x_hi_val, gamma[k, j, -1], dt,
            )

    after_y = np.empty_like(after_x)
    for k in range(nz):
        for i in range(nx):
            after_y[k, :, i] = implicit_diffuse_line(
                grid.y_face, grid.y_c, face_area_y, vol_y, gamma[k, :, i],
                after_x[k, :, i], y_lo_val, gamma[k, 0, i], y_hi_val, gamma[k, -1, i], dt,
            )

    after_z = np.empty_like(after_y)
    for j in range(ny):
        for i in range(nx):
            gamma_lo = gamma[0, j, i] if z_lo_active[j, i] else 0.0
            gamma_hi = gamma[-1, j, i] if z_hi_active[j, i] else 0.0
            d_extra = None if diag_extra is None else diag_extra[:, j, i]
            r_extra = None if rhs_extra is None else rhs_extra[:, j, i]
            after_z[:, j, i] = implicit_diffuse_line(
                grid.z_face, grid.z_c, face_area_z, vol_z, gamma[:, j, i],
                after_y[:, j, i], z_lo_val[j, i], gamma_lo, z_hi_val[j, i], gamma_hi, dt,
                diag_extra=d_extra, rhs_extra=r_extra,
            )

    return after_z
