import numpy as np

from emulator.common.linalg import implicit_diffuse_line


def diffuse2d(
    grid, phi, gamma, dt,
    r_hi_val=0.0, r_hi_active=True,
    z_lo_val=0.0, z_lo_active=True,
    z_hi_val=0.0, z_hi_active=True,
    diag_extra=None, rhs_extra=None,
):
    nz, nr = grid.nz, grid.nr
    z_lo_val = np.broadcast_to(z_lo_val, (nr,))
    z_lo_active = np.broadcast_to(z_lo_active, (nr,))
    z_hi_val = np.broadcast_to(z_hi_val, (nr,))
    z_hi_active = np.broadcast_to(z_hi_active, (nr,))

    r_hi_gamma_mult = 1.0 if r_hi_active else 0.0

    after_r = np.empty_like(phi)
    for j in range(nz):
        after_r[j, :] = implicit_diffuse_line(
            grid.r_face, grid.r_c, grid.r_face_area[j, :], grid.volume[j, :], gamma[j, :],
            phi[j, :], 0.0, 0.0, r_hi_val, gamma[j, -1] * r_hi_gamma_mult, dt,
        )

    after_z = np.empty_like(after_r)
    for i in range(nr):
        gamma_lo = gamma[0, i] if z_lo_active[i] else 0.0
        phi_lo = z_lo_val[i]
        gamma_hi = gamma[-1, i] if z_hi_active[i] else 0.0
        phi_hi = z_hi_val[i]
        col_diag_extra = None if diag_extra is None else diag_extra[:, i]
        col_rhs_extra = None if rhs_extra is None else rhs_extra[:, i]
        after_z[:, i] = implicit_diffuse_line(
            grid.z_face, grid.z_c, grid.z_face_area[:, i], grid.volume[:, i], gamma[:, i],
            after_r[:, i], phi_lo, gamma_lo, phi_hi, gamma_hi, dt,
            diag_extra=col_diag_extra, rhs_extra=col_rhs_extra,
        )

    return after_z
