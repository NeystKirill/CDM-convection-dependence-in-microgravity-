from emulator.common.upwind import upwind_gradient


def upwind_term(grid, phi, u_r, u_z):
    grad_z = upwind_gradient(phi, grid.z_c, u_z, axis=0)
    grad_r = upwind_gradient(phi, grid.r_c, u_r, axis=1)
    return -(u_r * grad_r + u_z * grad_z)
