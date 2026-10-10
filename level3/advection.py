from emulator.common.upwind import upwind_gradient


def upwind_term(grid, phi, u_x, u_y, u_z):
    grad_z = upwind_gradient(phi, grid.z_c, u_z, axis=0)
    grad_y = upwind_gradient(phi, grid.y_c, u_y, axis=1)
    grad_x = upwind_gradient(phi, grid.x_c, u_x, axis=2)
    return -(u_x * grad_x + u_y * grad_y + u_z * grad_z)
