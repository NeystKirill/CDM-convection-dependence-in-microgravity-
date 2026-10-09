def thermal_diffusivity(lam, rho, cp):
    return lam / (rho * cp)


def mass_diffusivity(alpha, Le=1.0):
    return alpha / Le
