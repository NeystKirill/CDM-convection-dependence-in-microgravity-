import numpy as np

from formulas import (
    minmod,
    reaction_rate,
    radiative_loss,
    thermal_diffusivity,
    mass_diffusivity,
    peclet,
    damkohler,
    zeldovich,
)


def test_minmod_same_sign_picks_smaller_magnitude():
    assert minmod(np.array([2.0]), np.array([3.0]))[0] == 2.0
    assert minmod(np.array([-2.0]), np.array([-5.0]))[0] == -2.0


def test_minmod_opposite_sign_is_zero():
    assert minmod(np.array([2.0]), np.array([-1.0]))[0] == 0.0
    assert minmod(np.array([0.0]), np.array([5.0]))[0] == 0.0


def test_reaction_rate_zero_without_reactants():
    assert reaction_rate(1000.0, 0.0, 0.2, B=1e7, Ta=15000.0) == 0.0
    assert reaction_rate(1000.0, 0.2, 0.0, B=1e7, Ta=15000.0) == 0.0


def test_reaction_rate_increases_with_temperature():
    low = reaction_rate(1000.0, 0.5, 0.2, B=1e7, Ta=15000.0)
    high = reaction_rate(2000.0, 0.5, 0.2, B=1e7, Ta=15000.0)
    assert high > low


def test_radiative_loss_zero_at_ambient():
    assert radiative_loss(300.0, 300.0, a_p=1.0) == 0.0


def test_radiative_loss_positive_when_hot():
    assert radiative_loss(2000.0, 300.0, a_p=1.0) > 0.0


def test_thermal_and_mass_diffusivity():
    alpha = thermal_diffusivity(lam=0.07, rho=0.3, cp=1200.0)
    assert np.isclose(alpha, 0.07 / (0.3 * 1200.0))
    assert np.isclose(mass_diffusivity(alpha, Le=1.0), alpha)


def test_dimensionless_numbers():
    alpha = 2e-4
    pe = peclet(U=0.2, L=0.1, alpha=alpha)
    assert np.isclose(pe, 0.2 * 0.1 / alpha)

    da = damkohler(flow_time=0.5, reaction_time=0.1)
    assert np.isclose(da, 5.0)

    ze = zeldovich(Ta=15000.0, Tf=2000.0, T_inf=300.0)
    assert np.isclose(ze, 15000.0 * (2000.0 - 300.0) / 2000.0**2)
