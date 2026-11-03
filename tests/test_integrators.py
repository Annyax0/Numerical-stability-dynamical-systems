import numpy as np
import pytest
from numerical_dynamics.integrators import euler, rk4, symplectic_euler, velocity_verlet


def test_euler_scalar_one_step_and_nfev():
    sol = euler(lambda t, y: y, (0, 0.1), 1.0, 0.1)
    assert sol.y.shape == (2, 1)
    assert sol.nfev == 1
    np.testing.assert_allclose(sol.y[:, 0], [1, 1.1])


def test_rk4_linear_ode_accurate():
    sol = rk4(lambda t, y: -2*y, (0, 1), [1], 0.1)
    assert abs(sol.y[-1, 0]-np.exp(-2)) < 5e-6
    assert sol.nfev == 40


@pytest.mark.parametrize("method", [euler, rk4])
def test_final_step_reduced_without_overshoot(method):
    sol = method(lambda t, y: np.ones_like(y), (1.0, 2.05), [0], 0.2)
    assert sol.t[-1] == 2.05
    assert np.all(np.diff(sol.t) > 0)
    assert np.max(np.diff(sol.t)) <= 0.2+1e-12
    np.testing.assert_allclose(sol.y[-1], [1.05], atol=1e-12)
    assert sol.y[0, 0] == 0


def test_vector_complex_solution():
    sol = rk4(lambda t, y: 1j*y, (0,1), [1+0j, 0j], 0.05)
    assert sol.y.shape == (21,2)
    assert np.iscomplexobj(sol.y)
    assert abs(sol.y[-1,0]-np.exp(1j)) < 1e-7


def test_zero_duration():
    sol = euler(lambda t, y: y, (2, 2), [3, 4], 0.1)
    assert sol.nfev == 0
    assert sol.y.shape == (1, 2)
    np.testing.assert_array_equal(sol.y[0], [3,4])


@pytest.mark.parametrize("bad", [0, -0.1, np.nan, np.inf])
def test_invalid_dt(bad):
    with pytest.raises(ValueError):
        euler(lambda t, y: y, (0,1), [1], bad)


def test_invalid_initial_rhs_and_final_time():
    with pytest.raises(ValueError):
        rk4(lambda t, y: y, (0,1), [np.nan], 0.1)
    with pytest.raises(ValueError):
        euler(lambda t, y: [1,2], (0,1), [1], 0.1)
    with pytest.raises(ValueError):
        euler(lambda t, y: [np.nan], (0,1), [1], 0.1)
    with pytest.raises(ValueError):
        euler(lambda t, y: 1j*y, (0,1), [1.0], 0.1)
    with pytest.raises(ValueError):
        rk4(lambda t, y: y, (1,0), [1], 0.1)


def test_symplectic_euler_kick_drift():
    sol = symplectic_euler(lambda t, q: -q, (0,0.1), [1], [0], 0.1)
    np.testing.assert_allclose(sol.y[1], [0.99, -0.1])
    assert sol.nfev == 1


def test_verlet_first_step_and_nfev():
    sol = velocity_verlet(lambda t, q: -q, (0,0.1), [1], [0], 0.1)
    np.testing.assert_allclose(sol.y[1], [0.995, -0.09975])
    assert sol.nfev == 2


def test_symplectic_mass_and_dimension_validation():
    sol = symplectic_euler(lambda t,q: -q, (0, 0.1), [1,1], [0,0], 0.1, mass=[1,2])
    assert sol.y.shape == (2,4)
    with pytest.raises(ValueError):
        symplectic_euler(lambda t,q: -q, (0,1), [1], [0], 0.1, mass=-1)
    with pytest.raises(ValueError):
        symplectic_euler(lambda t,q: -q, (0,1), [1,2], [0], 0.1)
