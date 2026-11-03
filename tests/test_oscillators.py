import numpy as np
import pytest
from numerical_dynamics.analytical import damped_exact, harmonic_exact
from numerical_dynamics.convergence import oscillator_solver, state_error
from numerical_dynamics.integrators import euler, rk4
from numerical_dynamics.systems import DampedOscillator, HarmonicOscillator


def test_harmonic_initial_condition_and_exact_energy():
    m = HarmonicOscillator(mass=2, spring=8)
    sol = harmonic_exact(m, np.linspace(0,12,1001), x0=0.6, v0=-0.4)
    np.testing.assert_allclose(sol[0], [.6,-.4], atol=1e-14)
    assert np.max(abs(m.energy(sol)-m.energy(sol[0]))) < 1e-13


def test_euler_energy_growth_and_symplectic_bounded():
    m = HarmonicOscillator()
    e = oscillator_solver(m, "euler", 100, 0.2)
    s = oscillator_solver(m, "symplectic_euler", 100, 0.2)
    assert m.energy(e.y)[-1] > m.energy(e.y)[0]*1e5
    assert np.max(abs(m.energy(s.y)-0.5)) < 0.07


def test_rk4_short_time_and_verlet_energy():
    m = HarmonicOscillator()
    s = oscillator_solver(m, "rk4", 20, 0.05)
    assert np.max(np.linalg.norm(s.y-harmonic_exact(m, s.t),axis=1)) < 2e-6
    verlet = oscillator_solver(m, "velocity_verlet", 100, 0.2)
    assert np.max(abs(m.energy(verlet.y)-0.5)) < 0.006


def test_damped_zero_damping_is_harmonic():
    a = DampedOscillator(mass=2, spring=8, damping=0)
    b = HarmonicOscillator(mass=2, spring=8)
    t = np.linspace(0, 10, 201)
    np.testing.assert_allclose(damped_exact(a,t,0.7,0.2), harmonic_exact(b,t,0.7,0.2), atol=1e-14)


def test_damped_exact_is_solution_and_physical_energy_decreases():
    model = DampedOscillator(damping=0.25)
    t = np.linspace(0,20,1000)
    s = damped_exact(model,t)
    assert np.all(np.diff(model.energy(s)) <= 1e-12)
    assert model.energy(s)[-1] < 0.01
    numerical = rk4(model.rhs, (0,20), [1,0], 0.05)
    assert state_error(numerical.y, damped_exact(model,numerical.t))["max_error"] < 1e-6


def test_damped_euler_accuracy_improves_with_dt():
    model = DampedOscillator()
    coarse = euler(model.rhs, (0,10), [1,0], 0.1)
    fine = euler(model.rhs, (0,10), [1,0], 0.05)
    e1 = state_error(coarse.y, damped_exact(model,coarse.t))["max_error"]
    e2 = state_error(fine.y, damped_exact(model,fine.t))["max_error"]
    assert 1.8 < e1/e2 < 2.3


def test_invalid_physical_parameters():
    with pytest.raises(ValueError):
        HarmonicOscillator(mass=0)
    with pytest.raises(ValueError):
        HarmonicOscillator(spring=-1)
    with pytest.raises(ValueError):
        DampedOscillator(damping=-1)
    with pytest.raises(ValueError):
        DampedOscillator(damping=5)
