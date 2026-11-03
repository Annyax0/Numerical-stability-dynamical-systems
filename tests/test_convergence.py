import numpy as np
import pytest
from numerical_dynamics.convergence import (fit_orders, oscillator_solver,
                                             state_error, study_convergence, validate_step_sizes)
from numerical_dynamics.stability import (oscillator_eigenvalues, euler_amplification,
                                          rk4_amplification, symplectic_euler_matrix)
from numerical_dynamics.analytical import harmonic_exact
from numerical_dynamics.systems import HarmonicOscillator


def test_expected_convergence_orders():
    rows = study_convergence([0.2,0.1,0.05,0.025], repeats=1, t_end=8.0)
    slopes = {(f["experiment"], f["method"]): f["slope"] for f in fit_orders(rows)}
    for exp in ("harmonic", "damped", "quantum"):
        assert 0.90 < slopes[(exp, "euler")] < 1.20
        assert 3.7 < slopes[(exp, "rk4")] < 4.2
    assert 0.9 < slopes[("harmonic", "symplectic_euler")] < 1.2
    assert 1.8 < slopes[("harmonic", "velocity_verlet")] < 2.2


def test_study_common_endpoint_and_counts():
    rows = study_convergence([.31,.2,.08], repeats=1, t_end=3.05)
    assert len(rows) == 24
    assert all(r["t_final"] == 3.05 for r in rows)
    assert all(r["nfev"] > 0 for r in rows)
    assert all(r["runtime_s_median"] >= 0 for r in rows)
    assert all(r["conservation_error"] is None for r in rows if r["experiment"] == "damped")


def test_zero_error_and_invalid_dt_fit_handling():
    rows = [{"experiment":"x", "method":"rk4", "dt":h, "max_error":0.0}
            for h in (.1,.05,.025)]
    assert fit_orders(rows)[0]["slope"] is None
    with pytest.raises(ValueError):
        validate_step_sizes([0,0.1,0.2])
    with pytest.raises(ValueError):
        validate_step_sizes([0.1,0.1,0.2])


def test_stability_amplification_and_threshold():
    eigenvalues = oscillator_eigenvalues()
    np.testing.assert_allclose(np.sort_complex(eigenvalues), [-1j, 1j], atol=1e-14)
    assert abs(euler_amplification(0.01j)) > 1
    assert abs(rk4_amplification(2.5j)) < 1
    assert abs(rk4_amplification(3.0j)) > 1
    assert max(abs(np.linalg.eigvals(symplectic_euler_matrix(1,1.0)))) < 1+1e-13
    assert max(abs(np.linalg.eigvals(symplectic_euler_matrix(1,2.2)))) > 1


def test_error_norm_definition():
    m = HarmonicOscillator()
    sol = oscillator_solver(m, "rk4", 2, .25)
    a = state_error(sol.y, harmonic_exact(m,sol.t))
    assert a["final_error"] <= a["max_error"]
    assert 0 <= a["rms_error"] <= a["max_error"]
    with pytest.raises(ValueError):
        state_error(np.zeros((3,2)), np.zeros((2,2)))
