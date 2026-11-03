import numpy as np
import pytest
from scipy.linalg import expm
from numerical_dynamics.analytical import quantum_exact
from numerical_dynamics.convergence import state_error
from numerical_dynamics.integrators import euler, rk4
from numerical_dynamics.systems import TwoStateSystem


def test_hamiltonian_is_hermitian():
    m = TwoStateSystem(delta=-0.3, coupling=0.7)
    np.testing.assert_allclose(m.hamiltonian, m.hamiltonian.conj().T)
    assert np.max(abs(np.linalg.eigvalsh(m.hamiltonian).imag)) == 0


def test_exact_unitarity_and_scipy_expm_agree():
    m = TwoStateSystem()
    t = np.linspace(0, 30, 103)
    exact = quantum_exact(m, t)
    np.testing.assert_allclose(m.probability(exact), 1.0, atol=5e-14)
    for j in [0,50,102]:
        np.testing.assert_allclose(exact[j], expm(-1j*m.hamiltonian*t[j]) @ np.array([1,0]), atol=3e-14)


def test_euler_norm_drift_and_rk4_accuracy():
    m = TwoStateSystem()
    psi0 = np.array([1+0j,0j])
    e = euler(m.rhs, (0,10), psi0, 0.1)
    r = rk4(m.rhs, (0,10), psi0, 0.1)
    assert m.probability(e.y)[-1] > 1.05
    assert np.max(abs(m.probability(r.y)-1)) < 1e-5
    assert state_error(r.y, quantum_exact(m,r.t))["max_error"] < 5e-5


def test_quantum_rk4_converges_by_fourth_order():
    m = TwoStateSystem()
    errors=[]
    for h in [0.2, 0.1, 0.05]:
        s = rk4(m.rhs, (0,9), np.array([1+0j,0j]), h)
        errors.append(state_error(s.y, quantum_exact(m,s.t))["max_error"])
    assert 14 < errors[0]/errors[1] < 18
    assert 14 < errors[1]/errors[2] < 18


def test_invalid_quantum_parameters():
    with pytest.raises(ValueError):
        TwoStateSystem(delta=np.inf)
    with pytest.raises(ValueError):
        quantum_exact(TwoStateSystem(), [0,1], [np.nan, 0])


def test_crossing_reference_unitarity_and_rk4_convergence():
    from numerical_dynamics.extensions import AvoidedCrossing
    model = AvoidedCrossing(rate=.5, coupling=.7)
    t0, tf = -5, 5
    errs=[]
    for h in (.2,.1):
        s = rk4(model.rhs, (t0,tf), np.array([1+0j,0j]), h)
        ref = model.reference(s.t)
        errs.append(np.max(np.linalg.norm(s.y-ref, axis=1)))
        np.testing.assert_allclose(np.sum(abs(ref)**2,axis=1), 1, atol=1e-9)
    assert errs[1] < errs[0]/10
    assert np.allclose(model.hamiltonian(.3), model.hamiltonian(.3).conj().T)
    np.testing.assert_allclose(model.energies([0])[0], [-.7,.7])
