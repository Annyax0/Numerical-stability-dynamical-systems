"""Exact reference trajectories; time t is measured from the starting state."""

import numpy as np
from .systems import HarmonicOscillator, DampedOscillator, TwoStateSystem


def harmonic_exact(model: HarmonicOscillator, t: np.ndarray,
                   x0: float = 1.0, v0: float = 0.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    w = model.omega
    x = x0*np.cos(w*t) + (v0/w)*np.sin(w*t)
    v = -x0*w*np.sin(w*t) + v0*np.cos(w*t)
    return np.stack((x, v), axis=-1)


def damped_exact(model: DampedOscillator, t: np.ndarray,
                  x0: float = 1.0, v0: float = 0.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    gamma = model.damping/(2*model.mass)
    wd = np.sqrt(model.omega**2-gamma**2)
    a, b = x0, (v0+gamma*x0)/wd
    oscillation = a*np.cos(wd*t) + b*np.sin(wd*t)
    x = np.exp(-gamma*t)*oscillation
    v = np.exp(-gamma*t)*(-gamma*oscillation - a*wd*np.sin(wd*t) + b*wd*np.cos(wd*t))
    return np.stack((x, v), axis=-1)


def quantum_exact(model: TwoStateSystem, t: np.ndarray,
                  psi0: np.ndarray | None = None) -> np.ndarray:
    """Exact unitary spectral evolution, equivalent to scipy.linalg.expm(-1j*H*t).

    A constant Hermitian H has real eigenvalues and orthonormal eigenvectors.
    Computing eigenpairs once is faster than calling expm for every sample.
    """
    if psi0 is None:
        psi0 = np.array([1.0+0j, 0j])
    psi0 = np.asarray(psi0, dtype=complex)
    if psi0.shape != (2,) or not np.all(np.isfinite(psi0)):
        raise ValueError("psi0 must be a finite complex length-two vector")
    times = np.atleast_1d(np.asarray(t, dtype=float))
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    energies, basis = np.linalg.eigh(model.hamiltonian)
    amplitudes = basis.conj().T @ psi0
    phases = np.exp(-1j * times[:, None] * energies[None, :])
    return (phases*amplitudes[None, :]) @ basis.T
