"""Linear amplification analysis of oscillator numerical methods."""

import numpy as np


def oscillator_eigenvalues(omega: float = 1.0) -> np.ndarray:
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be positive and finite")
    return np.linalg.eigvals([[0.0, 1.0], [-omega**2, 0.0]])


def euler_amplification(z: complex) -> complex:
    return 1 + z


def rk4_amplification(z: complex) -> complex:
    return 1 + z + z*z/2 + z**3/6 + z**4/24


def symplectic_euler_matrix(omega: float, dt: float) -> np.ndarray:
    """Kick-drift mapping for the (q,p) oscillator with unit mass."""
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("omega and dt must be positive and finite")
    return np.array([[1-(dt*omega)**2, dt], [-dt*omega**2, 1]])


def stability_report(omega: float = 1.0, dt: float = 0.2) -> dict:
    z = 1j*omega*dt
    matrix_eigenvalues = np.linalg.eigvals(symplectic_euler_matrix(omega, dt))
    return {
        "omega": omega,
        "dt": dt,
        "continuous_eigenvalues": [str(val) for val in oscillator_eigenvalues(omega)],
        "euler_amplification_modulus": float(abs(euler_amplification(z))),
        "rk4_amplification_modulus": float(abs(rk4_amplification(z))),
        "symplectic_euler_eigenvalue_moduli": [float(abs(lam)) for lam in matrix_eigenvalues],
        "rk4_imaginary_axis_threshold_omega_dt": float(np.sqrt(8)),
        "symplectic_euler_threshold_omega_dt": 2.0,
    }
