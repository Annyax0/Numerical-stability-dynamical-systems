"""Optional time-dependent two-level avoided-crossing model.

This is NOT fewest-switches surface hopping. The classical nuclear trajectory,
nonadiabatic coupling calculations, and stochastic hopping are absent. SciPy is
used only for an independent high-accuracy reference, not for custom methods.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .systems import _positive_finite


@dataclass(frozen=True)
class AvoidedCrossing:
    rate: float = 0.5
    coupling: float = 0.7

    def __post_init__(self) -> None:
        _positive_finite("rate", self.rate)
        if not np.isfinite(self.coupling):
            raise ValueError("coupling must be finite")

    def hamiltonian(self, t: float) -> np.ndarray:
        return np.array([[self.rate*t, self.coupling],
                         [self.coupling, -self.rate*t]], dtype=np.complex128)

    def rhs(self, t: float, psi: np.ndarray) -> np.ndarray:
        return -1j * (self.hamiltonian(t) @ psi)

    def energies(self, t: np.ndarray) -> np.ndarray:
        gap = np.sqrt((self.rate*np.asarray(t))**2+self.coupling**2)
        return np.stack((-gap, gap), axis=-1)

    def reference(self, times: np.ndarray, psi0: np.ndarray | None = None) -> np.ndarray:
        """Trusted high-accuracy DOP853 reference; not an analytic solution.

        Numerical error in the reference is limited by rtol=1e-11 and
        atol=1e-13, so experimental errors below that scale aren't meaningful.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("times must be a strictly increasing 1-D grid")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite")
        if psi0 is None:
            psi0 = np.array([1+0j, 0j])
        psi0 = np.asarray(psi0, dtype=np.complex128)
        if psi0.shape != (2,) or not np.all(np.isfinite(psi0)):
            raise ValueError("psi0 must be a finite complex two-vector")
        sol = solve_ivp(self.rhs, (times[0], times[-1]), psi0,
                        t_eval=times, method="DOP853", rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise RuntimeError(f"reference ODE integration failed: {sol.message}")
        return sol.y.T
