"""Physical models in dimensionless units unless explicitly parameterized."""

from dataclasses import dataclass
import numpy as np


def _positive_finite(name: str, x: float, *, allow_zero: bool = False) -> None:
    if not np.isfinite(x) or (x < 0 if allow_zero else x <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")


@dataclass(frozen=True)
class HarmonicOscillator:
    mass: float = 1.0
    spring: float = 1.0

    def __post_init__(self) -> None:
        _positive_finite("mass", self.mass)
        _positive_finite("spring", self.spring)

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.spring / self.mass))

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -(self.spring/self.mass)*y[0]])

    def force(self, t: float, q: np.ndarray) -> np.ndarray:
        return -self.spring*q

    def energy(self, y: np.ndarray) -> np.ndarray:
        z = np.asarray(y)
        return 0.5*self.mass*z[..., 1]**2 + 0.5*self.spring*z[..., 0]**2


@dataclass(frozen=True)
class DampedOscillator(HarmonicOscillator):
    damping: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        _positive_finite("damping", self.damping, allow_zero=True)
        if self.damping**2 >= 4*self.mass*self.spring:
            raise ValueError("this demonstration requires an underdamped oscillator")

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -(self.spring*y[0]+self.damping*y[1])/self.mass])


@dataclass(frozen=True)
class TwoStateSystem:
    delta: float = 1.0
    coupling: float = 0.7

    def __post_init__(self) -> None:
        if not (np.isfinite(self.delta) and np.isfinite(self.coupling)):
            raise ValueError("Hamiltonian parameters must be finite")

    @property
    def hamiltonian(self) -> np.ndarray:
        return np.array([[self.delta/2, self.coupling],
                         [self.coupling, -self.delta/2]], dtype=np.complex128)

    def rhs(self, t: float, psi: np.ndarray) -> np.ndarray:
        """Schrödinger equation with hbar=1: dpsi/dt = -i H psi."""
        return -1j*(self.hamiltonian @ psi)

    @staticmethod
    def populations(y: np.ndarray) -> np.ndarray:
        return np.abs(y)**2

    @staticmethod
    def probability(y: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(y)**2, axis=-1)
