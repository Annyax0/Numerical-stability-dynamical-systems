"""Fixed-step integration and transparent numerical experiments."""

from .integrators import IntegrationResult, euler, rk4, symplectic_euler, velocity_verlet
from .systems import HarmonicOscillator, DampedOscillator, TwoStateSystem

__version__ = "1.0.0"
__all__ = [
    "IntegrationResult", "euler", "rk4", "symplectic_euler", "velocity_verlet",
    "HarmonicOscillator", "DampedOscillator", "TwoStateSystem",
]
