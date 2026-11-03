"""Original fixed-step ODE integrators; no SciPy ODE solver wrappers.

The last interval is shortened if it does not divide the requested time span.
All states have shape (n_times, n_components). For geometric methods the
returned state is ordered [q_1,...,q_d,p_1,...,p_d] in canonical variables.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray
ODE = Callable[[float, Array], Array]
Force = Callable[[float, Array], Array]


@dataclass(frozen=True)
class IntegrationResult:
    t: Array
    y: Array
    nfev: int
    method: str


def _initial_vector(y0: Array | list[float] | complex | float) -> Array:
    raw = np.asarray(y0)
    if raw.ndim > 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("initial state must be a numeric scalar or nonempty 1-D vector")
    arr = np.atleast_1d(raw).astype(np.complex128 if np.iscomplexobj(raw) else np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("initial state must contain only finite values")
    return arr


def _grid(t_span: tuple[float, float], dt: float) -> Array:
    try:
        t0, tf = map(float, t_span)
        dt = float(dt)
    except (ValueError, TypeError) as exc:
        raise ValueError("t_span must contain two real endpoints and dt must be real") from exc
    if not (np.isfinite(t0) and np.isfinite(tf) and np.isfinite(dt)):
        raise ValueError("time endpoints and time step must be finite")
    if dt <= 0:
        raise ValueError("time step dt must be positive")
    if tf < t0:
        raise ValueError("final time must not precede initial time")
    if tf == t0:
        return np.array([t0])
    ratio = (tf - t0) / dt
    if not np.isfinite(ratio) or ratio > 2_000_000:
        raise ValueError("time grid exceeds the maximum of 2,000,000 steps")
    near_integer = round(ratio)
    n = int(near_integer if np.isclose(ratio, near_integer, rtol=1e-12, atol=1e-12)
            else np.ceil(ratio))
    t = t0 + dt * np.arange(n + 1, dtype=float)
    t[-1] = tf  # requested final time, not a different experimental endpoint
    if np.any(np.diff(t) <= 0):
        raise ValueError("time increment too small to resolve at these time values")
    return t


def _checked_rhs(f: ODE, t: float, y: Array) -> Array:
    out = np.asarray(f(float(t), y))
    if out.size == 1 and y.size == 1:
        out = out.reshape(1)
    if out.shape != y.shape or not np.issubdtype(out.dtype, np.number):
        raise ValueError(f"RHS must return numeric shape {y.shape}; got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("RHS returned a nonfinite derivative")
    if np.iscomplexobj(out) and not np.iscomplexobj(y):
        raise ValueError("complex-valued derivatives require a complex initial state")
    return out


def _integrate(f: ODE, t_span: tuple[float, float], y0: Array,
               dt: float, method: str) -> IntegrationResult:
    if not callable(f):
        raise ValueError("RHS f must be callable")
    t = _grid(t_span, dt)
    initial = _initial_vector(y0)
    y = np.empty((t.size, initial.size), dtype=initial.dtype)
    y[0] = initial
    nfev = 0
    for i, h in enumerate(np.diff(t)):
        ti, yi = t[i], y[i]
        if method == "euler":
            update = yi + h * _checked_rhs(f, ti, yi)
            nfev += 1
        else:
            k1 = _checked_rhs(f, ti, yi)
            k2 = _checked_rhs(f, ti + h/2, yi + (h/2)*k1)
            k3 = _checked_rhs(f, ti + h/2, yi + (h/2)*k2)
            k4 = _checked_rhs(f, ti + h, yi + h*k3)
            update = yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            nfev += 4
        if not np.all(np.isfinite(update)):
            raise FloatingPointError(f"{method} produced nonfinite state at t={t[i+1]:g}")
        y[i+1] = update
    return IntegrationResult(t, y, nfev, method)


def euler(f: ODE, t_span: tuple[float, float], y0: Array, dt: float) -> IntegrationResult:
    """Forward Euler: y[n+1] = y[n] + h*f(t[n], y[n]); global order 1."""
    return _integrate(f, t_span, y0, dt, "euler")


def rk4(f: ODE, t_span: tuple[float, float], y0: Array, dt: float) -> IntegrationResult:
    """Classical explicit four-stage Runge–Kutta; global order 4 for smooth ODEs."""
    return _integrate(f, t_span, y0, dt, "rk4")


def _geometric(force: Force, t_span: tuple[float, float], q0: Array, p0: Array,
               dt: float, mass: float | Array, method: str) -> IntegrationResult:
    """Integrate separable H(q,p,t)=sum(p_i**2/(2*m_i))+U(q,t).

    force(t,q) is -grad_q U. Neither p-dependent forces nor arbitrary damping
    are supported by these geometric methods. The final step may be shorter.
    """
    if not callable(force):
        raise ValueError("force must be callable")
    t = _grid(t_span, dt)
    q, p = _initial_vector(q0), _initial_vector(p0)
    if q.shape != p.shape or np.iscomplexobj(q) or np.iscomplexobj(p):
        raise ValueError("q0 and p0 must be real vectors of equal size")
    m = np.asarray(mass, dtype=float)
    if (m.ndim > 1 or (m.ndim == 1 and m.shape != q.shape)
            or not np.all(np.isfinite(m)) or np.any(m <= 0)):
        raise ValueError("mass must be a positive scalar or a positive vector matching q0")
    y = np.empty((t.size, 2*q.size), dtype=float)
    y[0] = np.concatenate([q,p])
    nfev = 0

    def checked_force(ti: float, qi: Array) -> Array:
        raw_force = np.asarray(force(float(ti), qi))
        if not np.issubdtype(raw_force.dtype, np.number) or np.iscomplexobj(raw_force):
            raise ValueError("force must be real-valued and numeric")
        force_vec = np.asarray(raw_force, dtype=float)
        if force_vec.size == 1 and qi.size == 1:
            force_vec = force_vec.reshape(1)
        if force_vec.shape != qi.shape or not np.all(np.isfinite(force_vec)):
            raise ValueError("force must return a finite real vector matching q0")
        return force_vec

    for i, h in enumerate(np.diff(t)):
        if method == "symplectic_euler":
            p = p + h*checked_force(t[i], q)  # kick, then drift
            q = q + h*p/m
            nfev += 1
        else:
            a0 = checked_force(t[i], q)
            q_new = q + h*p/m + (h*h/2)*a0/m
            a1 = checked_force(t[i+1], q_new)
            p = p + (h/2)*(a0+a1)
            q = q_new
            nfev += 2
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            raise FloatingPointError(f"{method} produced nonfinite state at t={t[i+1]:g}")
        y[i+1] = np.concatenate([q,p])
    return IntegrationResult(t, y, nfev, method)


def symplectic_euler(force: Force, t_span: tuple[float, float], q0: Array,
                      p0: Array, dt: float, mass: float | Array = 1.0) -> IntegrationResult:
    """Kick-drift symplectic Euler: p+=h*F(t,q), q+=h*p/m.

    Symplecticity applies to separable canonical Hamiltonians for each fixed
    step; this method does NOT conserve the exact physical energy every step.
    """
    return _geometric(force, t_span, q0, p0, dt, mass, "symplectic_euler")


def velocity_verlet(force: Force, t_span: tuple[float, float], q0: Array,
                    p0: Array, dt: float, mass: float | Array = 1.0) -> IntegrationResult:
    """Velocity Verlet (kick-drift-kick), second-order symplectic for H=T(p)+U(q).

    For explicitly time-dependent potentials, this implementation uses force
    evaluations at each step's endpoints and is not a generic Hamiltonian solver.
    """
    return _geometric(force, t_span, q0, p0, dt, mass, "velocity_verlet")
