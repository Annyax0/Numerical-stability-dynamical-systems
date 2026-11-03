"""Error definitions, repeated timing, convergence fitting, and CSV-ready studies."""

from time import perf_counter
from typing import Callable
import numpy as np
from .analytical import damped_exact, harmonic_exact, quantum_exact
from .integrators import IntegrationResult, euler, rk4, symplectic_euler, velocity_verlet
from .systems import DampedOscillator, HarmonicOscillator, TwoStateSystem

DEFAULT_STEPS = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)


def state_error(numerical: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Euclidean state norm, measured on the *numerical* time grid."""
    if numerical.shape != exact.shape:
        raise ValueError("reference must have same shape as numerical trajectory")
    deviations = np.linalg.norm(numerical-exact, axis=-1)
    return {
        "final_error": float(deviations[-1]),
        "max_error": float(np.max(deviations)),
        "rms_error": float(np.sqrt(np.mean(deviations**2))),
    }


def median_timing(solver: Callable[[], IntegrationResult], repeats: int = 3
                  ) -> tuple[IntegrationResult, float]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = solver()
        timings.append(perf_counter()-start)
    assert result is not None
    return result, float(np.median(timings))


def oscillator_solver(model: HarmonicOscillator, method: str,
                      t_end: float, dt: float) -> IntegrationResult:
    if method == "euler":
        return euler(model.rhs, (0, t_end), [1, 0], dt)
    if method == "rk4":
        return rk4(model.rhs, (0, t_end), [1, 0], dt)
    if method in ("symplectic_euler", "velocity_verlet"):
        raw = (symplectic_euler if method == "symplectic_euler" else velocity_verlet)(
            model.force, (0, t_end), [1.0], [0.0], dt, mass=model.mass)
        state = np.column_stack([raw.y[:, 0], raw.y[:, 1]/model.mass])
        return IntegrationResult(raw.t, state, raw.nfev, raw.method)
    raise ValueError(f"unsupported oscillator integrator: {method}")


def validate_step_sizes(step_sizes: tuple[float, ...] | list[float]) -> list[float]:
    a = np.asarray(step_sizes, dtype=float)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("convergence requires at least three positive, finite time steps")
    if len(set(map(float, a))) != a.size:
        raise ValueError("time steps must be distinct")
    return sorted(map(float, a), reverse=True)


def study_convergence(step_sizes: tuple[float, ...] | list[float] = DEFAULT_STEPS,
                      repeats: int = 3, t_end: float = 11.0) -> list[dict]:
    """For each method/dt, use a common final time and analytical references.

    Report the median wall-clock runtime from repeated *solver-only* executions.
    Function evaluation counts exclude evaluation of the analytical reference.
    Damped systems physically dissipate energy: 'conservation_error' is blank.
    """
    steps = validate_step_sizes(step_sizes)
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("t_end must be positive and finite")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    harmonic = HarmonicOscillator(mass=1, spring=1)
    damped = DampedOscillator(mass=1, spring=1, damping=0.25)
    quantum = TwoStateSystem(delta=1, coupling=0.7)
    records: list[dict] = []
    experiments = (
        ("harmonic", ("euler", "rk4", "symplectic_euler", "velocity_verlet")),
        ("damped", ("euler", "rk4")),
        ("quantum", ("euler", "rk4")),
    )
    for experiment, methods in experiments:
        for method in methods:
            for dt in steps:
                if experiment == "harmonic":
                    get = lambda: oscillator_solver(harmonic, method, t_end, dt)
                elif experiment == "damped":
                    get = lambda: (euler if method == "euler" else rk4)(
                        damped.rhs, (0, t_end), [1, 0], dt)
                else:
                    get = lambda: (euler if method == "euler" else rk4)(
                        quantum.rhs, (0, t_end), np.array([1+0j, 0j]), dt)
                result, runtime = median_timing(get, repeats)
                if experiment == "harmonic":
                    exact = harmonic_exact(harmonic, result.t)
                    energy = harmonic.energy(result.y)
                    conservation = float(np.max(np.abs(energy-energy[0]))/abs(energy[0]))
                elif experiment == "damped":
                    exact = damped_exact(damped, result.t)
                    conservation = None  # the model is intentionally dissipative
                else:
                    exact = quantum_exact(quantum, result.t)
                    conservation = float(np.max(np.abs(quantum.probability(result.y)-1.0)))
                errors = state_error(result.y, exact)
                records.append({
                    "experiment": experiment,
                    "method": method,
                    "dt": dt,
                    "t_final": float(result.t[-1]),
                    **errors,
                    "conservation_error": conservation,
                    "runtime_s_median": runtime,
                    "nfev": result.nfev,
                    "n_steps": result.t.size-1,
                })
    return records


def fit_orders(records: list[dict], fit_points: int = 4,
               metric: str = "max_error") -> list[dict]:
    """Fit log(error) vs log(dt) on the smallest reliable steps.

    Exclude errors at or below 1e-13 (near floating-point noise). Return the
    exact dt range and R² so the choice is auditable. Select up to 4 smallest
    remaining dt values by default; never regress over zeros.
    """
    if fit_points < 3:
        raise ValueError("fit_points must be at least 3")
    pairs = sorted({(r["experiment"], r["method"]) for r in records})
    output = []
    for exp, method in pairs:
        rows = sorted((r for r in records if r["experiment"] == exp and r["method"] == method
                       and np.isfinite(r[metric]) and r[metric] > 1e-13),
                      key=lambda r: r["dt"])
        selected = rows[:fit_points]
        if len(selected) < 3:
            output.append({"experiment": exp, "method": method, "slope": None,
                           "r_squared": None, "fit_dts": []})
            continue
        x = np.log([r["dt"] for r in selected])
        y = np.log([r[metric] for r in selected])
        slope, intercept = np.polyfit(x, y, 1)
        prediction = slope*x+intercept
        residual = float(np.sum((y-prediction)**2))
        total = float(np.sum((y-np.mean(y))**2))
        output.append({"experiment": exp, "method": method,
                       "slope": float(slope),
                       "r_squared": float(1-residual/total) if total else None,
                       "fit_dts": [r["dt"] for r in selected]})
    return output
