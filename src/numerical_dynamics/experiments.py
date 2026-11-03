"""Reproducible end-to-end experiments, diagnostics, and output generation."""

from pathlib import Path
import numpy as np
from .analytical import damped_exact, harmonic_exact, quantum_exact
from .convergence import (DEFAULT_STEPS, fit_orders, oscillator_solver,
                          state_error, study_convergence)
from .integrators import euler, rk4
from .reporting import export_csv, export_json
from .stability import stability_report
from .systems import DampedOscillator, HarmonicOscillator, TwoStateSystem
from .extensions import AvoidedCrossing
from .visualization import (harmonic_figures, damped_figures, quantum_figures,
                            convergence_figure, stability_figure, crossing_figures)


def single_run_metrics(experiment: str, dt: float, t_end: float) -> list[dict]:
    """State errors and physical diagnostics for one visualization run.

    No timing here: benchmark medians are in the convergence study CSV.
    """
    if experiment == "harmonic":
        model = HarmonicOscillator()
        methods = ("euler", "rk4", "symplectic_euler", "velocity_verlet")
    elif experiment == "damped":
        model = DampedOscillator()
        methods = ("euler", "rk4")
    elif experiment == "crossing":
        model = AvoidedCrossing()
        methods = ("euler", "rk4")
    elif experiment == "quantum":
        model = TwoStateSystem()
        methods = ("euler", "rk4")
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    entries = []
    for method in methods:
        if experiment == "harmonic":
            result = oscillator_solver(model, method, t_end, dt)
            exact = harmonic_exact(model, result.t)
            energy = model.energy(result.y)
            diagnostic = float(np.max(abs(energy-energy[0]))/energy[0])
        elif experiment == "damped":
            result = (euler if method == "euler" else rk4)(model.rhs, (0, t_end), [1,0], dt)
            exact = damped_exact(model, result.t)
            diagnostic = None
        elif experiment == "crossing":
            result = (euler if method == "euler" else rk4)(
                model.rhs, (-t_end/2, t_end/2), np.array([1+0j, 0j]), dt)
            exact = model.reference(result.t)
            diagnostic = float(np.max(abs(np.sum(abs(result.y)**2, axis=1)-1)))
        else:
            result = (euler if method == "euler" else rk4)(
                model.rhs, (0, t_end), np.array([1+0j, 0j]), dt)
            exact = quantum_exact(model, result.t)
            diagnostic = float(np.max(abs(model.probability(result.y)-1)))
        entry = {"experiment": experiment, "method": method, "dt": dt, "t_final": t_end,
                 **state_error(result.y, exact), "conservation_error": diagnostic,
                 "runtime_s_median": None, "nfev": result.nfev,
                 "n_steps": result.t.size-1}
        entries.append(entry)
    return entries


def run_experiment(experiment: str, output: Path, dt: float = 0.1,
                   duration: float | None = None) -> dict:
    folder = Path(output)
    figs, data = folder/"figures", folder/"data"
    if experiment == "harmonic":
        t_end = 30.0 if duration is None else duration
        figures = harmonic_figures(figs, dt, t_end)
    elif experiment == "damped":
        t_end = 30.0 if duration is None else duration
        figures = damped_figures(figs, dt, t_end)
    elif experiment == "quantum":
        t_end = 25.0 if duration is None else duration
        figures = quantum_figures(figs, dt, t_end)
    elif experiment == "crossing":
        t_end = 12.0 if duration is None else duration
        figures = crossing_figures(figs, dt, -t_end/2, t_end/2)
    else:
        raise ValueError(f"unknown experiment '{experiment}'")
    metrics = single_run_metrics(experiment, dt, t_end)
    csvpath = export_csv(metrics, data/f"{experiment}_results.csv")
    return {"experiment": experiment, "dt": dt, "t_end": t_end,
            "figures": [str(p) for p in figures], "data": str(csvpath), "metrics": metrics}


def run_convergence(output: Path, step_sizes=DEFAULT_STEPS, repeats: int = 3,
                    duration: float = 11.0) -> dict:
    folder = Path(output)
    records = study_convergence(step_sizes, repeats, duration)
    fits = fit_orders(records)
    csv = export_csv(records, folder/"data"/"convergence_results.csv")
    fits_csv = export_csv(fits, folder/"data"/"convergence_slopes.csv",
                          ["experiment", "method", "slope", "r_squared", "fit_dts"])
    figure = convergence_figure(folder/"figures", records, fits)
    stability_fig = stability_figure(folder/"figures")
    stability = stability_report()
    export_json(stability, folder/"data"/"stability_analysis.json")
    return {"experiment": "convergence", "time_horizon": duration,
            "dt_values": list(step_sizes), "timing_repeats": repeats,
            "data": str(csv), "slope_data": str(fits_csv),
            "figures": [str(figure), str(stability_fig)],
            "slopes": fits, "stability": stability}


def run_all(output: Path, dt: float = 0.1, step_sizes=DEFAULT_STEPS,
            repeats: int = 3, duration: float | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sections = [run_experiment(exp, output, dt, duration)
                for exp in ("harmonic", "damped", "quantum", "crossing")]
    sections.append(run_convergence(output, step_sizes, repeats,
                                    duration if duration is not None else 11.0))
    result = {"project": "Numerical Stability & Error Analysis in Dynamical Systems",
              "units": "dimensionless; two-level system uses hbar=1",
              "timing_warning": "Median wall times depend on hardware, load, and Python version",
              "runs": sections}
    export_json(result, output/"data"/"run_summary.json")
    return result
