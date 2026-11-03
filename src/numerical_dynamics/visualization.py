"""Matplotlib-only publication-style visual summaries; noninteractive backend."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .analytical import harmonic_exact, damped_exact, quantum_exact
from .convergence import oscillator_solver
from .integrators import euler, rk4
from .systems import DampedOscillator, HarmonicOscillator, TwoStateSystem
from .extensions import AvoidedCrossing

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.18,
})
LABELS = {"euler": "Forward Euler", "rk4": "RK4", "symplectic_euler": "Symplectic Euler",
          "velocity_verlet": "Velocity Verlet"}


def _save(fig, folder: Path, name: str) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def harmonic_figures(folder: Path, dt: float = 0.1, t_end: float = 30.0) -> list[Path]:
    model = HarmonicOscillator()
    methods = ("euler", "rk4", "symplectic_euler", "velocity_verlet")
    sols = {method: oscillator_solver(model, method, t_end, dt) for method in methods}
    tt = np.linspace(0, t_end, 1800)
    exact = harmonic_exact(model, tt)
    saved = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(tt, exact[:, 0], "k--", lw=1.8, label="Analytical solution")
    for method, s in sols.items():
        ax.plot(s.t, s.y[:, 0], lw=1.2, alpha=0.88, label=LABELS[method])
    ax.set(xlabel="Time (dimensionless)", ylabel="Position x", title=f"Harmonic oscillator | dt={dt:g}")
    ax.legend(ncol=3)
    saved.append(_save(fig, folder, "harmonic_position"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for method, s in sols.items():
        err = np.linalg.norm(s.y-harmonic_exact(model, s.t), axis=1)
        ax.semilogy(s.t, np.maximum(err, 1e-16), lw=1.6, label=LABELS[method])
    ax.set(xlabel="Time (dimensionless)", ylabel="Full-state error (Euclidean norm)",
           title="Accumulated trajectory error")
    ax.legend(ncol=2)
    saved.append(_save(fig, folder, "harmonic_error"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for ax in axes:
        ax.axhline(0.5, color="black", ls="--", lw=1.3, label="Exact energy")
        ax.set(xlabel="Time (dimensionless)", ylabel="Mechanical energy E")
    for method, s in sols.items():
        axes[0].plot(s.t, model.energy(s.y), lw=1.45, label=LABELS[method])
        if method != "euler":
            axes[1].plot(s.t, model.energy(s.y), lw=1.45, label=LABELS[method])
    axes[0].set_title("Full scale: Euler's energy growth")
    axes[1].set(title="Zoom: geometric and RK4 methods", ylim=(0.445, 0.555))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    saved.append(_save(fig, folder, "harmonic_energy"))

    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    ax.plot(exact[:, 0], exact[:, 1], "k--", label="Exact closed orbit")
    for method, s in sols.items():
        ax.plot(s.y[:, 0], s.y[:, 1], lw=1.4, label=LABELS[method])
    ax.set(xlabel="Position x", ylabel="Velocity v", title="Phase space: physical and numerical orbits")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)
    saved.append(_save(fig, folder, "harmonic_phase_space"))

    long_time = 200.0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for method in methods:
        s = oscillator_solver(model, method, long_time, 0.15)
        drift = abs(model.energy(s.y)-0.5)/0.5
        ax.semilogy(s.t[1:], np.maximum(drift[1:], 1e-16), label=LABELS[method], lw=1.35)
    ax.set(xlabel="Time (dimensionless)", ylabel="Absolute relative energy drift",
           title="Long-time energy behavior | dt=0.15 (same for all methods)")
    ax.legend(ncol=2)
    saved.append(_save(fig, folder, "harmonic_long_time_energy_drift"))
    return saved


def damped_figures(folder: Path, dt: float = 0.1, t_end: float = 30.0) -> list[Path]:
    model = DampedOscillator()
    solutions = {method: integrator(model.rhs, (0, t_end), [1,0], dt)
                 for method, integrator in (("euler", euler), ("rk4", rk4))}
    tt = np.linspace(0, t_end, 1000)
    exact = damped_exact(model, tt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    axes[0].plot(tt, exact[:, 0], "k--", label="Analytical solution")
    axes[1].plot(tt, model.energy(exact), "k--", label="Analytical energy")
    for method, s in solutions.items():
        axes[0].plot(s.t, s.y[:, 0], label=LABELS[method], lw=1.4)
        axes[1].plot(s.t, model.energy(s.y), label=LABELS[method], lw=1.4)
    axes[0].set(xlabel="Time", ylabel="Position", title="Damped oscillation")
    axes[1].set(xlabel="Time", ylabel="Mechanical energy", title="Physical dissipation (E decreases)")
    for ax in axes:
        ax.legend(fontsize=8)
    return [_save(fig, folder, "damped_comparison")]


def quantum_figures(folder: Path, dt: float = 0.1, t_end: float = 25.0) -> list[Path]:
    model = TwoStateSystem()
    psi0 = np.array([1+0j, 0j])
    sols = {method: integrator(model.rhs, (0,t_end), psi0, dt)
            for method, integrator in (("euler", euler), ("rk4", rk4))}
    tt = np.linspace(0, t_end, 1000)
    exact_p = model.populations(quantum_exact(model, tt))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for j in range(2):
        axes[j].plot(tt, exact_p[:, j], "k--", lw=1.8, label="Exact unitary")
        for method, s in sols.items():
            axes[j].plot(s.t, model.populations(s.y)[:, j], label=LABELS[method], lw=1.2)
        axes[j].set(xlabel="Time (ℏ=1)", ylabel=f"Population P{j+1}",
                     title=f"State {j+1} population")
        axes[j].legend(fontsize=8)
    saved = [_save(fig, folder, "quantum_populations")]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    for method, s in sols.items():
        drift = np.abs(model.probability(s.y)-1)
        ax.semilogy(s.t, np.maximum(drift, 1e-16), lw=1.5, label=LABELS[method])
    ax.set(xlabel="Time (ℏ=1)", ylabel="|P₁ + P₂ − 1|",
           title="Numerical probability-norm violation (exact evolution is unitary)")
    ax.legend()
    saved.append(_save(fig, folder, "quantum_probability_drift"))
    return saved


def convergence_figure(folder: Path, records: list[dict], fits: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, exp in zip(axes, ("harmonic", "damped", "quantum")):
        methods = sorted({r["method"] for r in records if r["experiment"] == exp})
        for method in methods:
            rows = sorted((r for r in records if r["experiment"] == exp and r["method"] == method),
                          key=lambda r: r["dt"])
            fit = next(f for f in fits if f["experiment"] == exp and f["method"] == method)
            slope = f"p={fit['slope']:.2f}" if fit["slope"] is not None else "fit unavailable"
            ax.loglog([r["dt"] for r in rows], [r["max_error"] for r in rows],
                      "o-", markersize=4, label=f"{LABELS[method]} ({slope})")
        ax.set(xlabel="Time step dt", title=f"{exp.capitalize()} (T=11)")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Maximum full-state error")
    fig.suptitle("Measured convergence: slope of log(error) vs log(dt)")
    return _save(fig, folder, "convergence_order")


def stability_figure(folder: Path) -> Path:
    model = HarmonicOscillator()
    cases = (("euler", 0.25, "Euler: always amplifies"),
             ("rk4", 2.5, "RK4: stable dt=2.5"),
             ("rk4", 3.0, "RK4: unstable dt=3.0"),
             ("symplectic_euler", 1.0, "Symplectic: bounded dt=1.0"),
             ("symplectic_euler", 2.2, "Symplectic: unstable dt=2.2"))
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.axhline(1, color="black", ls="--", label="Exact phase-space norm")
    for method, dt, label in cases:
        s = oscillator_solver(model, method, 60.0, dt)
        ax.semilogy(s.t, np.linalg.norm(s.y, axis=1), lw=1.45, label=label)
    ax.set(xlabel="Time", ylabel="Euclidean norm of (x, v)",
           title="Stability regions at ω=1 (bounded ≠ accurate)")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, folder, "stability_regimes")


def crossing_figures(folder: Path, dt: float = 0.1, t_start: float = -6,
                     t_end: float = 6) -> list[Path]:
    """Optional avoided-crossing dynamics versus a tight-tolerance DOP853 reference."""
    model = AvoidedCrossing()
    psi0 = np.array([1+0j, 0j])
    solutions = {name: solver(model.rhs, (t_start,t_end), psi0,dt)
                 for name, solver in (("euler", euler), ("rk4", rk4))}
    t = solutions["rk4"].t
    reference = model.reference(t, psi0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, abs(reference[:,0])**2, "k--", label="DOP853 reference |1⟩")
    for name, sol in solutions.items():
        axes[0].plot(sol.t, abs(sol.y[:,0])**2, lw=1.4, label=f"{LABELS[name]} |1⟩")
    axes[0].set(xlabel="Time (ℏ=1)", ylabel="Diabatic state population P₁",
                title="Population transfer near the avoided crossing")
    axes[0].legend(fontsize=8)
    for name, sol in solutions.items():
        err = np.linalg.norm(sol.y-reference, axis=1)
        axes[1].semilogy(sol.t, np.maximum(err, 1e-16), label=LABELS[name], lw=1.5)
    axes[1].set(xlabel="Time (ℏ=1)", ylabel="Full-state error vs DOP853",
                title="Independent numerical reference (not exact)")
    axes[1].legend()
    saved = [_save(fig, folder, "crossing_population_and_error")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    levels = model.energies(np.linspace(t_start,t_end,500))
    tt = np.linspace(t_start,t_end,500)
    ax.plot(tt, levels[:,0], label="Lower adiabatic energy")
    ax.plot(tt, levels[:,1], label="Upper adiabatic energy")
    ax.plot(tt, model.rate*tt, "k:", lw=1.0, alpha=.6, label="Uncoupled diabatic energies")
    ax.plot(tt, -model.rate*tt, "k:", lw=1.0, alpha=.6)
    ax.set(xlabel="Time", ylabel="Energy (ℏ=1)", title="Avoided crossing: gap = 2|V| at t=0")
    ax.legend(fontsize=9)
    saved.append(_save(fig, folder, "crossing_energy_levels"))
    return saved
