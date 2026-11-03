"""CLI: python -m numerical_dynamics.cli --experiment all."""

import argparse
from pathlib import Path
import sys
from .convergence import DEFAULT_STEPS, validate_step_sizes
from .experiments import run_all, run_convergence, run_experiment
from .reporting import export_json


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible fixed-step numerical dynamics and convergence experiments")
    parser.add_argument("--experiment", choices=("harmonic", "damped", "quantum", "crossing", "convergence", "all"),
                        default="all", help="experiment to run")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="destination directory")
    parser.add_argument("--dt", type=float, default=0.1, help="time step for individual example plots")
    parser.add_argument("--dt-values", type=str,
                        default=",".join(map(str, DEFAULT_STEPS)),
                        help="comma-separated time steps for convergence analysis")
    parser.add_argument("--repeats", type=int, default=3,
                        help="number of repeated solver timings per method and dt")
    parser.add_argument("--duration", type=float, default=None,
                        help="override final time (single experiment defaults: 30, 30, 25, 12; convergence: 11)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        if not (0 < args.dt < float("inf")):
            raise ValueError("--dt must be positive and finite")
        if args.repeats < 1:
            raise ValueError("--repeats must be a positive integer")
        if args.duration is not None and not (0 < args.duration < float("inf")):
            raise ValueError("--duration must be positive and finite")
        steps = validate_step_sizes([float(s.strip()) for s in args.dt_values.split(",")])
        if args.experiment == "all":
            result = run_all(args.output, args.dt, steps, args.repeats, args.duration)
            print("Finished harmonic, damped, quantum, optional crossing, convergence, and stability analyses.")
            slopes = result["runs"][-1]["slopes"]
        elif args.experiment == "convergence":
            result = run_convergence(args.output, steps, args.repeats,
                                     args.duration if args.duration is not None else 11.0)
            export_json(result, args.output/"data"/"convergence_summary.json")
            slopes = result["slopes"]
            print("Finished convergence and stability analysis.")
        else:
            result = run_experiment(args.experiment, args.output, args.dt, args.duration)
            export_json(result, args.output/"data"/f"{args.experiment}_summary.json")
            slopes = []
            print(f"Finished {args.experiment} experiment.")
        for fit in slopes:
            slope = fit["slope"]
            print(f"  {fit['experiment']:<9} {fit['method']:<19} observed p = "
                  f"{slope:.3f}" if slope is not None else
                  f"  {fit['experiment']} {fit['method']}: insufficient non-roundoff data")
        print(f"Data and figures saved in: {args.output.resolve()}")
    except (ValueError, FloatingPointError, OverflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
