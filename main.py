"""
main.py
=======

Physics-Informed Neural Network (PINN) for the Duffing oscillator

        m x''(t) + c x'(t) + k x(t) + alpha x(t)^3 = 0,
        x(t0) = x0,   x'(t0) = v0,      t in [TIME_START, TIME_END]

Two experiments (choose with EXPERIMENT_MODE in config.py or --mode):

    physics_only        (Mode A) all parameters known; the network learns from the
                        ODE residual and the initial conditions only, and is then
                        compared with a numerical reference solution (SciPy).
    damping_estimation  (Mode B) c is unknown; it is learned together with the
                        network from position measurements (damping_estimation.py).

Run with:
    python main.py                            # mode from config.py
    python main.py --mode physics_only
    python main.py --mode damping_estimation
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from physics import analytical_solution, make_reference_solver
from plots import (check_metrics_are_finite, describe_system, plot_results, print_metrics,
                   save_json, settings_summary)
from training import (EXPERIMENT_MODES, build_model, evaluate_pinn, load_config,
                      make_evaluation_points, set_seed, train_pinn)
from visualization import create_training_visualizations


# ----------------------------------------------------------------------------
# Mode A: physics-only PINN
# ----------------------------------------------------------------------------
def run_physics_only(cfg):
    """Train on physics + ICs only (NO data loss), then compare with the reference."""
    output_dir = os.path.join(cfg.OUTPUT_DIR, "physics_only")
    set_seed(cfg.SEED)
    torch.set_default_dtype(torch.float32)

    subtitle = describe_system(cfg)
    print("Mode A: physics-only PINN")
    print("System:", subtitle)
    print(f"Time interval: [{cfg.TIME_START:g}, {cfg.TIME_END:g}] s | "
          f"collocation points: {cfg.N_COLLOCATION} | evaluation points: {cfg.N_EVALUATION_POINTS}")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params} trainable parameters")
    initial_params = [p.detach().clone() for p in model.parameters()]

    start = time.time()
    # observations=None: the loss is physics + initial conditions ONLY.
    history, best_epoch, snapshots = train_pinn(model, cfg, observations=None)
    training_time_s = time.time() - start
    print(f"Training took {training_time_s:.1f} s")

    # Check that training actually changed the parameters.
    change = sum((p.detach() - p0).abs().sum().item()
                 for p, p0 in zip(model.parameters(), initial_params))
    print(f"Total absolute change in parameters: {change:.4e} (must be > 0)")

    # The numerical reference solution is computed only now, AFTER training.
    reference = make_reference_solver(cfg, cfg.DAMPING_COEFFICIENT)
    t_eval = make_evaluation_points(cfg).numpy().astype(np.float64)
    x_reference = reference(t_eval)
    t, x_pred, residual, metrics = evaluate_pinn(model, cfg, x_reference)

    if cfg.NONLINEAR_COEFFICIENT == 0:
        # Linear case: a closed-form solution exists, so we can also check the
        # numerical reference itself.
        x_exact = analytical_solution(t_eval, cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT,
                                      cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY, cfg.TIME_START)
        metrics["reference_vs_analytical_max_abs_diff_m"] = float(np.max(np.abs(x_reference - x_exact)))
    metrics["last_epoch_loss_total_weighted"] = history["total"][-1]
    metrics["last_epoch_loss_physics_N2"] = history["physics"][-1]
    metrics["last_epoch_loss_ic"] = history["ic"][-1]

    check_metrics_are_finite(metrics)
    print_metrics(metrics, "Evaluation metrics (computed AFTER training; not training objectives)")
    print(f"{'best_epoch':45s} {best_epoch}")

    plot_results(t, x_pred, x_reference, residual, history, cfg, subtitle, output_dir)

    visualization_info = {"enabled": False}
    if cfg.ENABLE_TRAINING_VISUALIZATION:
        print("\nCreating training visualizations ...")
        visualization_info = create_training_visualizations(snapshots, history, cfg, subtitle,
                                                            reference, output_dir)
        visualization_info["enabled"] = True

    report = settings_summary(cfg, n_params, damping_known=True)
    report["results"] = {"best_epoch": best_epoch, "training_time_s": training_time_s}
    report["metrics"] = metrics
    report["training_visualization"] = visualization_info
    report["notes"] = (
        "Metrics are computed AFTER training; the reference solution is never used in the training "
        "loss (physics + initial conditions only). Reference = scipy solve_ivp, a numerical "
        "approximation. 'final_model_*' losses are the training losses of the weights actually used "
        "(restored best weights if KEEP_BEST_WEIGHTS). Physics residual is in newtons (N).")
    path = save_json(report, output_dir)
    print(f"\nPlots and metrics saved in: {output_dir}  ({os.path.basename(path)})")
    return metrics


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main(overrides=None):
    """Run the experiment selected by EXPERIMENT_MODE (overrides can change any setting)."""
    cfg = load_config(overrides)
    if cfg.EXPERIMENT_MODE == "physics_only":
        return run_physics_only(cfg)
    from damping_estimation import run_damping_estimation
    return run_damping_estimation(cfg)


def parse_command_line():
    parser = argparse.ArgumentParser(description="PINN for the Duffing oscillator")
    parser.add_argument("--mode", choices=EXPERIMENT_MODES,
                        help="experiment to run (default: EXPERIMENT_MODE in config.py)")
    parser.add_argument("--epochs", type=int, help="override EPOCHS from config.py")
    args = parser.parse_args()
    overrides = {}
    if args.mode:
        overrides["EXPERIMENT_MODE"] = args.mode
    if args.epochs:
        overrides["EPOCHS"] = args.epochs
    return overrides


if __name__ == "__main__":
    main(parse_command_line())
