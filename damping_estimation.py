"""
damping_estimation.py
=====================

Mode B: estimate an UNKNOWN damping coefficient from measured positions.

What the estimation model knows
-------------------------------
* the form of the equation   m x'' + c x' + k x + alpha x^3 = 0
* MASS, SPRING_CONSTANT, NONLINEAR_COEFFICIENT, INITIAL_POSITION, INITIAL_VELOCITY
* N_OBSERVATION_POINTS measured positions (t_j, x_obs_j)

What it does NOT know
---------------------
* the damping coefficient c. It starts from INITIAL_DAMPING_ESTIMATE and
  learns c together with the network weights, using
        L = lambda_f L_physics(c_est) + lambda_IC L_IC + lambda_d L_data.

Why the true value is kept out of the estimation
------------------------------------------------
If TRUE_DAMPING_COEFFICIENT appeared in the model or the loss, the "estimate"
could simply copy it and the experiment would prove nothing. So the true value
is used in exactly two places, both OUTSIDE training:
    1. generate_synthetic_observations()  -- to simulate the measurements
    2. the evaluation after training      -- to measure the estimation error
Training receives a configuration from which the true value (and the Mode-A
DAMPING_COEFFICIENT) have been removed; any accidental use would crash.
"""

import csv
import math
import os
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from physics import make_reference_solver
from plots import (check_metrics_are_finite, describe_system, plot_results, print_metrics,
                   save_json, settings_summary)
from training import (build_model, evaluate_pinn, make_evaluation_points, set_seed, train_pinn)
from visualization import create_training_visualizations

# Settings that the estimation model must never see.
HIDDEN_FROM_ESTIMATION = ("TRUE_DAMPING_COEFFICIENT", "DAMPING_COEFFICIENT")


# ----------------------------------------------------------------------------
# Synthetic measurements
# ----------------------------------------------------------------------------
def generate_synthetic_observations(cfg):
    """Simulate position measurements with the HIDDEN true damping coefficient.

    Returns (t_observed, x_observed, x_noise_free), all NumPy arrays of length
    N_OBSERVATION_POINTS. The times are evenly spaced over [TIME_START, TIME_END].
    Noise (if enabled) is Gaussian with std OBSERVATION_NOISE_STD and a fixed
    seed (SEED), so the same settings always give the same data.
    """
    t_observed = np.linspace(cfg.TIME_START, cfg.TIME_END, cfg.N_OBSERVATION_POINTS)
    true_solution = make_reference_solver(cfg, cfg.TRUE_DAMPING_COEFFICIENT)
    x_noise_free = true_solution(t_observed)
    if cfg.ADD_OBSERVATION_NOISE:
        rng = np.random.default_rng(cfg.SEED)
        noise = rng.normal(0.0, cfg.OBSERVATION_NOISE_STD, size=t_observed.shape)
    else:
        noise = np.zeros_like(t_observed)
    return t_observed, x_noise_free + noise, x_noise_free


def make_known_parameter_config(cfg):
    """Copy of cfg WITHOUT the damping values -- this is all the estimation model gets."""
    known = {name: value for name, value in vars(cfg).items() if name not in HIDDEN_FROM_ESTIMATION}
    return SimpleNamespace(**known)


def observations_to_tensors(t_observed, x_observed):
    """NumPy arrays -> float32 tensors of shape (N_obs, 1). No gradient w.r.t. time is needed."""
    as_column = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).reshape(-1, 1)
    return as_column(t_observed), as_column(x_observed)


def save_observations_csv(t_observed, x_observed, x_noise_free, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "observations.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "observed_position_m", "noise_free_position_m"])
        for row in zip(t_observed, x_observed, x_noise_free):
            writer.writerow([f"{value:.12g}" for value in row])
    return path


# ----------------------------------------------------------------------------
# Mode-B specific plots
# ----------------------------------------------------------------------------
def plot_damping_estimation_history(history, best_epoch, true_damping, initial_damping,
                                    subtitle, output_dir):
    """Estimated c per epoch. The true value is drawn ONLY as a reference line."""
    estimates = np.asarray(history["damping_estimate"])
    epochs = np.arange(1, len(estimates) + 1)
    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, estimates, color="tab:blue", lw=1.6, label="Estimated c (trainable)")
    plt.axhline(true_damping, color="black", ls="--", lw=1.2,
                label=f"True c = {true_damping:g} N*s/m (hidden from training; reference only)")
    plt.axhline(initial_damping, color="gray", ls=":", lw=1.0,
                label=f"Initial estimate = {initial_damping:g} N*s/m")
    plt.plot(best_epoch, estimates[best_epoch - 1], "o", color="tab:red",
             label=f"Restored best epoch {best_epoch}: c = {estimates[best_epoch - 1]:.4f} N*s/m")
    plt.title("Damping coefficient estimate during training\n" + subtitle, fontsize=9)
    plt.xlabel("Epoch")
    plt.ylabel("Damping coefficient c [N*s/m]")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(output_dir, "damping_estimation_history.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_observations(t_observed, x_observed, t, x_reference, x_pred, noisy, subtitle, output_dir):
    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(t, x_reference, "k-", lw=1.8, label="Numerical reference (true c)")
    plt.plot(t, x_pred, "--", color="tab:red", lw=1.8, label="Trained PINN (estimated c)")
    plt.plot(t_observed, x_observed, "o", color="tab:blue", ms=4,
             label=f"Observations ({len(t_observed)} points{', noisy' if noisy else ''})")
    plt.title("Observed positions, numerical reference and trained PINN\n" + subtitle, fontsize=9)
    plt.xlabel("Time t [s]")
    plt.ylabel("Position x(t) [m]")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(output_dir, "damping_observations.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
def run_damping_estimation(cfg):
    output_dir = os.path.join(cfg.OUTPUT_DIR, "damping_estimation")
    set_seed(cfg.SEED)
    torch.set_default_dtype(torch.float32)

    # ---- 1. Synthetic measurements (the ONLY use of the true value before evaluation)
    t_observed, x_observed, x_noise_free = generate_synthetic_observations(cfg)
    csv_path = save_observations_csv(t_observed, x_observed, x_noise_free, output_dir)
    observations = observations_to_tensors(t_observed, x_observed)

    # ---- 2. Everything the estimation model is allowed to know
    known_cfg = make_known_parameter_config(cfg)
    if math.isclose(cfg.INITIAL_DAMPING_ESTIMATE, cfg.TRUE_DAMPING_COEFFICIENT):
        print("WARNING: INITIAL_DAMPING_ESTIMATE equals TRUE_DAMPING_COEFFICIENT; the "
              "experiment cannot show that the estimate was learned.")

    subtitle_unknown = describe_system(known_cfg, damping_text="c = unknown (estimated)")
    print("Mode B: damping estimation")
    print("System:", subtitle_unknown)
    print(f"Observations: {cfg.N_OBSERVATION_POINTS} points, noise "
          f"{'std ' + format(cfg.OBSERVATION_NOISE_STD, 'g') + ' m' if cfg.ADD_OBSERVATION_NOISE else 'off'}"
          f" | collocation points: {cfg.N_COLLOCATION} | evaluation points: {cfg.N_EVALUATION_POINTS}")
    print(f"Initial damping estimate: {cfg.INITIAL_DAMPING_ESTIMATE:g} N*s/m")

    model = build_model(known_cfg, learn_damping=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params} trainable parameters (network + 1 damping parameter)")
    initial_estimate = model.damping_coefficient().item()

    # ---- 3. Train: physics (with c_est) + ICs + data. No true value available here.
    start = time.time()
    history, best_epoch, snapshots = train_pinn(model, known_cfg, observations)
    training_time_s = time.time() - start
    estimated_damping = model.damping_coefficient().item()     # restored best weights
    print(f"Training took {training_time_s:.1f} s")
    print(f"Damping estimate moved from {initial_estimate:.4f} to {estimated_damping:.4f} N*s/m")

    # ---- 4. Evaluation: NOW the hidden true value may be used.
    true_damping = cfg.TRUE_DAMPING_COEFFICIENT
    true_solution = make_reference_solver(cfg, true_damping)
    t_eval = make_evaluation_points(cfg).numpy().astype(np.float64)
    x_reference = true_solution(t_eval)
    t, x_pred, residual, metrics = evaluate_pinn(model, known_cfg, x_reference, observations)

    with torch.no_grad():
        x_pred_at_obs = model(observations[0]).numpy().ravel()
    absolute_error = abs(estimated_damping - true_damping)
    metrics = {
        "true_damping_coefficient_N_s_per_m": true_damping,
        "initial_damping_estimate_N_s_per_m": initial_estimate,
        "estimated_damping_coefficient_N_s_per_m": estimated_damping,
        "damping_estimate_at_last_epoch_N_s_per_m": history["damping_estimate"][-1],
        "damping_absolute_error_N_s_per_m": absolute_error,
        # Relative error is undefined when the true value is 0.
        "damping_relative_error": absolute_error / true_damping if true_damping > 0 else None,
        "data_mse_at_observations_m2": float(np.mean((x_pred_at_obs - x_observed) ** 2)),
        "position_mse_at_observation_times_vs_noise_free_m2":
            float(np.mean((x_pred_at_obs - x_noise_free) ** 2)),
        **metrics,
        "last_epoch_loss_total_weighted": history["total"][-1],
        "last_epoch_loss_physics_N2": history["physics"][-1],
        "last_epoch_loss_ic": history["ic"][-1],
        "last_epoch_loss_data_m2": history["data"][-1],
    }
    check_metrics_are_finite(metrics)
    print_metrics(metrics, "Evaluation (computed AFTER training; true c used only here)")
    print(f"{'best_epoch':45s} {best_epoch}")

    # ---- 5. Plots and files
    subtitle = describe_system(known_cfg, damping_text=f"c true = {true_damping:g}, "
                                                       f"estimated = {estimated_damping:.4f} N*s/m")
    plot_results(t, x_pred, x_reference, residual, history, known_cfg, subtitle, output_dir)
    plot_damping_estimation_history(history, best_epoch, true_damping, cfg.INITIAL_DAMPING_ESTIMATE,
                                    subtitle, output_dir)
    plot_observations(t_observed, x_observed, t, x_reference, x_pred, cfg.ADD_OBSERVATION_NOISE,
                      subtitle, output_dir)

    visualization_info = {"enabled": False}
    if cfg.ENABLE_TRAINING_VISUALIZATION:
        print("\nCreating training visualizations ...")
        visualization_info = create_training_visualizations(snapshots, history, known_cfg, subtitle,
                                                            true_solution, output_dir)
        visualization_info["enabled"] = True

    report = settings_summary(cfg, n_params, damping_known=False)
    report["observations"] = {
        "N_OBSERVATION_POINTS": cfg.N_OBSERVATION_POINTS,
        "ADD_OBSERVATION_NOISE": cfg.ADD_OBSERVATION_NOISE,
        "OBSERVATION_NOISE_STD_m": cfg.OBSERVATION_NOISE_STD,
        "file": os.path.basename(csv_path),
    }
    report["results"] = {"best_epoch": best_epoch, "training_time_s": training_time_s}
    report["metrics"] = metrics
    report["training_visualization"] = visualization_info
    report["notes"] = (
        "TRUE_DAMPING_COEFFICIENT was used only to generate the observations and to compute the "
        "damping errors after training; training received a configuration without it. "
        "'estimated_damping_coefficient' belongs to the restored best weights (lowest TOTAL "
        "training loss). Reference = scipy solve_ivp with the true c (a numerical approximation).")
    path = save_json(report, output_dir)
    print(f"\nPlots, observations and metrics saved in: {output_dir}  ({os.path.basename(path)})")
    return metrics
