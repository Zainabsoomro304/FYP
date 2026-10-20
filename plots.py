"""
plots.py
========

Plots and the metrics file shared by both experiments. Nothing here affects
training; these functions only draw or save results that already exist.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")                    # save figures to files; no screen needed
import matplotlib.pyplot as plt
import numpy as np

from physics import classify_damping_regime, critical_damping

LOSS_FLOOR = 1e-16                       # zero losses are drawn at this value on log axes


def describe_system(cfg, damping_text=None):
    """One-line summary of the physical setup for plot titles.

    damping_text overrides the "c = ..." part (Mode B, where c is unknown).
    """
    if damping_text is None:
        regime = classify_damping_regime(cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT)
        if cfg.NONLINEAR_COEFFICIENT != 0:
            regime = f"linearised: {regime}"
        damping_text = f"c = {cfg.DAMPING_COEFFICIENT:g} N*s/m ({regime})"
    return (f"m = {cfg.MASS:g} kg, k = {cfg.SPRING_CONSTANT:g} N/m, {damping_text}, "
            f"alpha = {cfg.NONLINEAR_COEFFICIENT:g} N/m^3, x0 = {cfg.INITIAL_POSITION:g} m, "
            f"v0 = {cfg.INITIAL_VELOCITY:g} m/s, t0 = {cfg.TIME_START:g} s")


def _save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_solution_comparison(t, x_pred, x_reference, subtitle, output_dir,
                             reference_label="Numerical reference (SciPy solve_ivp)",
                             filename="duffing_solution_comparison.png"):
    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(t, x_reference, "k-", lw=2, label=reference_label)
    plt.plot(t, x_pred, "--", color="tab:red", lw=2, label=r"PINN prediction $\hat{x}(t)$")
    plt.title("PINN vs numerical reference:  m x'' + c x' + k x + alpha x^3 = 0\n" + subtitle,
              fontsize=9)
    plt.xlabel("Time t [s]")
    plt.ylabel("Position x(t) [m]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    return _save(fig, output_dir, filename)


def plot_absolute_error(t, x_pred, x_reference, subtitle, output_dir,
                        filename="duffing_absolute_error.png"):
    abs_error = np.abs(np.asarray(x_pred) - np.asarray(x_reference))
    i_max = int(np.argmax(abs_error))
    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(t, abs_error, color="tab:purple", lw=1.5, label=r"$|\hat{x}(t) - x_{ref}(t)|$")
    plt.plot(t[i_max], abs_error[i_max], "o", color="tab:red",
             label=f"Max abs error = {abs_error[i_max]:.3e} m at t = {t[i_max]:.3f} s")
    plt.title("Absolute position error (PINN vs numerical reference)\n" + subtitle, fontsize=9)
    plt.xlabel("Time t [s]")
    plt.ylabel("Absolute error [m]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    return _save(fig, output_dir, filename)


def plot_training_losses(history, cfg, subtitle, output_dir, filename="duffing_training_losses.png"):
    """Weighted total loss and the UNWEIGHTED components (data loss only if present)."""
    epochs = np.arange(1, len(history["total"]) + 1)
    has_data = "data" in history
    total_label = f"Total loss (weighted: {cfg.LAMBDA_PHYSICS:g}·L_phys + {cfg.LAMBDA_IC:g}·L_IC"
    total_label += f" + {cfg.LAMBDA_DATA:g}·L_data)" if has_data else ")"

    fig = plt.figure(figsize=(9, 4.8))
    plt.semilogy(epochs, np.maximum(history["total"], LOSS_FLOOR), lw=1.6, color="black",
                 label=total_label)
    plt.semilogy(epochs, np.maximum(history["physics"], LOSS_FLOOR), lw=1.0, color="tab:blue",
                 label="Physics loss L_phys [N^2] (unweighted)")
    plt.semilogy(epochs, np.maximum(history["ic"], LOSS_FLOOR), lw=1.0, color="tab:orange",
                 label="Initial-condition loss L_IC (unweighted)")
    if has_data:
        plt.semilogy(epochs, np.maximum(history["data"], LOSS_FLOOR), lw=1.0, color="tab:green",
                     label="Data loss L_data [m^2] (unweighted)")
    plt.title(f"Training losses (values below {LOSS_FLOOR:g} drawn at {LOSS_FLOOR:g})\n" + subtitle,
              fontsize=9)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    return _save(fig, output_dir, filename)


def plot_physics_residual(t, residual, subtitle, output_dir, filename="physics_residual.png"):
    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(t, residual, color="tab:green", lw=1.2,
             label=r"$r(t) = m\hat{x}'' + c\hat{x}' + k\hat{x} + \alpha\hat{x}^3$")
    plt.axhline(0.0, color="black", lw=0.8, ls="--", label="zero (ODE exactly satisfied)")
    plt.title("Physics residual on the evaluation grid\n" + subtitle, fontsize=9)
    plt.xlabel("Time t [s]")
    plt.ylabel("Residual r(t) [N]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    return _save(fig, output_dir, filename)


def plot_results(t, x_pred, x_reference, residual, history, cfg, subtitle, output_dir):
    """The four standard plots, used by both experiments."""
    plot_solution_comparison(t, x_pred, x_reference, subtitle, output_dir)
    plot_absolute_error(t, x_pred, x_reference, subtitle, output_dir)
    plot_training_losses(history, cfg, subtitle, output_dir)
    plot_physics_residual(t, residual, subtitle, output_dir)


def settings_summary(cfg, n_params, damping_known=True):
    """The configuration part of metrics.json (shared by both experiments)."""
    physical = {
        "MASS_kg": cfg.MASS,
        "SPRING_CONSTANT_N_per_m": cfg.SPRING_CONSTANT,
        "NONLINEAR_COEFFICIENT_N_per_m3": cfg.NONLINEAR_COEFFICIENT,
        "INITIAL_POSITION_m": cfg.INITIAL_POSITION,
        "INITIAL_VELOCITY_m_per_s": cfg.INITIAL_VELOCITY,
    }
    if damping_known:
        c_crit = critical_damping(cfg.MASS, cfg.SPRING_CONSTANT)
        physical.update({
            "DAMPING_COEFFICIENT_N_s_per_m": cfg.DAMPING_COEFFICIENT,
            "linear_damping_regime": classify_damping_regime(cfg.MASS, cfg.SPRING_CONSTANT,
                                                             cfg.DAMPING_COEFFICIENT),
            "linear_natural_frequency_rad_per_s": math.sqrt(cfg.SPRING_CONSTANT / cfg.MASS),
            "critical_damping_N_s_per_m": c_crit,
            "damping_ratio": cfg.DAMPING_COEFFICIENT / c_crit,
        })
    else:
        physical.update({
            "DAMPING_COEFFICIENT": "unknown to the model (estimated)",
            "TRUE_DAMPING_COEFFICIENT_N_s_per_m (evaluation only)": cfg.TRUE_DAMPING_COEFFICIENT,
            "INITIAL_DAMPING_ESTIMATE_N_s_per_m": cfg.INITIAL_DAMPING_ESTIMATE,
        })
    training = {
        "SEED": cfg.SEED,
        "HIDDEN_SIZE": cfg.HIDDEN_SIZE,
        "NUM_HIDDEN_LAYERS": cfg.NUM_HIDDEN_LAYERS,
        "EPOCHS": cfg.EPOCHS,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "LAMBDA_PHYSICS": cfg.LAMBDA_PHYSICS,
        "LAMBDA_IC": cfg.LAMBDA_IC,
        "KEEP_BEST_WEIGHTS": cfg.KEEP_BEST_WEIGHTS,
        "NORMALIZE_TIME_INPUT": cfg.NORMALIZE_TIME_INPUT,
        "trainable_parameters": n_params,
    }
    if not damping_known:
        training["LAMBDA_DATA"] = cfg.LAMBDA_DATA
    return {
        "experiment_mode": cfg.EXPERIMENT_MODE,
        "physical_parameters": physical,
        "simulation_settings": {
            "TIME_START_s": cfg.TIME_START,
            "TIME_END_s": cfg.TIME_END,
            "N_COLLOCATION": cfg.N_COLLOCATION,
            "N_EVALUATION_POINTS": cfg.N_EVALUATION_POINTS,
        },
        "reference_solver": {
            "method": f"scipy.integrate.solve_ivp ({cfg.REFERENCE_SOLVER_METHOD})",
            "rtol": cfg.REFERENCE_RTOL,
            "atol": cfg.REFERENCE_ATOL,
        },
        "training_settings": training,
    }


def save_json(report, output_dir, filename="metrics.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def check_metrics_are_finite(metrics):
    """Raise if any numeric metric is NaN/inf (None is allowed: 'not defined')."""
    for name, value in metrics.items():
        if value is None or isinstance(value, (str, bool)):
            continue
        if not np.isfinite(value):
            raise RuntimeError(f"Metric '{name}' is not finite: {value}")


def print_metrics(metrics, title):
    print(f"\n=== {title} ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:45s} {value:.6e}")
        else:
            print(f"{name:45s} {value}")
