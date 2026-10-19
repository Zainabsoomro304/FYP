"""
main.py
=======

Train a Physics-Informed Neural Network (PINN) to solve the damped
mass-spring equation

        m x''(t) + c x'(t) + k x(t) = 0,
        x(t0) = x0,   x'(t0) = v0,      t in [TIME_START, TIME_END]

then evaluate it against the analytical solution and save plots + metrics.

All settings live in config.py. Run with:   python main.py
"""

import copy
import json
import math
import os
import random
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")                    # save figures to files; no screen needed
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from physics import (analytical_solution, classify_damping_regime, critical_damping,
                     validate_physical_parameters)
from pinn_model import (PINN, compute_initial_condition_loss, compute_physics_loss,
                        compute_physics_residual, predict_with_derivatives)
from visualization import (create_training_visualizations, make_snapshot,
                           make_visualization_points, select_snapshot_epochs)


# ----------------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------------
def load_config(overrides=None):
    """Collect every UPPER_CASE setting from config.py into one object.

    `overrides` (a dict like {"DAMPING_COEFFICIENT": 0.5}) lets scripts and
    tests run other configurations without editing config.py.
    """
    settings = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    for name, value in (overrides or {}).items():
        if name not in settings:
            raise KeyError(f"Unknown setting '{name}'. Valid names: {sorted(settings)}")
        settings[name] = value
    cfg = SimpleNamespace(**settings)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Fail early with a clear message instead of training on a bad configuration."""
    validate_physical_parameters(cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT)
    for name in ["INITIAL_POSITION", "INITIAL_VELOCITY", "TIME_START", "TIME_END",
                 "LEARNING_RATE", "LAMBDA_PHYSICS", "LAMBDA_IC"]:
        if not math.isfinite(getattr(cfg, name)):
            raise ValueError(f"{name} must be finite, got {getattr(cfg, name)!r}")
    if not cfg.TIME_END > cfg.TIME_START:
        raise ValueError(f"TIME_END ({cfg.TIME_END}) must be greater than TIME_START ({cfg.TIME_START})")
    for name, minimum in [("N_COLLOCATION", 2), ("N_EVALUATION_POINTS", 2), ("EPOCHS", 1),
                          ("HIDDEN_SIZE", 1), ("NUM_HIDDEN_LAYERS", 1), ("PRINT_EVERY", 1),
                          ("VISUALIZATION_INTERVAL", 1), ("VISUALIZATION_POINTS", 2),
                          ("ANIMATION_INTERVAL_MS", 1)]:
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if cfg.LEARNING_RATE <= 0 or cfg.LAMBDA_PHYSICS < 0 or cfg.LAMBDA_IC < 0:
        raise ValueError("LEARNING_RATE must be > 0 and the LAMBDA weights must be >= 0")
    if cfg.ANIMATION_FORMAT not in ("gif", "mp4"):
        raise ValueError(f"ANIMATION_FORMAT must be 'gif' or 'mp4', got {cfg.ANIMATION_FORMAT!r}")


def set_seed(seed: int) -> None:
    """Make runs repeatable (same initial weights -> same training on the same machine)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg):
    if cfg.NORMALIZE_TIME_INPUT:
        return PINN(cfg.HIDDEN_SIZE, cfg.NUM_HIDDEN_LAYERS, t_min=cfg.TIME_START, t_max=cfg.TIME_END)
    return PINN(cfg.HIDDEN_SIZE, cfg.NUM_HIDDEN_LAYERS)


def make_collocation_points(cfg):
    """N_COLLOCATION evenly spaced times in [TIME_START, TIME_END], shape (N_f, 1).

    They carry NO target values -- the ODE itself is the "label".
    requires_grad=True lets autograd compute d/dt of the network output.
    The same points are reused every epoch (fixed during one training run).
    """
    t = torch.linspace(cfg.TIME_START, cfg.TIME_END, cfg.N_COLLOCATION).reshape(-1, 1)
    return t.requires_grad_(True)


def compute_training_losses(model, t_collocation, cfg):
    """The ONLY quantities the optimiser sees: physics residual + initial conditions.

    No analytical-solution values appear here.
    """
    loss_physics = compute_physics_loss(model, t_collocation, cfg.MASS,
                                        cfg.DAMPING_COEFFICIENT, cfg.SPRING_CONSTANT)
    loss_ic, position_error, velocity_error = compute_initial_condition_loss(
        model, cfg.TIME_START, cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY)
    loss_total = (
        cfg.LAMBDA_PHYSICS * loss_physics
        + cfg.LAMBDA_IC * loss_ic
    )
    return {
        "total": loss_total,
        "physics": loss_physics,
        "ic": loss_ic,
        "ic_position": position_error ** 2,     # (x_hat(t0) - x0)^2   [m^2]
        "ic_velocity": velocity_error ** 2,     # (x_hat'(t0) - v0)^2  [m^2/s^2]
    }


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_pinn(model, cfg):
    """Minimise  L_total = lambda_f * L_physics + lambda_IC * L_IC  with Adam.

    Returns (history, best_epoch, snapshots). `snapshots` is an empty list when
    ENABLE_TRAINING_VISUALIZATION is False; otherwise it holds the predicted
    curve (a NumPy array, not the weights) at selected epochs.
    """
    t_collocation = make_collocation_points(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    history = {"total": [], "physics": [], "ic": [], "ic_position": [], "ic_velocity": []}
    best_loss, best_state, best_epoch = float("inf"), None, None

    snapshots = []
    if cfg.ENABLE_TRAINING_VISUALIZATION:
        snapshot_epochs = select_snapshot_epochs(cfg.EPOCHS, cfg.VISUALIZATION_INTERVAL)
        t_visual = make_visualization_points(cfg)

    for epoch in range(1, cfg.EPOCHS + 1):
        optimizer.zero_grad()                         # clear gradients from the previous step

        # All losses are recomputed every iteration with the current weights,
        # at the same fixed collocation points.
        losses = compute_training_losses(model, t_collocation, cfg)
        loss_total = losses["total"]

        for name, value in losses.items():
            if not torch.isfinite(value):
                raise RuntimeError(
                    f"Training loss '{name}' became non-finite ({value.item()}) at epoch {epoch}. "
                    "Try a smaller LEARNING_RATE, a shorter interval, or different LAMBDA weights.")

        # Record the losses for these weights (before the update below changes them).
        for name, value in losses.items():
            history[name].append(value.item())
        if cfg.KEEP_BEST_WEIGHTS and loss_total.item() < best_loss:
            best_loss, best_epoch = loss_total.item(), epoch
            best_state = copy.deepcopy(model.state_dict())

        # Snapshot for the visualization. At this point `epoch - 1` optimizer
        # updates have been applied, so these are "epoch (epoch - 1)" weights in
        # the snapshot convention (epoch 0 = untrained network). We reuse the
        # losses just computed and only run an extra no-grad forward pass.
        if cfg.ENABLE_TRAINING_VISUALIZATION and (epoch - 1) in snapshot_epochs:
            snapshots.append(make_snapshot(epoch - 1, "training", model, t_visual, losses))

        loss_total.backward()                         # backprop: d(loss)/d(every weight and bias)

        if epoch == 1:                                # sanity check: the network receives gradients
            grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
            print(f"Gradient norm at epoch 1: {grad_norm.item():.4e} (must be > 0)")
            assert grad_norm.item() > 0, "No gradients reached the network!"

        optimizer.step()                              # Adam updates the parameters

        if epoch % cfg.PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch {epoch:6d} | total {loss_total.item():.4e} | "
                  f"physics {losses['physics'].item():.4e} | IC {losses['ic'].item():.4e}")

    if cfg.ENABLE_TRAINING_VISUALIZATION:
        # The weights after the LAST update have no recorded loss yet, so compute
        # it here (needs autograd for the derivatives; no backward, no update).
        final_losses = compute_training_losses(model, t_collocation, cfg)
        snapshots.append(make_snapshot(cfg.EPOCHS, "final", model, t_visual,
                                       {k: v.item() for k, v in final_losses.items()}))

    if cfg.KEEP_BEST_WEIGHTS and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored weights from epoch {best_epoch} "
              f"(lowest total training loss = {best_loss:.4e})")
        if cfg.ENABLE_TRAINING_VISUALIZATION:
            # Snapshot of the RESTORED weights, kept separate from the final ones.
            # Loop epoch `best_epoch` saw the weights after best_epoch - 1 updates.
            best_index = best_epoch - 1
            snapshots.append(make_snapshot(best_index, "best", model, t_visual,
                                           {name: history[name][best_index]
                                            for name in ("total", "physics", "ic")}))
    else:
        best_epoch = cfg.EPOCHS                       # the final weights are the ones used

    return history, best_epoch, snapshots


# ----------------------------------------------------------------------------
# Evaluation (the analytical solution is used from here on, never before)
# ----------------------------------------------------------------------------
def evaluate_pinn(model, cfg):
    """Compare the trained PINN with the analytical solution. Evaluation ONLY."""
    # Dense grid, shape (N_EVAL, 1). requires_grad so we can also check the ODE residual.
    t_eval = torch.linspace(cfg.TIME_START, cfg.TIME_END, cfg.N_EVALUATION_POINTS).reshape(-1, 1)
    t_eval.requires_grad_(True)
    x_pred, _, _ = predict_with_derivatives(model, t_eval)
    residual = compute_physics_residual(model, t_eval, cfg.MASS,
                                        cfg.DAMPING_COEFFICIENT, cfg.SPRING_CONSTANT)

    # Initial conditions as learned by the network (at TIME_START).
    _, position_error, velocity_error = compute_initial_condition_loss(
        model, cfg.TIME_START, cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY)

    # Training losses of the final (possibly restored) weights, on the collocation points.
    final_losses = compute_training_losses(model, make_collocation_points(cfg), cfg)

    # Convert to NumPy (detach = leave the autograd graph).
    t_np = t_eval.detach().numpy().ravel()
    x_np = x_pred.detach().numpy().ravel()
    residual_np = residual.detach().numpy().ravel()

    # The analytical solution appears HERE for the first time -- never in training.
    x_exact = analytical_solution(t_np, cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT,
                                  cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY, cfg.TIME_START)
    error = x_np - x_exact

    metrics = {
        "mse_position_m2": float(np.mean(error ** 2)),
        "rmse_position_m": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_position_error_m": float(np.max(np.abs(error))),
        "physics_residual_mse_eval_grid_N2": float(np.mean(residual_np.astype(np.float64) ** 2)),
        "initial_position_error_m": abs(position_error.item()),
        "initial_velocity_error_m_per_s": abs(velocity_error.item()),
        "final_model_loss_total_weighted": final_losses["total"].item(),
        "final_model_loss_physics_N2": final_losses["physics"].item(),
        "final_model_loss_ic": final_losses["ic"].item(),
        "final_model_loss_ic_position_m2": final_losses["ic_position"].item(),
        "final_model_loss_ic_velocity_m2_per_s2": final_losses["ic_velocity"].item(),
    }
    return t_np, x_np, x_exact, residual_np, metrics


# ----------------------------------------------------------------------------
# Plots and output files
# ----------------------------------------------------------------------------
def describe_system(cfg):
    regime = classify_damping_regime(cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT)
    return (f"m = {cfg.MASS:g} kg, k = {cfg.SPRING_CONSTANT:g} N/m, "
            f"c = {cfg.DAMPING_COEFFICIENT:g} N*s/m ({regime}), "
            f"x0 = {cfg.INITIAL_POSITION:g} m, v0 = {cfg.INITIAL_VELOCITY:g} m/s, "
            f"t0 = {cfg.TIME_START:g} s")


def plot_results(t, x_pred, x_exact, residual, history, cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    subtitle = describe_system(cfg)

    # 1. Solution comparison
    plt.figure(figsize=(9, 4.8))
    plt.plot(t, x_exact, "k-", lw=2, label="Analytical solution x(t)")
    plt.plot(t, x_pred, "--", color="tab:red", lw=2, label=r"PINN prediction $\hat{x}(t)$")
    plt.title("PINN vs analytical solution:  m x'' + c x' + k x = 0\n" + subtitle, fontsize=10)
    plt.xlabel("Time t [s]")
    plt.ylabel("Position x(t) [m]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "solution_comparison.png"), dpi=150)
    plt.close()

    # 2. Training losses (log scale). Clip to a tiny floor so log(0) never happens.
    epochs = np.arange(1, len(history["total"]) + 1)
    floor = 1e-16
    plt.figure(figsize=(9, 4.8))
    plt.semilogy(epochs, np.maximum(history["total"], floor), lw=1.6, color="black",
                 label=rf"Total loss (weighted: {cfg.LAMBDA_PHYSICS:g}·L_phys + {cfg.LAMBDA_IC:g}·L_IC)")
    plt.semilogy(epochs, np.maximum(history["physics"], floor), lw=1.0, color="tab:blue",
                 label="Physics loss L_phys (unweighted)")
    plt.semilogy(epochs, np.maximum(history["ic"], floor), lw=1.0, color="tab:orange",
                 label="Initial-condition loss L_IC (unweighted)")
    plt.title(f"Training losses (values below {floor:g} drawn at {floor:g})\n" + subtitle, fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_losses.png"), dpi=150)
    plt.close()

    # 3. Absolute error, with the maximum marked
    abs_error = np.abs(x_pred - x_exact)
    i_max = int(np.argmax(abs_error))
    plt.figure(figsize=(9, 4.8))
    plt.plot(t, abs_error, color="tab:purple", lw=1.5, label=r"$|\hat{x}(t) - x(t)|$")
    plt.plot(t[i_max], abs_error[i_max], "o", color="tab:red",
             label=f"Max abs error = {abs_error[i_max]:.3e} m at t = {t[i_max]:.3f} s")
    plt.title("Absolute position error\n" + subtitle, fontsize=10)
    plt.xlabel("Time t [s]")
    plt.ylabel("Absolute error [m]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "absolute_error.png"), dpi=150)
    plt.close()

    # 4. Physics residual on the evaluation grid
    plt.figure(figsize=(9, 4.8))
    plt.plot(t, residual, color="tab:green", lw=1.2,
             label=r"$r(t) = m\hat{x}'' + c\hat{x}' + k\hat{x}$")
    plt.axhline(0.0, color="black", lw=0.8, ls="--", label="zero (ODE exactly satisfied)")
    plt.title("Physics residual on the evaluation grid\n" + subtitle, fontsize=10)
    plt.xlabel("Time t [s]")
    plt.ylabel("Residual r(t) [N]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "physics_residual.png"), dpi=150)
    plt.close()


def save_metrics(metrics, cfg, best_epoch, training_time_s, n_params, visualization_info=None):
    """Write metrics.json with the configuration that produced the results."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    report = {
        "physical_parameters": {
            "MASS_kg": cfg.MASS,
            "SPRING_CONSTANT_N_per_m": cfg.SPRING_CONSTANT,
            "DAMPING_COEFFICIENT_N_s_per_m": cfg.DAMPING_COEFFICIENT,
            "INITIAL_POSITION_m": cfg.INITIAL_POSITION,
            "INITIAL_VELOCITY_m_per_s": cfg.INITIAL_VELOCITY,
            "damping_regime": classify_damping_regime(cfg.MASS, cfg.SPRING_CONSTANT,
                                                      cfg.DAMPING_COEFFICIENT),
            "natural_frequency_rad_per_s": math.sqrt(cfg.SPRING_CONSTANT / cfg.MASS),
            "critical_damping_N_s_per_m": critical_damping(cfg.MASS, cfg.SPRING_CONSTANT),
            "damping_ratio": cfg.DAMPING_COEFFICIENT / critical_damping(cfg.MASS, cfg.SPRING_CONSTANT),
        },
        "simulation_settings": {
            "TIME_START_s": cfg.TIME_START,
            "TIME_END_s": cfg.TIME_END,
            "N_COLLOCATION": cfg.N_COLLOCATION,
            "N_EVALUATION_POINTS": cfg.N_EVALUATION_POINTS,
        },
        "training_settings": {
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
        },
        "results": {
            "best_epoch": best_epoch,
            "training_time_s": training_time_s,
        },
        "metrics": metrics,
        "training_visualization": visualization_info or {"enabled": False},
        "notes": "Metrics are computed AFTER training; the analytical solution is never used "
                 "in the training loss. 'final_model_*' losses are the training losses of the "
                 "weights actually used (restored best weights if KEEP_BEST_WEIGHTS). Physics "
                 "residual is in newtons (N), so physics losses are in N^2.",
    }
    path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(overrides=None):
    cfg = load_config(overrides)
    set_seed(cfg.SEED)
    torch.set_default_dtype(torch.float32)

    print("System:", describe_system(cfg))
    print(f"Time interval: [{cfg.TIME_START:g}, {cfg.TIME_END:g}] s | "
          f"collocation points: {cfg.N_COLLOCATION} | evaluation points: {cfg.N_EVALUATION_POINTS}")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params} trainable parameters")

    initial_params = [p.detach().clone() for p in model.parameters()]

    start = time.time()
    history, best_epoch, snapshots = train_pinn(model, cfg)
    training_time_s = time.time() - start
    print(f"Training took {training_time_s:.1f} s")

    # Check that training actually changed the parameters.
    change = sum((p.detach() - p0).abs().sum().item()
                 for p, p0 in zip(model.parameters(), initial_params))
    print(f"Total absolute change in parameters: {change:.4e} (must be > 0)")

    t, x_pred, x_exact, residual, metrics = evaluate_pinn(model, cfg)
    metrics["last_epoch_loss_total_weighted"] = history["total"][-1]
    metrics["last_epoch_loss_physics_N2"] = history["physics"][-1]
    metrics["last_epoch_loss_ic"] = history["ic"][-1]

    for name, value in metrics.items():
        if not math.isfinite(value):
            raise RuntimeError(f"Metric '{name}' is not finite: {value}")

    print("\n=== Evaluation metrics (computed AFTER training; not training objectives) ===")
    for name, value in metrics.items():
        print(f"{name:42s} {value:.6e}")
    print(f"{'best_epoch':42s} {best_epoch}")

    plot_results(t, x_pred, x_exact, residual, history, cfg)

    visualization_info = None
    if cfg.ENABLE_TRAINING_VISUALIZATION:
        print("\nCreating training visualizations ...")
        visualization_info = create_training_visualizations(snapshots, history, cfg,
                                                            describe_system(cfg))
        visualization_info["enabled"] = True

    path = save_metrics(metrics, cfg, best_epoch, training_time_s, n_params, visualization_info)
    print(f"\nPlots and metrics saved in: {cfg.OUTPUT_DIR}  ({os.path.basename(path)})")
    return metrics


if __name__ == "__main__":
    main()
