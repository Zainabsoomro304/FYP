"""
training.py
===========

The training engine shared by both experiments:

    Mode A  physics_only        L = lambda_f L_physics + lambda_IC L_IC
    Mode B  damping_estimation  L = lambda_f L_physics + lambda_IC L_IC + lambda_d L_data

How the losses guide training
-----------------------------
* L_physics pushes the network towards ANY curve that obeys the equation of
  motion (there are infinitely many: one per initial condition).
* L_IC selects the one curve that starts at x0 with velocity v0.
* L_data (Mode B only) pulls the curve through the measured positions. Since
  the damping coefficient c inside L_physics is trainable in Mode B, the only
  way to satisfy the equation AND match the measurements is for c to move
  towards the value that actually produced the measurements.

Nothing in this file uses a reference solution: no analytical or SciPy
solution is imported here.
"""

import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import torch

import config
from physics import validate_physical_parameters
from pinn_model import (PINN, TrainableDampingPINN, compute_data_loss,
                        compute_initial_condition_loss, compute_physics_loss,
                        compute_physics_residual, predict_with_derivatives)
from visualization import make_snapshot, make_visualization_points, select_snapshot_epochs

EXPERIMENT_MODES = ("physics_only", "damping_estimation")


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
    if cfg.EXPERIMENT_MODE not in EXPERIMENT_MODES:
        raise ValueError(f"EXPERIMENT_MODE must be one of {EXPERIMENT_MODES}, got {cfg.EXPERIMENT_MODE!r}")
    validate_physical_parameters(cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT)
    for name in ["NONLINEAR_COEFFICIENT", "INITIAL_POSITION", "INITIAL_VELOCITY", "TIME_START",
                 "TIME_END", "LEARNING_RATE", "LAMBDA_PHYSICS", "LAMBDA_IC", "LAMBDA_DATA",
                 "REFERENCE_RTOL", "REFERENCE_ATOL", "TRUE_DAMPING_COEFFICIENT",
                 "INITIAL_DAMPING_ESTIMATE", "OBSERVATION_NOISE_STD"]:
        value = getattr(cfg, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not cfg.TIME_END > cfg.TIME_START:
        raise ValueError(f"TIME_END ({cfg.TIME_END}) must be greater than TIME_START ({cfg.TIME_START})")
    for name, minimum in [("N_COLLOCATION", 2), ("N_EVALUATION_POINTS", 2), ("EPOCHS", 1),
                          ("HIDDEN_SIZE", 1), ("NUM_HIDDEN_LAYERS", 1), ("PRINT_EVERY", 1),
                          ("VISUALIZATION_INTERVAL", 1), ("VISUALIZATION_POINTS", 2),
                          ("ANIMATION_INTERVAL_MS", 1), ("N_OBSERVATION_POINTS", 1)]:
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if cfg.LEARNING_RATE <= 0 or min(cfg.LAMBDA_PHYSICS, cfg.LAMBDA_IC, cfg.LAMBDA_DATA) < 0:
        raise ValueError("LEARNING_RATE must be > 0 and the LAMBDA weights must be >= 0")
    if cfg.REFERENCE_RTOL <= 0 or cfg.REFERENCE_ATOL <= 0:
        raise ValueError("REFERENCE_RTOL and REFERENCE_ATOL must be > 0")
    if cfg.ANIMATION_FORMAT not in ("gif", "mp4"):
        raise ValueError(f"ANIMATION_FORMAT must be 'gif' or 'mp4', got {cfg.ANIMATION_FORMAT!r}")
    if cfg.TRUE_DAMPING_COEFFICIENT < 0:
        raise ValueError(f"TRUE_DAMPING_COEFFICIENT must be >= 0, got {cfg.TRUE_DAMPING_COEFFICIENT}")
    if cfg.INITIAL_DAMPING_ESTIMATE <= 0:
        raise ValueError("INITIAL_DAMPING_ESTIMATE must be > 0 (the estimate is kept positive "
                         f"with softplus), got {cfg.INITIAL_DAMPING_ESTIMATE}")
    if cfg.OBSERVATION_NOISE_STD < 0:
        raise ValueError(f"OBSERVATION_NOISE_STD must be >= 0, got {cfg.OBSERVATION_NOISE_STD}")


def set_seed(seed: int) -> None:
    """Make runs repeatable (same initial weights -> same training on the same machine)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, learn_damping=False):
    """PINN for Mode A, TrainableDampingPINN (starting at INITIAL_DAMPING_ESTIMATE) for Mode B."""
    scaling = {"t_min": cfg.TIME_START, "t_max": cfg.TIME_END} if cfg.NORMALIZE_TIME_INPUT else {}
    if learn_damping:
        return TrainableDampingPINN(cfg.HIDDEN_SIZE, cfg.NUM_HIDDEN_LAYERS,
                                    initial_damping=cfg.INITIAL_DAMPING_ESTIMATE, **scaling)
    return PINN(cfg.HIDDEN_SIZE, cfg.NUM_HIDDEN_LAYERS, **scaling)


def make_collocation_points(cfg):
    """N_COLLOCATION evenly spaced times in [TIME_START, TIME_END], shape (N_f, 1).

    They carry NO target values -- the ODE itself is the "label".
    requires_grad=True lets autograd compute d/dt of the network output.
    The same points are reused every epoch (fixed during one training run).
    """
    t = torch.linspace(cfg.TIME_START, cfg.TIME_END, cfg.N_COLLOCATION).reshape(-1, 1)
    return t.requires_grad_(True)


def damping_used_in_residual(model, cfg):
    """Which damping coefficient goes into the physics residual?

    Mode A (PINN): the KNOWN value cfg.DAMPING_COEFFICIENT.
    Mode B (TrainableDampingPINN): the model's own trainable ESTIMATE. The cfg
    passed in Mode B does not even contain the true value, so it cannot leak.
    """
    if model.learns_damping:
        return model.damping_coefficient()
    return cfg.DAMPING_COEFFICIENT


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def compute_training_losses(model, t_collocation, cfg, observations=None):
    """The ONLY quantities the optimiser sees.

    observations=None (Mode A): physics residual + initial conditions only.
    observations=(t_obs, x_obs) (Mode B): additionally the measured positions.
    No reference-solution values appear here.
    """
    loss_physics = compute_physics_loss(model, t_collocation, cfg.MASS,
                                        damping_used_in_residual(model, cfg),
                                        cfg.SPRING_CONSTANT, cfg.NONLINEAR_COEFFICIENT)
    loss_ic, position_error, velocity_error = compute_initial_condition_loss(
        model, cfg.TIME_START, cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY)
    loss_total = (
        cfg.LAMBDA_PHYSICS * loss_physics
        + cfg.LAMBDA_IC * loss_ic
    )
    losses = {
        "physics": loss_physics,
        "ic": loss_ic,
        "ic_position": position_error ** 2,     # (x_hat(t0) - x0)^2   [m^2]
        "ic_velocity": velocity_error ** 2,     # (x_hat'(t0) - v0)^2  [m^2/s^2]
    }
    if observations is not None:
        t_observed, x_observed = observations
        loss_data = compute_data_loss(model, t_observed, x_observed)
        loss_total = loss_total + cfg.LAMBDA_DATA * loss_data
        losses["data"] = loss_data
    losses["total"] = loss_total
    return losses


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_pinn(model, cfg, observations=None):
    """Minimise the total loss with Adam.

    Mode A: train_pinn(PINN, cfg)                         -> physics + IC
    Mode B: train_pinn(TrainableDampingPINN, known_cfg, (t_obs, x_obs))
            -> physics + IC + data; the damping estimate is trained too, because
               it is one of model.parameters().

    Returns (history, best_epoch, snapshots). `snapshots` is an empty list when
    ENABLE_TRAINING_VISUALIZATION is False; otherwise it holds the predicted
    curve (a NumPy array, not the weights) at selected epochs.
    """
    t_collocation = make_collocation_points(cfg)
    # model.parameters() contains the network weights and biases and, for a
    # TrainableDampingPINN, also raw_damping -- so Adam updates all of them.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    history = {}
    best_loss, best_state, best_epoch = float("inf"), None, None

    snapshots = []
    if cfg.ENABLE_TRAINING_VISUALIZATION:
        snapshot_epochs = select_snapshot_epochs(cfg.EPOCHS, cfg.VISUALIZATION_INTERVAL)
        t_visual = make_visualization_points(cfg)

    for epoch in range(1, cfg.EPOCHS + 1):
        optimizer.zero_grad()                         # clear gradients from the previous step

        # All losses are recomputed every iteration with the current weights,
        # at the same fixed collocation points.
        losses = compute_training_losses(model, t_collocation, cfg, observations)
        loss_total = losses["total"]

        for name, value in losses.items():
            if not torch.isfinite(value):
                raise RuntimeError(
                    f"Training loss '{name}' became non-finite ({value.item()}) at epoch {epoch}. "
                    "Try a smaller LEARNING_RATE, a shorter interval, or different LAMBDA weights.")

        # Record the losses for these weights (before the update below changes them).
        for name, value in losses.items():
            history.setdefault(name, []).append(value.item())
        if model.learns_damping:
            history.setdefault("damping_estimate", []).append(model.damping_coefficient().item())
        if cfg.KEEP_BEST_WEIGHTS and loss_total.item() < best_loss:
            best_loss, best_epoch = loss_total.item(), epoch
            # state_dict() includes raw_damping, so the damping estimate is
            # checkpointed and restored together with the network.
            best_state = copy.deepcopy(model.state_dict())

        # Snapshot for the visualization. At this point `epoch - 1` optimizer
        # updates have been applied, so these are "epoch (epoch - 1)" weights in
        # the snapshot convention (epoch 0 = untrained network). We reuse the
        # losses just computed and only run an extra no-grad forward pass.
        if cfg.ENABLE_TRAINING_VISUALIZATION and (epoch - 1) in snapshot_epochs:
            snapshots.append(make_snapshot(epoch - 1, "training", model, t_visual, losses))

        loss_total.backward()                         # backprop: d(loss)/d(every trainable parameter)

        if epoch == 1:                                # sanity checks: gradients reach the parameters
            grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
            print(f"Gradient norm at epoch 1: {grad_norm.item():.4e} (must be > 0)")
            assert grad_norm.item() > 0, "No gradients reached the network!"
            if model.learns_damping:
                damping_grad = model.raw_damping.grad
                print(f"Gradient of the damping parameter at epoch 1: {damping_grad.item():.4e} "
                      "(must be nonzero)")
                assert damping_grad is not None and damping_grad.item() != 0.0, \
                    "The damping estimate receives no gradient!"

        optimizer.step()                              # Adam updates the parameters

        if epoch % cfg.PRINT_EVERY == 0 or epoch == 1:
            line = (f"Epoch {epoch:6d} | total {loss_total.item():.4e} | "
                    f"physics {losses['physics'].item():.4e} | IC {losses['ic'].item():.4e}")
            if "data" in losses:
                line += f" | data {losses['data'].item():.4e}"
            if model.learns_damping:
                line += f" | c_est {history['damping_estimate'][-1]:.4f}"
            print(line)

    if cfg.ENABLE_TRAINING_VISUALIZATION:
        # The weights after the LAST update have no recorded loss yet, so compute
        # it here (needs autograd for the derivatives; no backward, no update).
        final_losses = compute_training_losses(model, t_collocation, cfg, observations)
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
# Evaluation (reference solutions are compared from here on, never before)
# ----------------------------------------------------------------------------
def evaluate_pinn(model, cfg, x_reference, observations=None):
    """Compare the trained PINN with a reference solution. Evaluation ONLY.

    x_reference: reference positions on the evaluation grid (computed by the
    caller AFTER training, e.g. with the SciPy solver).
    Returns (t, x_pred, residual, metrics) as NumPy arrays + a dict.
    """
    # Dense grid, shape (N_EVAL, 1). requires_grad so we can also check the ODE residual.
    t_eval = make_evaluation_points(cfg).reshape(-1, 1).requires_grad_(True)
    x_pred, _, _ = predict_with_derivatives(model, t_eval)
    residual = compute_physics_residual(model, t_eval, cfg.MASS, damping_used_in_residual(model, cfg),
                                        cfg.SPRING_CONSTANT, cfg.NONLINEAR_COEFFICIENT)

    # Initial conditions as learned by the network (at TIME_START).
    _, position_error, velocity_error = compute_initial_condition_loss(
        model, cfg.TIME_START, cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY)

    # Training losses of the final (possibly restored) weights, on the collocation points.
    final_losses = compute_training_losses(model, make_collocation_points(cfg), cfg, observations)

    # Convert to NumPy (detach = leave the autograd graph).
    t_np = t_eval.detach().numpy().ravel()
    x_np = x_pred.detach().numpy().ravel()
    residual_np = residual.detach().numpy().ravel()

    x_reference = np.asarray(x_reference, dtype=np.float64)
    if x_reference.shape != x_np.shape:
        raise ValueError(f"x_reference has shape {x_reference.shape}, expected {x_np.shape}")
    error = x_np - x_reference

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
    if "data" in final_losses:
        metrics["final_model_loss_data_m2"] = final_losses["data"].item()
    return t_np, x_np, residual_np, metrics


def make_evaluation_points(cfg):
    """N_EVALUATION_POINTS evenly spaced times in [TIME_START, TIME_END] (1-D tensor)."""
    return torch.linspace(cfg.TIME_START, cfg.TIME_END, cfg.N_EVALUATION_POINTS)
