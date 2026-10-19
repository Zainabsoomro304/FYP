"""
visualization.py
================

Shows how the PINN's predicted curve evolves during training.

How it works
------------
1. DURING training (train_pinn() in main.py) the current network is evaluated
   on a fixed time grid at selected epochs, and only the resulting prediction
   array plus the loss values are stored ("snapshots"). No weights are copied,
   nothing is retrained, and the model parameters are not modified.
2. AFTER training, the analytical solution is computed once on the same grid
   (for plotting and RMSE only -- it is never part of the training loss), and
   the snapshots are turned into a static figure and an animation.

Epoch convention used in every snapshot: "epoch N" = the weights after N
optimizer updates. Epoch 0 is the randomly initialised, untrained network.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch

from physics import analytical_solution

# Epochs that are always captured (if they are <= EPOCHS).
MILESTONE_EPOCHS = [0, 1, 10, 100, 500, 1000, 2500, 5000, 7500]
MAX_ANIMATION_FRAMES = 300       # longer runs are thinned out so the GIF stays small
HOLD_LAST_FRAMES_MS = 1500       # how long the final / best frames stay on screen
LOSS_FLOOR = 1e-16               # zero or non-finite losses are drawn at this value on log axes


# ----------------------------------------------------------------------------
# Snapshot capture (called from the training loop; no analytical solution here)
# ----------------------------------------------------------------------------
def select_snapshot_epochs(total_epochs, interval):
    """Set of epochs (= number of optimizer updates) at which to take a snapshot."""
    epochs = {e for e in MILESTONE_EPOCHS if e <= total_epochs}
    epochs.update(range(0, total_epochs + 1, interval))
    epochs.add(total_epochs)
    return epochs


def make_visualization_points(cfg):
    """Fixed time grid used for every snapshot, shape (VISUALIZATION_POINTS, 1)."""
    return torch.linspace(cfg.TIME_START, cfg.TIME_END, cfg.VISUALIZATION_POINTS).reshape(-1, 1)


def capture_prediction(model, t_visual):
    """Current predicted positions on the visualization grid, as a NumPy array.

    model.eval() + torch.no_grad(): a plain forward pass, no autograd graph,
    no effect on gradients or weights. The model's train/eval mode is restored
    afterwards so training continues exactly as before.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_pred = model(t_visual).numpy().ravel().copy()
    model.train(was_training)
    return x_pred


def make_snapshot(epoch, kind, model, t_visual, losses):
    """kind: 'training' (during the loop), 'final' (last weights) or 'best' (restored).

    `losses` may hold tensors or plain floats; only their numeric values are kept.
    """
    def as_float(value):
        return value.item() if torch.is_tensor(value) else float(value)

    return {
        "epoch": epoch,
        "kind": kind,
        "x_pred": capture_prediction(model, t_visual),
        "loss_total": as_float(losses["total"]),
        "loss_physics": as_float(losses["physics"]),
        "loss_ic": as_float(losses["ic"]),
    }


# ----------------------------------------------------------------------------
# Helpers for plotting
# ----------------------------------------------------------------------------
def _safe_for_log(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, LOSS_FLOOR)


def _snapshot_title(snap):
    if snap["kind"] == "final":
        return f"Final epoch {snap['epoch']} (last weights)"
    if snap["kind"] == "best":
        return (f"Best weights, epoch {snap['epoch']} "
                f"(restored; lowest training loss)")
    return f"Epoch {snap['epoch']}"


def _y_limits(x_exact, snapshots):
    """Fixed y-range: the analytical curve plus the predictions, but never more
    than ~3x the analytical range so a wild untrained curve can't squash the plot."""
    lo, hi = float(np.min(x_exact)), float(np.max(x_exact))
    span = max(hi - lo, 1e-6)
    pred_lo = min(float(np.min(s["x_pred"])) for s in snapshots)
    pred_hi = max(float(np.max(s["x_pred"])) for s in snapshots)
    lo, hi = max(min(lo, pred_lo), lo - span), min(max(hi, pred_hi), hi + span)
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def _info_text(snap, rmse):
    return (f"{_snapshot_title(snap)}\n"
            f"total (weighted) {snap['loss_total']:.2e} | physics {snap['loss_physics']:.2e} N^2 | "
            f"IC {snap['loss_ic']:.2e} | RMSE {rmse:.2e} m")


# ----------------------------------------------------------------------------
# Static figure
# ----------------------------------------------------------------------------
def plot_training_snapshots(t, x_exact, snapshots, rmse_by_id, cfg, subtitle):
    """training_snapshots.png: one panel per milestone epoch + final + best."""
    training = [s for s in snapshots if s["kind"] == "training" and s["epoch"] in MILESTONE_EPOCHS]
    chosen = training + [s for s in snapshots if s["kind"] in ("final", "best")]

    n_cols = 3 if len(chosen) in (5, 6, 9) else min(4, len(chosen))
    n_rows = math.ceil(len(chosen) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    y_lo, y_hi = _y_limits(x_exact, chosen)

    for ax, snap in zip(axes.flat, chosen):
        ax.plot(t, x_exact, "k-", lw=1.8, label="Analytical")
        ax.plot(t, snap["x_pred"], "--", color="tab:red", lw=1.6, label="PINN")
        ax.set_title(f"{_snapshot_title(snap)}\nRMSE = {rmse_by_id[id(snap)]:.2e} m, "
                     f"loss = {snap['loss_total']:.2e}", fontsize=8.5)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(chosen):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time t [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Position x [m]")
    axes.flat[0].legend(fontsize=8, loc="best")

    fig.suptitle("PINN prediction at different training epochs (epoch N = after N optimizer updates)\n"
                 + subtitle, fontsize=10)
    fig.tight_layout()
    path = os.path.join(cfg.OUTPUT_DIR, "training_snapshots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------------
def save_solution_animation(t, x_exact, snapshots, rmse_by_id, history, cfg, subtitle):
    """Left: solution curve evolving. Right: loss history revealed up to that epoch.

    Returns (path or None, message).
    """
    training = sorted((s for s in snapshots if s["kind"] == "training"), key=lambda s: s["epoch"])
    if len(training) > MAX_ANIMATION_FRAMES:      # thin out evenly, always keeping epoch 0
        keep = np.unique(np.linspace(0, len(training) - 1, MAX_ANIMATION_FRAMES).round().astype(int))
        training = [training[i] for i in keep]
    hold = max(1, round(HOLD_LAST_FRAMES_MS / cfg.ANIMATION_INTERVAL_MS))
    final = [s for s in snapshots if s["kind"] == "final"]
    best = [s for s in snapshots if s["kind"] == "best"]
    frames = training + final * hold + best * hold

    # Loss history indexed by "number of updates": entry i = loss of the weights
    # after i updates. history[...] has entries 0..EPOCHS-1; the final weights'
    # losses (after EPOCHS updates) come from the 'final' snapshot.
    loss_curves = {name: list(history[name]) for name in ("total", "physics", "ic")}
    if final:
        loss_curves["total"].append(final[0]["loss_total"])
        loss_curves["physics"].append(final[0]["loss_physics"])
        loss_curves["ic"].append(final[0]["loss_ic"])
    loss_curves = {name: _safe_for_log(v) for name, v in loss_curves.items()}
    loss_epochs = np.arange(len(loss_curves["total"]))

    fig, (ax_sol, ax_loss) = plt.subplots(1, 2, figsize=(13, 5), dpi=80,
                                          gridspec_kw={"width_ratios": [1.25, 1]})

    # Left panel: analytical curve drawn ONCE and never changed.
    ax_sol.plot(t, x_exact, "k-", lw=2, label="Analytical (fixed)")
    (pred_line,) = ax_sol.plot(t, frames[0]["x_pred"], "--", color="tab:red", lw=2,
                               label="PINN prediction")
    ax_sol.set_ylim(*_y_limits(x_exact, frames))
    ax_sol.set_xlabel("Time t [s]")
    ax_sol.set_ylabel("Position x [m]")
    ax_sol.grid(True, alpha=0.3)
    ax_sol.legend(loc="lower right", fontsize=8)

    # Right panel: loss curves, revealed progressively.
    loss_lines = {
        "total": ax_loss.semilogy([], [], color="black", lw=1.4,
                                  label="Total loss (weighted)")[0],
        "physics": ax_loss.semilogy([], [], color="tab:blue", lw=1.0,
                                    label="Physics loss (unweighted)")[0],
        "ic": ax_loss.semilogy([], [], color="tab:orange", lw=1.0,
                               label="IC loss (unweighted)")[0],
    }
    all_losses = np.concatenate(list(loss_curves.values()))
    ax_loss.set_xlim(0, max(1, loss_epochs[-1]))
    ax_loss.set_ylim(all_losses.min() / 3, all_losses.max() * 3)
    ax_loss.set_xlabel("Epoch (optimizer updates)")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=8)
    epoch_marker = ax_loss.axvline(0, color="tab:red", lw=1, ls=":")
    fig.suptitle(subtitle, fontsize=9)
    ax_sol.set_title(_info_text(frames[0], 0.0), fontsize=9)    # reserve space for the title
    fig.tight_layout()

    def draw(snap):
        pred_line.set_ydata(snap["x_pred"])
        upto = len(loss_epochs) if snap["kind"] == "best" else snap["epoch"] + 1
        for name, line in loss_lines.items():
            line.set_data(loss_epochs[:upto], loss_curves[name][:upto])
        epoch_marker.set_xdata([snap["epoch"], snap["epoch"]])
        # Epoch, losses and RMSE go in the title so they never cover the curves.
        ax_sol.set_title(_info_text(snap, rmse_by_id[id(snap)]), fontsize=9,
                         color="tab:green" if snap["kind"] == "best" else "black")
        return [pred_line, epoch_marker, *loss_lines.values()]

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=cfg.ANIMATION_INTERVAL_MS,
                                   blit=False, cache_frame_data=False)
    fps = 1000.0 / cfg.ANIMATION_INTERVAL_MS
    message = ""
    try:
        if cfg.ANIMATION_FORMAT == "mp4" and animation.writers.is_available("ffmpeg"):
            path = os.path.join(cfg.OUTPUT_DIR, "solution_evolution.mp4")
            anim.save(path, writer=animation.FFMpegWriter(fps=fps))
        else:
            if cfg.ANIMATION_FORMAT == "mp4":
                message = "ffmpeg not found, so MP4 export is unavailable; saved a GIF instead. "
            path = os.path.join(cfg.OUTPUT_DIR, "solution_evolution.gif")
            anim.save(path, writer=animation.PillowWriter(fps=fps))
        message += f"Animation saved: {path} ({len(frames)} frames)"
    except Exception as error:                     # missing Pillow/encoder, disk problems, ...
        path = None
        message = (f"Animation could not be saved ({type(error).__name__}: {error}). "
                   "The static training_snapshots.png is still available.")
    finally:
        plt.close(fig)
    return path, message


# ----------------------------------------------------------------------------
# Entry point used by main.py after training
# ----------------------------------------------------------------------------
def create_training_visualizations(snapshots, history, cfg, subtitle):
    """Make training_snapshots.png and the animation. Returns info for metrics.json."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    t = make_visualization_points(cfg).numpy().ravel()

    # The analytical solution is computed once here, AFTER training, for plotting
    # and RMSE only. It stays fixed for every frame.
    x_exact = analytical_solution(t, cfg.MASS, cfg.SPRING_CONSTANT, cfg.DAMPING_COEFFICIENT,
                                  cfg.INITIAL_POSITION, cfg.INITIAL_VELOCITY, cfg.TIME_START)
    rmse_by_id = {id(s): float(np.sqrt(np.mean((s["x_pred"] - x_exact) ** 2))) for s in snapshots}

    snapshot_path = plot_training_snapshots(t, x_exact, snapshots, rmse_by_id, cfg, subtitle)
    print(f"Snapshot figure saved: {snapshot_path}")
    animation_path, message = save_solution_animation(t, x_exact, snapshots, rmse_by_id,
                                                      history, cfg, subtitle)
    print(message)

    return {
        "snapshot_epochs": sorted(s["epoch"] for s in snapshots if s["kind"] == "training"),
        "final_epoch": next((s["epoch"] for s in snapshots if s["kind"] == "final"), None),
        "best_weights_epoch": next((s["epoch"] for s in snapshots if s["kind"] == "best"), None),
        "snapshot_rmse_m": {f"{s['kind']}_{s['epoch']}": rmse_by_id[id(s)] for s in snapshots},
        "training_snapshots_file": os.path.basename(snapshot_path),
        "animation_file": os.path.basename(animation_path) if animation_path else None,
        "animation_message": message,
    }
