"""
Code agent (agent 3).

It does NOT write PyTorch from scratch. It reuses Zainab's PINN project: it
translates the ODE agent's JSON and the PINN agent's JSON into settings for
her code (the names in config.py), then runs her training code.

Limits of this version (on purpose, and reported honestly):
  * only the equation  m x'' + c x' + k x (+ alpha x^3) = 0  is supported
  * for an inverse problem only the damping c can be unknown
  * her network is MLP + tanh + Adam with evenly spaced collocation points;
    other choices from the PINN agent are replaced and listed in "warnings"
  * her code applies the initial conditions at the start of the interval

This file must sit in Zainab's project folder (next to main.py and config.py).
"""
import contextlib
import json
import os
import re
import time

SUPPORTED_FORMS = {
    "m*x''+c*x'+k*x=0": False,
    "m*x''+c*x'+k*x+alpha*x**3=0": True,
}


class UnsupportedProblem(Exception):
    """The request is valid, but this version of the code agent cannot run it."""


def _norm(s):
    return re.sub(r"\s+", "", s).replace("^", "**")


def build_spec(ode, pinn):
    """Translate agents 1 and 2 into a run specification (settings for Zainab's code)."""
    warnings = []
    form = _norm(ode["equation"]["form"])
    if form not in SUPPORTED_FORMS:
        raise UnsupportedProblem(
            f"equation {ode['equation']['form']!r} is not one of the forms Zainab's code "
            "supports (m x'' + c x' + k x = 0, optionally + alpha x^3)")
    has_alpha = SUPPORTED_FORMS[form]

    params = ode["parameters"]
    unknown = sorted(k for k, v in params.items() if not v.get("known", True))
    ic = ode["initial_conditions"]
    t0, t1 = ode["time_interval"]
    if float(ic["time"]) != float(t0):
        raise UnsupportedProblem(
            f"initial conditions are at t={ic['time']} but the interval starts at t={t0}; "
            "Zainab's code applies the initial conditions at the start of the interval")

    def val(name):
        v = params.get(name, {}).get("value")
        if v is None:
            raise UnsupportedProblem(f"parameter {name} has no value")
        return float(v)

    losses, arch, opt, col = pinn["losses"], pinn["architecture"], pinn["optimizer"], pinn["collocation"]
    if str(arch["activation"]).lower() != "tanh":
        warnings.append(f"activation {arch['activation']} requested, Zainab's code uses tanh")
    if opt["name"] != "Adam":
        warnings.append(f"optimizer {opt['name']} requested, Zainab's code uses Adam")
    if col["sampling"] != "uniform":
        warnings.append("random collocation sampling requested, Zainab's code uses evenly spaced points")

    overrides = {
        "MASS": val("m"), "SPRING_CONSTANT": val("k"),
        "NONLINEAR_COEFFICIENT": val("alpha") if has_alpha else 0.0,
        "INITIAL_POSITION": float(ic["x"]), "INITIAL_VELOCITY": float(ic["x_dot"]),
        "TIME_START": float(t0), "TIME_END": float(t1),
        "N_COLLOCATION": int(col["n_points"]),
        "HIDDEN_SIZE": int(arch["neurons_per_layer"]),
        "NUM_HIDDEN_LAYERS": int(arch["hidden_layers"]),
        "EPOCHS": int(opt["epochs"]), "LEARNING_RATE": float(opt["learning_rate"]),
        "LAMBDA_PHYSICS": float(losses["physics"]["weight"]),
        "LAMBDA_IC": float(losses["initial_condition"]["weight"]),
        "ENABLE_TRAINING_VISUALIZATION": False, "PRINT_EVERY": 1000,
    }
    spec = {"warnings": warnings, "observations": None}

    if ode["problem_type"] == "forward":
        overrides["EXPERIMENT_MODE"] = "physics_only"
        overrides["DAMPING_COEFFICIENT"] = val("c")
        spec["mode"] = "physics_only"
    else:
        if unknown != ["c"]:
            raise UnsupportedProblem(f"only the damping c can be unknown, got {unknown}")
        obs = ode.get("observed_data")
        if not obs or not obs.get("t") or len(obs["t"]) != len(obs["x"]):
            raise UnsupportedProblem("inverse problem needs observed_data with equal-length t and x")
        if min(obs["t"]) < t0 or max(obs["t"]) > t1:
            raise UnsupportedProblem("some observations lie outside the time interval")
        if len(obs["t"]) < 10:
            warnings.append(f"only {len(obs['t'])} observations: the estimate of c may not be "
                            "well determined")
        overrides["EXPERIMENT_MODE"] = "damping_estimation"
        overrides["LAMBDA_DATA"] = float(losses["data"]["weight"])
        overrides["INITIAL_DAMPING_ESTIMATE"] = float(pinn["initial_guesses"]["c"])
        spec["mode"] = "inverse"
        spec["observations"] = {"t": [float(v) for v in obs["t"]],
                                "x": [float(v) for v in obs["x"]]}
    spec["overrides"] = overrides
    return spec


# --------------------------------------------------------------------------- running
def run_spec(spec, run_dir):
    """Train with Zainab's code. Returns a summary dict with the same keys for both modes."""
    import numpy as np
    import torch

    os.makedirs(run_dir, exist_ok=True)
    overrides = dict(spec["overrides"], OUTPUT_DIR=os.path.abspath(run_dir))
    log_path = os.path.join(run_dir, "training_log.txt")
    start = time.time()
    # buffering=1: each printed line reaches the file at once, so progress can be followed live
    with open(log_path, "w", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log):
        if spec["mode"] == "physics_only":
            summary = _run_forward(overrides)
        else:
            summary = _run_inverse(overrides, spec["observations"], np, torch)
    summary["training_time_s"] = time.time() - start
    summary["epochs"] = overrides["EPOCHS"]
    summary["log_file"] = log_path
    return summary


def _run_forward(overrides):
    import main as zainab_main
    metrics = zainab_main.main(overrides)
    with open(os.path.join(overrides["OUTPUT_DIR"], "physics_only", "metrics.json"),
              encoding="utf-8") as f:
        report = json.load(f)
    return {
        "mode": "forward",
        "position_rmse_m": metrics["rmse_position_m"],
        "position_rmse_is": "PINN vs SciPy numerical solution",
        "physics_residual_mse": metrics["physics_residual_mse_eval_grid_N2"],
        "ic_position_error": metrics["initial_position_error_m"],
        "ic_velocity_error": metrics["initial_velocity_error_m_per_s"],
        "best_epoch": report["results"]["best_epoch"],
        "plots_folder": os.path.join(overrides["OUTPUT_DIR"], "physics_only"),
        "data_file": os.path.join(overrides["OUTPUT_DIR"], "physics_only", "solution_data.npz"),
    }


def _run_inverse(overrides, obs, np, torch):
    """Inverse problem with the user's own measurements.

    Zainab's Mode B makes synthetic measurements from a hidden true c. Here the
    measurements come from the user, so we use her training functions directly.
    There is no true c to compare with, so the checks are: does the curve match
    the data, does it obey the physics, and does SciPy (run with the estimated c)
    agree with the PINN's curve?
    """
    from damping_estimation import make_known_parameter_config, observations_to_tensors
    from physics import solve_reference_ode
    from pinn_model import (compute_initial_condition_loss, compute_physics_residual,
                            predict_with_derivatives)
    from plots import save_solution_data
    from training import (build_model, compute_training_losses, load_config,
                          make_collocation_points, make_evaluation_points, set_seed,
                          train_pinn)

    cfg = load_config(overrides)
    known = make_known_parameter_config(cfg)
    set_seed(cfg.SEED)
    torch.set_default_dtype(torch.float32)
    model = build_model(known, learn_damping=True)
    observations = observations_to_tensors(obs["t"], obs["x"])
    history, best_epoch, _ = train_pinn(model, known, observations)

    c_est = model.damping_coefficient()
    losses = compute_training_losses(model, make_collocation_points(known), known, observations)
    _, pos_err, vel_err = compute_initial_condition_loss(
        model, known.TIME_START, known.INITIAL_POSITION, known.INITIAL_VELOCITY)

    t_eval = make_evaluation_points(known).reshape(-1, 1).requires_grad_(True)
    residual = compute_physics_residual(model, t_eval, known.MASS, c_est, known.SPRING_CONSTANT,
                                        known.NONLINEAR_COEFFICIENT)
    x_pred_tensor, v_pred, _ = predict_with_derivatives(model, t_eval)
    x_pred = x_pred_tensor.detach().numpy().ravel()
    t_np = t_eval.detach().numpy().astype(np.float64).ravel()
    x_ref, v_ref = solve_reference_ode(t_np, known.MASS, known.SPRING_CONSTANT, float(c_est.item()),
                                       known.NONLINEAR_COEFFICIENT, known.INITIAL_POSITION,
                                       known.INITIAL_VELOCITY, known.TIME_START)
    data_file = save_solution_data(
        overrides["OUTPUT_DIR"], history, t=t_np, x_pred=x_pred,
        v_pred=v_pred.detach().numpy().ravel(), x_reference=x_ref, v_reference=v_ref,
        residual=residual.detach().numpy().ravel(),
        t_collocation=make_collocation_points(known).detach().numpy().ravel(),
        t_observed=obs["t"], x_observed=obs["x"])
    return {
        "mode": "inverse",
        "estimated_c": float(c_est.item()),
        "data_mse": float(losses["data"].item()),
        "position_rmse_m": float(np.sqrt(np.mean((x_pred - x_ref) ** 2))),
        "position_rmse_is": "PINN vs SciPy solved with the estimated c",
        "physics_residual_mse": float(np.mean(residual.detach().numpy().astype(np.float64) ** 2)),
        "ic_position_error": abs(pos_err.item()),
        "ic_velocity_error": abs(vel_err.item()),
        "best_epoch": best_epoch,
        "n_observations": len(obs["t"]),
        "data_file": data_file,
    }
