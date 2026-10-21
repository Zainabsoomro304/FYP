"""
Feedback agent (agent 4).

After a training run it looks at the metrics and decides: pass, or retry with
changed settings.

Two parts, on purpose:
  * quality(): fixed rules in code that decide pass or fail from the metrics.
    This is the authority, so a wrong verdict from the AI cannot slip through.
  * the AI part: it reads the same metrics, writes a diagnosis in plain words,
    and proposes setting changes. check() validates its answer (verdict agrees
    with the rules, only allowed settings, values in a sane range).
If the AI's proposal is invalid, fallback_changes() supplies a simple one.
"""
import json
import math

# Starting thresholds. They are choices we can tune and should discuss with sir.
THRESHOLDS = {
    "position_error_vs_amplitude": 0.02,   # position RMSE must be below 2% of the starting amplitude
    "ic_error": 5e-3,                      # initial position and velocity errors (m, m/s)
    "physics_residual_mse": 1e-3,          # mean squared equation residual (N^2)
    "data_mse": 1e-4,                      # inverse problems: mean squared mismatch with the data (m^2)
}

# Settings the agent may change, with (low, high) limits.
ALLOWED = {
    "EPOCHS": (500, 30000), "N_COLLOCATION": (50, 2000), "LEARNING_RATE": (1e-5, 1e-2),
    "LAMBDA_PHYSICS": (0.01, 100.0), "LAMBDA_IC": (0.1, 1000.0), "LAMBDA_DATA": (0.1, 1000.0),
    "HIDDEN_SIZE": (16, 128), "NUM_HIDDEN_LAYERS": (2, 6),
}
INTEGER_SETTINGS = {"EPOCHS", "N_COLLOCATION", "HIDDEN_SIZE", "NUM_HIDDEN_LAYERS"}

SYSTEM_PROMPT = """You are the feedback agent in a pipeline that solves differential equations
with physics-informed neural networks (PINNs). You get the problem, the
current training settings, the metrics of the run and the pass thresholds.
Decide whether the run is good enough and, if not, what to change. Output JSON only.

How to read the metrics:
- position_rmse_m is the average position error. For a forward problem it
  compares the PINN with the SciPy numerical solution. For an inverse problem
  it compares the PINN with SciPy solved using the estimated parameter.
- ic_position_error and ic_velocity_error say whether the curve starts at the
  right place. If they are too big, raise LAMBDA_IC.
- physics_residual_mse says whether the curve obeys the equation. If it is too
  big while the initial conditions are fine, train longer (EPOCHS) or use more
  N_COLLOCATION.
- data_mse (inverse only) says whether the curve passes through the measurements.
  If it is too big, raise LAMBDA_DATA.
- If training failed or the loss blew up, lower LEARNING_RATE.

Rules:
1. verdict is "pass" only if every metric is within its threshold; otherwise "retry".
2. Change at most 3 settings, and only ones the metrics justify.
3. You may only change: EPOCHS, N_COLLOCATION, LEARNING_RATE, LAMBDA_PHYSICS,
   LAMBDA_IC, LAMBDA_DATA (inverse only), HIDDEN_SIZE, NUM_HIDDEN_LAYERS.
4. If the verdict is "pass", "changes" is {}.

Output format:
{"verdict": "pass" | "retry",
 "diagnosis": "one or two plain sentences",
 "changes": {"EPOCHS": 20000},
 "reason": "why these changes should help"}
"""


def amplitude(ode):
    ic = ode["initial_conditions"]
    return abs(ic["x"]) if ic["x"] else 1.0


def quality(summary, ode):
    """Fixed rules. Returns (passed, list of failed checks, list of notes)."""
    failed, notes = [], []
    amp = amplitude(ode)
    t = THRESHOLDS
    values = [summary[k] for k in ("position_rmse_m", "physics_residual_mse",
                                   "ic_position_error", "ic_velocity_error")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return False, ["a metric is missing or not finite"], notes
    if summary["position_rmse_m"] > t["position_error_vs_amplitude"] * amp:
        failed.append(f"position_rmse_m {summary['position_rmse_m']:.2e} is above "
                      f"{t['position_error_vs_amplitude'] * amp:.2e}")
    if summary["ic_position_error"] > t["ic_error"]:
        failed.append(f"ic_position_error {summary['ic_position_error']:.2e} is above {t['ic_error']:.0e}")
    if summary["ic_velocity_error"] > t["ic_error"]:
        failed.append(f"ic_velocity_error {summary['ic_velocity_error']:.2e} is above {t['ic_error']:.0e}")
    if summary["physics_residual_mse"] > t["physics_residual_mse"]:
        failed.append(f"physics_residual_mse {summary['physics_residual_mse']:.2e} is above "
                      f"{t['physics_residual_mse']:.0e}")
    if summary["mode"] == "inverse":
        if summary["data_mse"] > t["data_mse"]:
            failed.append(f"data_mse {summary['data_mse']:.2e} is above {t['data_mse']:.0e}")
        if summary.get("n_observations", 99) < 10:
            notes.append(f"only {summary['n_observations']} observations, so the estimated "
                         "parameter may not be unique")
    return not failed, failed, notes


def agent_input(ode, pinn_settings, summary, attempt):
    return json.dumps({
        "problem": {"equation": ode["equation"]["form"], "problem_type": ode["problem_type"],
                    "time_interval": ode["time_interval"],
                    "initial_conditions": ode["initial_conditions"]},
        "current_settings": pinn_settings, "attempt": attempt,
        "metrics": {k: v for k, v in summary.items() if k not in ("log_file", "plots_folder")},
        "thresholds": THRESHOLDS,
    }, indent=2)


def check(out, passed_by_rules, mode, current):
    """Check the AI's answer. Returns a list of problems (empty = fine)."""
    problems = [f"missing field: {k}" for k in ("verdict", "diagnosis", "changes", "reason")
                if k not in out]
    if problems:
        return problems
    if out["verdict"] not in ("pass", "retry"):
        return [f"verdict {out['verdict']!r} is not pass or retry"]
    if (out["verdict"] == "pass") != passed_by_rules:
        problems.append(f"verdict is {out['verdict']} but the fixed rules say "
                        f"{'pass' if passed_by_rules else 'fail'}")
    changes = out["changes"] or {}
    if out["verdict"] == "pass":
        if changes:
            problems.append("verdict is pass but it still proposes changes")
        return problems
    if not changes:
        problems.append("retry without any change would repeat the same run")
    if len(changes) > 3:
        problems.append("more than 3 changes at once")
    for name, value in changes.items():
        if name not in ALLOWED:
            problems.append(f"{name} is not an allowed setting")
            continue
        if name == "LAMBDA_DATA" and mode != "inverse":
            problems.append("LAMBDA_DATA only applies to inverse problems")
        lo, hi = ALLOWED[name]
        if not isinstance(value, (int, float)) or not lo <= value <= hi:
            problems.append(f"{name}={value} is outside {lo} to {hi}")
        elif name in INTEGER_SETTINGS and int(value) != value:
            problems.append(f"{name} must be a whole number")
    if changes and all(current.get(k) == v for k, v in changes.items()):
        problems.append("proposed settings are the same as the current ones")
    return problems


def fallback_changes(failed, mode, current):
    """Simple rule-based proposal, used when the AI's proposal is invalid."""
    changes = {}
    text = " ".join(failed)
    if "ic_" in text:
        changes["LAMBDA_IC"] = min(current["LAMBDA_IC"] * 2, ALLOWED["LAMBDA_IC"][1])
    if "data_mse" in text and mode == "inverse":
        changes["LAMBDA_DATA"] = min(current["LAMBDA_DATA"] * 2, ALLOWED["LAMBDA_DATA"][1])
    if not changes or "physics" in text or "position_rmse" in text:
        changes["EPOCHS"] = int(min(current["EPOCHS"] * 2, ALLOWED["EPOCHS"][1]))
    return changes
