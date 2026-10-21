"""
PINN agent: takes the ODE agent's JSON and decides the PINN setup.
Keep this file in the same folder as ode_agent.py.

  python pinn_agent.py --backend selftest   # checks the checker, needs nothing
  python pinn_agent.py --backend gemini     # real run (needs GEMINI_API_KEY)
  python pinn_agent.py --backend manual     # paste a chatbot's replies by hand
"""
import json
import re
import sys

from ode_agent import call_llm, extract_json

SYSTEM_PROMPT = """You are the PINN agent in a pipeline that solves differential equations
with physics-informed neural networks (PINNs). You receive the JSON produced
by the ODE agent. Decide how the PINN should be set up and output JSON only.

Rules:
1. If the input status is not "ok" (for example "needs_clarification"), do NOT
   design anything. Return {"status": "blocked_upstream", "reason": "..."}.
2. The network is an MLP with one input (time t) and one output (the unknown x).
3. The activation must be smooth, such as tanh. The physics loss needs the
   second derivative of the network, and ReLU has a second derivative of zero
   everywhere, so never use relu or leaky_relu.
4. The physics loss uses the SAME equation as the input, copied exactly, with
   derivatives taken by autograd at collocation points.
5. Collocation points must cover the input's time_interval. Use at least 100.
6. The initial condition loss uses the input's initial_conditions exactly
   (time, x, x_dot). It is a separate loss from the physics loss.
7. If problem_type is "forward": trainable_parameters is [] and the data
   loss is null.
8. If problem_type is "inverse": trainable_parameters lists every parameter
   with known=false, and the data loss must exist (it compares the network
   with observed_data). Give each trainable parameter a positive starting
   guess in "initial_guesses". Keep trainable physical parameters positive
   by setting "positive_parameterization" to "softplus"; otherwise the
   optimizer could try an unphysical negative value (for example negative
   damping would add energy).
9. All loss weights must be positive. These are starting values to tune,
   say so in "notes". Good starting values: physics 1, initial condition 10,
   data 10.
10. Training length: use 10000 epochs. A nonlinear equation (one with an
   alpha*x^3 term) on a 10 second interval was not accurate enough with 3000.

Output format:
{
  "status": "ok",
  "architecture": {"type": "MLP", "n_inputs": 1, "n_outputs": 1,
                   "hidden_layers": 3, "neurons_per_layer": 32,
                   "activation": "tanh"},
  "collocation": {"n_points": 200, "time_interval": [0, 10],
                  "sampling": "uniform" | "random"},
  "losses": {
    "physics": {"residual": "m*x'' + c*x' + k*x = 0", "derivatives": "autograd",
                "weight": 1.0},
    "initial_condition": {"time": 0, "x": 1, "x_dot": 0, "weight": 10.0},
    "data": null | {"weight": 10.0}
  },
  "trainable_parameters": [],
  "initial_guesses": {},
  "positive_parameterization": null | "softplus",
  "optimizer": {"name": "Adam", "learning_rate": 0.001, "epochs": 10000},
  "notes": []
}
"""


# ---------------------------------------------------------------- test inputs
def ode(**over):
    """A valid ODE-agent output, to use as the PINN agent's input."""
    base = {
        "status": "ok", "problem_type": "forward",
        "equation": {"order": 2, "variable": "t", "unknown": "x",
                     "form": "m*x'' + c*x' + k*x = 0"},
        "parameters": {"m": {"value": 2, "known": True}, "c": {"value": 0.3, "known": True},
                       "k": {"value": 0.3, "known": True}},
        "initial_conditions": {"time": 0, "x": 1, "x_dot": 0, "source": "given"},
        "time_interval": [0, 10], "observed_data": None,
        "expected_behavior": {"damping_ratio": 0.19, "regime": "underdamped"},
        "assumptions": [], "questions_for_user": [],
    }
    base.update(over)
    return base


NONLINEAR = {"m": {"value": 2, "known": True}, "c": {"value": 0.3, "known": True},
             "k": {"value": 0.3, "known": True}, "alpha": {"value": 0.2, "known": True}}
INVERSE = {"m": {"value": 2, "known": True}, "c": {"value": None, "known": False},
           "k": {"value": 0.3, "known": True}}
NO_EB = {"damping_ratio": None, "regime": None}

TESTS = [
    {"name": "1_forward_linear", "ode": ode()},
    {"name": "2_forward_nonlinear_interval_starts_at_1",
     "ode": ode(parameters=NONLINEAR, expected_behavior=NO_EB, time_interval=[1, 10],
                equation={"order": 2, "variable": "t", "unknown": "x",
                          "form": "m*x'' + c*x' + k*x + alpha*x^3 = 0"},
                initial_conditions={"time": 1, "x": 1, "x_dot": 0, "source": "given"})},
    {"name": "3_inverse_unknown_damping",
     "ode": ode(problem_type="inverse", parameters=INVERSE, expected_behavior=NO_EB,
                observed_data={"t": [1, 2, 3], "x": [0.8, 0.5, 0.1]})},
    {"name": "4_blocked_upstream_missing_ic",
     "ode": ode(status="needs_clarification", parameters=NONLINEAR, expected_behavior=NO_EB,
                initial_conditions={"time": 1, "x": 1, "x_dot": 0, "source": "assumed"},
                questions_for_user=["What are the starting position and velocity at t=1?"])},
]

REQUIRED = ["status", "architecture", "collocation", "losses",
            "trainable_parameters", "initial_guesses", "positive_parameterization",
            "optimizer", "notes"]
SMOOTH = {"tanh", "sin", "silu", "swish", "gelu", "softplus", "sigmoid"}


def norm(s):
    return re.sub(r"\s+", "", s).replace("^", "**")


# ---------------------------------------------------------------- the checker
def check(out, ode_in):
    """Return a list of problems. An empty list means the output passed."""
    if ode_in["status"] != "ok":
        if out.get("status") != "blocked_upstream":
            return [f"input status is {ode_in['status']} so the agent must return "
                    f"blocked_upstream, got {out.get('status')}"]
        return []
    if out.get("status") != "ok":
        return [f"status is {out.get('status')}, expected ok"]
    problems = [f"missing field: {k}" for k in REQUIRED if k not in out]
    if problems:
        return problems

    a, c, L = out["architecture"], out["collocation"], out["losses"]

    # network
    if a.get("type") != "MLP":
        problems.append(f"architecture type is {a.get('type')}, expected MLP")
    if a.get("n_inputs") != 1 or a.get("n_outputs") != 1:
        problems.append("an ODE in time needs 1 input and 1 output")
    if not 2 <= a.get("hidden_layers", 0) <= 6:
        problems.append("hidden_layers should be between 2 and 6")
    if not 16 <= a.get("neurons_per_layer", 0) <= 128:
        problems.append("neurons_per_layer should be between 16 and 128")
    if str(a.get("activation", "")).lower() not in SMOOTH:
        problems.append(f"activation {a.get('activation')} is not smooth; the physics loss "
                        "needs second derivatives")

    # collocation points
    if c.get("n_points", 0) < 100:
        problems.append("fewer than 100 collocation points")
    if list(map(float, c.get("time_interval", []))) != list(map(float, ode_in["time_interval"])):
        problems.append(f"collocation interval {c.get('time_interval')} does not match "
                        f"the ODE agent's {ode_in['time_interval']}")
    if c.get("sampling") not in ("uniform", "random"):
        problems.append("sampling must be uniform or random")

    # physics loss: same equation as the input (catches a copied-wrong sign)
    phys = L.get("physics", {})
    if norm(phys.get("residual", "")) != norm(ode_in["equation"]["form"]):
        problems.append(f"physics residual {phys.get('residual')!r} differs from the input "
                        f"equation {ode_in['equation']['form']!r}")
    if phys.get("derivatives") != "autograd":
        problems.append("derivatives must be computed with autograd")

    # initial condition loss: same numbers as the input
    ic_in, ic_out = ode_in["initial_conditions"], L.get("initial_condition", {})
    for f in ("time", "x", "x_dot"):
        if ic_out.get(f) is None or float(ic_out[f]) != float(ic_in[f]):
            problems.append(f"initial condition {f} is {ic_out.get(f)}, input says {ic_in[f]}")

    # weights
    for name, loss in L.items():
        if loss is not None and not loss.get("weight", 0) > 0:
            problems.append(f"{name} loss weight must be positive")

    # forward vs inverse
    unknown = sorted(k for k, v in ode_in["parameters"].items() if not v.get("known", True))
    if sorted(out["trainable_parameters"]) != unknown:
        problems.append(f"trainable_parameters is {out['trainable_parameters']}, "
                        f"should be {unknown}")
    if ode_in["problem_type"] == "inverse" and L.get("data") is None:
        problems.append("inverse problem needs a data loss")
    if ode_in["problem_type"] == "forward" and L.get("data") is not None:
        problems.append("forward problem should not have a data loss")

    # trainable parameters need a positive starting guess and a positivity rule
    guesses = out["initial_guesses"] or {}
    if sorted(guesses) != unknown:
        problems.append(f"initial_guesses has {sorted(guesses)}, should have {unknown}")
    for name, g in guesses.items():
        if not isinstance(g, (int, float)) or not g > 0:
            problems.append(f"initial guess for {name} must be a positive number")
    want_pos = "softplus" if unknown else None
    if out["positive_parameterization"] != want_pos:
        problems.append(f"positive_parameterization is {out['positive_parameterization']}, "
                        f"expected {want_pos}")

    # optimizer
    o = out["optimizer"]
    nonlinear = any(k not in ("m", "c", "k") for k in ode_in["parameters"])
    if nonlinear and o.get("epochs", 0) < 8000:
        problems.append("nonlinear equation needs about 10000 epochs (3000 was not enough)")
    if not 1e-5 <= o.get("learning_rate", 0) <= 1e-1:
        problems.append("learning_rate outside 1e-5 to 1e-1")
    if o.get("epochs", 0) < 1000:
        problems.append("fewer than 1000 epochs")
    return problems


# ---------------------------------------------------------------- self test
def good(ode_in):
    unknown = [k for k, v in ode_in["parameters"].items() if not v.get("known", True)]
    ic = ode_in["initial_conditions"]
    return {
        "status": "ok",
        "architecture": {"type": "MLP", "n_inputs": 1, "n_outputs": 1, "hidden_layers": 3,
                         "neurons_per_layer": 32, "activation": "tanh"},
        "collocation": {"n_points": 200, "time_interval": ode_in["time_interval"],
                        "sampling": "uniform"},
        "losses": {
            "physics": {"residual": ode_in["equation"]["form"], "derivatives": "autograd",
                        "weight": 1.0},
            "initial_condition": {"time": ic["time"], "x": ic["x"], "x_dot": ic["x_dot"],
                                  "weight": 10.0},
            "data": {"weight": 10.0} if ode_in["problem_type"] == "inverse" else None},
        "trainable_parameters": unknown,
        "initial_guesses": {k: 1.0 for k in unknown},
        "positive_parameterization": "softplus" if unknown else None,
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "epochs": 10000},
        "notes": ["Loss weights are starting values to tune."],
    }


def selftest():
    t = {x["name"]: x["ode"] for x in TESTS}
    t1, t2 = t["1_forward_linear"], t["2_forward_nonlinear_interval_starts_at_1"]
    t3, t4 = t["3_inverse_unknown_damping"], t["4_blocked_upstream_missing_ic"]

    def mod(ode_in, path, value):
        out = good(ode_in)
        node = out
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return out

    cases = [
        ("good: forward linear", good(t1), t1, True),
        ("good: forward nonlinear", good(t2), t2, True),
        ("good: inverse", good(t3), t3, True),
        ("good: blocked when upstream needs clarification",
         {"status": "blocked_upstream", "reason": "initial conditions missing"}, t4, True),
        # wrong on purpose; the checker must catch each one
        ("ReLU activation", mod(t1, ["architecture", "activation"], "relu"), t1, False),
        ("collocation interval is 0-10 but the problem is 1-10",
         mod(t2, ["collocation", "time_interval"], [0, 10]), t2, False),
        ("residual with a sign error",
         mod(t1, ["losses", "physics", "residual"], "m*x'' - c*x' + k*x = 0"), t1, False),
        ("initial condition value changed",
         mod(t1, ["losses", "initial_condition", "x"], 2), t1, False),
        ("inverse problem but no trainable parameter",
         mod(t3, ["trainable_parameters"], []), t3, False),
        ("inverse problem but no data loss", mod(t3, ["losses", "data"], None), t3, False),
        ("forward problem with a data loss",
         mod(t1, ["losses", "data"], {"weight": 1.0}), t1, False),
        ("designed a PINN even though the ODE agent said needs_clarification",
         good(t1), t4, False),
        ("only 20 collocation points", mod(t1, ["collocation", "n_points"], 20), t1, False),
        ("inverse problem without a starting guess for c",
         mod(t3, ["initial_guesses"], {}), t3, False),
        ("inverse problem with a negative starting guess",
         mod(t3, ["initial_guesses"], {"c": -0.5}), t3, False),
        ("inverse problem without the softplus positivity rule",
         mod(t3, ["positive_parameterization"], None), t3, False),
        ("nonlinear equation trained for only 3000 epochs",
         mod(t2, ["optimizer", "epochs"], 3000), t2, False),
    ]
    failed = 0
    for label, out, ode_in, should_pass in cases:
        problems = check(out, ode_in)
        ok = (not problems) == should_pass
        failed += not ok
        print(f"{'OK ' if ok else 'BAD'}  {label}")
        for p in problems:
            print(f"       caught: {p}")
    print("\nchecker works" if not failed else f"\n{failed} checker cases misbehaved")
    return failed == 0


def run(backend):
    results, saved = [], []
    for test in TESTS:
        out = None
        try:
            user_text = json.dumps(test["ode"], indent=2)
            out = extract_json(call_llm(user_text, backend, system=SYSTEM_PROMPT))
            problems = check(out, test["ode"])
        except Exception as e:
            problems = [f"could not read the reply: {e}"]
        results.append((test["name"], problems))
        saved.append({"test": test["name"], "input": test["ode"],
                      "output": out, "problems": problems})
    with open("results_pinn.json", "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    print("Saved every input and output to results_pinn.json")
    print("\n" + "=" * 70)
    for name, problems in results:
        print(f"{'PASS' if not problems else 'FAIL'}  {name}")
        for p in problems:
            print(f"        - {p}")
    print(f"\n{sum(1 for _, p in results if not p)}/{len(results)} passed")


if __name__ == "__main__":
    backend = sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv else "selftest"
    if backend == "selftest":
        sys.exit(0 if selftest() else 1)
    run(backend)
