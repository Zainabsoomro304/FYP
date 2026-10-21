"""
All four agents connected, from plain language to a trained and checked PINN.

  text -> agent 1 (ODE) -> agent 2 (PINN setup) -> agent 3 (code: runs Zainab's PINN)
       -> agent 4 (feedback) -> retry with new settings, or finish

Put the agent files in Zainab's project folder (next to main.py and config.py).

  python run_all.py --backend gemini
  python run_all.py --backend gemini --text "Simulate a spring with m=1, k=4, c=0.5 ..."
  python run_all.py --backend gemini --max-attempts 1          # faster, no retry loop
  python run_all.py --backend selftest                         # fake AI, real training
"""
import argparse
import json
import os
import sys
import time

import code_agent
import feedback_agent
import ode_agent
import pinn_agent
from ode_agent import call_llm, extract_json

DEMO_TEXT = ("Simulate a spring with mass 2 kg, stiffness 3, damping 2.5 and nonlinear "
             "coefficient 0.6, released from x=1 m at rest, from 0 to 10 s.")


def say(msg=""):
    print(msg, flush=True)


def ask(llm, system, user_text, check_fn, tries=2):
    """Call the AI; if the checker finds problems, tell it and let it try once more."""
    problems, out, text = [], None, user_text
    for attempt in range(1, tries + 1):
        try:
            out = extract_json(llm(text, system))
            problems = check_fn(out)
        except Exception as e:
            out, problems = None, [f"could not read the reply: {e}"]
        if not problems:
            return out, [], attempt
        text = (user_text + "\n\nYour previous answer had these problems, please fix them "
                "and answer again with JSON only:\n- " + "\n- ".join(problems))
    return out, problems, tries


def ode_self_check(text):
    def check(out):
        unknown = sorted(k for k, v in out.get("parameters", {}).items() if not v.get("known", True))
        test = {"text": text, "expect": {"status": out.get("status"),
                                         "problem_type": out.get("problem_type"),
                                         "unknown": unknown}}
        return ode_agent.check(out, test)
    return check


def tunables(spec):
    return {k: v for k, v in spec["overrides"].items() if k in feedback_agent.ALLOWED}


def run_system(text, llm, out_dir, max_attempts=3, first_epochs=None):
    result = {"input": text, "stopped_at": None, "attempts": [], "final": None}
    os.makedirs(out_dir, exist_ok=True)

    # ---- agent 1
    say("Agent 1 (ODE): reading the problem ...")
    ode, problems, tries = ask(llm, ode_agent.SYSTEM_PROMPT, text, ode_self_check(text))
    result["ode"] = ode
    if problems:
        result["stopped_at"] = "agent 1 could not produce a valid answer"
        result["problems"] = problems
        say("  stopped, problems: " + "; ".join(problems))
        return result
    say(f"  equation: {ode['equation']['form']}   type: {ode['problem_type']}   status: {ode['status']}")
    if ode["status"] != "ok":
        result["stopped_at"] = "agent 1 needs more information from the user"
        say("  Agent 1 needs more information before anything can run:")
        for q in ode["questions_for_user"]:
            say(f"   - {q}")
        return result

    # ---- agent 2
    say("Agent 2 (PINN): choosing the network and training setup ...")
    pinn, problems, tries = ask(llm, pinn_agent.SYSTEM_PROMPT, json.dumps(ode, indent=2),
                                lambda o: pinn_agent.check(o, ode))
    result["pinn"] = pinn
    if problems:
        result["stopped_at"] = "agent 2 could not produce a valid answer"
        result["problems"] = problems
        say("  stopped, problems: " + "; ".join(problems))
        return result
    a, o = pinn["architecture"], pinn["optimizer"]
    say(f"  MLP {a['hidden_layers']}x{a['neurons_per_layer']} {a['activation']}, "
        f"{pinn['collocation']['n_points']} collocation points, {o['epochs']} epochs, "
        f"trainable: {pinn['trainable_parameters'] or 'none'}")

    # ---- agent 3 (build the run)
    say("Agent 3 (code): translating the plan into settings for Zainab's PINN code ...")
    try:
        spec = code_agent.build_spec(ode, pinn)
    except code_agent.UnsupportedProblem as e:
        result["stopped_at"] = f"agent 3 cannot run this: {e}"
        say(f"  stopped: {e}")
        return result
    if first_epochs:
        spec["overrides"]["EPOCHS"] = first_epochs
    for w in spec["warnings"]:
        say(f"  note: {w}")

    # ---- agents 3 and 4 in a loop
    for attempt in range(1, max_attempts + 1):
        say(f"\nAttempt {attempt}: training ({spec['overrides']['EPOCHS']} epochs) ...")
        run_dir = os.path.join(out_dir, f"attempt_{attempt}")
        try:
            summary = code_agent.run_spec(spec, run_dir)
        except Exception as e:
            result["stopped_at"] = f"training failed: {e}"
            say(f"  training failed: {e}")
            return result
        passed, failed, notes = feedback_agent.quality(summary, ode)
        say(f"  trained in {summary['training_time_s']:.0f}s | position error "
            f"{summary['position_rmse_m']:.2e} m | physics residual "
            f"{summary['physics_residual_mse']:.2e} | IC errors "
            f"{summary['ic_position_error']:.1e}, {summary['ic_velocity_error']:.1e}"
            + (f" | data fit {summary['data_mse']:.2e} | estimated c = {summary['estimated_c']:.4f}"
               if summary["mode"] == "inverse" else ""))

        say("Agent 4 (feedback): judging the run ...")
        current = tunables(spec)
        fb, fb_problems, _ = ask(
            llm, feedback_agent.SYSTEM_PROMPT,
            feedback_agent.agent_input(ode, current, summary, attempt),
            lambda o: feedback_agent.check(o, passed, summary["mode"], current))
        record = {"attempt": attempt, "settings": current, "summary": summary,
                  "passed_by_rules": passed, "failed_checks": failed, "notes": notes,
                  "feedback": fb, "feedback_problems": fb_problems}
        result["attempts"].append(record)
        if fb:
            say(f"  diagnosis: {fb.get('diagnosis')}")
        for n in notes:
            say(f"  note: {n}")

        if passed:
            say("  PASS: all checks are within their thresholds.")
            result["final"] = "pass"
            return result
        for f in failed:
            say(f"  failed: {f}")
        if attempt == max_attempts:
            say("  out of attempts.")
            result["final"] = "did not pass"
            return result

        if fb_problems:
            say("  the AI's proposal was invalid (" + "; ".join(fb_problems) +
                "), using the simple fallback rule instead")
            changes = feedback_agent.fallback_changes(failed, summary["mode"], current)
            record["used_fallback"] = True
        else:
            changes = fb["changes"]
        for k, v in changes.items():
            spec["overrides"][k] = int(v) if k in feedback_agent.INTEGER_SETTINGS else float(v)
        record["applied_changes"] = changes
        say(f"  retrying with: {changes}")
    return result


# ------------------------------------------------------------------ fake AI for self-test
def make_fake_llm(ode_json):
    def llm(user_text, system):
        if system == ode_agent.SYSTEM_PROMPT:
            return json.dumps(ode_json)
        if system == pinn_agent.SYSTEM_PROMPT:
            return json.dumps(pinn_agent.good(json.loads(user_text)))
        if system == feedback_agent.SYSTEM_PROMPT:
            data = json.loads(user_text)
            passed, failed, _ = feedback_agent.quality(
                data["metrics"], {"initial_conditions": data["problem"]["initial_conditions"]})
            if passed:
                return json.dumps({"verdict": "pass", "diagnosis": "All metrics are within "
                                   "their thresholds.", "changes": {}, "reason": "n/a"})
            return json.dumps({"verdict": "retry", "diagnosis": "; ".join(failed),
                               "changes": {"EPOCHS": data["current_settings"]["EPOCHS"] * 2},
                               "reason": "train longer"})
        raise ValueError("unknown agent")
    return llm


def selftest(out_dir, epochs):
    import numpy as np
    from physics import solve_reference_ode
    forward = pinn_agent.ode()
    t = np.linspace(0, 10, 25)
    x, _ = solve_reference_ode(t, 2.0, 0.3, 0.3, 0.0, 1.0, 0.0, 0.0)   # data made with true c = 0.3
    inverse = pinn_agent.ode(problem_type="inverse", parameters=pinn_agent.INVERSE,
                             expected_behavior=pinn_agent.NO_EB,
                             observed_data={"t": [round(v, 4) for v in t],
                                            "x": [round(v, 5) for v in x]})
    needs_info = pinn_agent.ode(status="needs_clarification", parameters=pinn_agent.NONLINEAR,
                                expected_behavior=pinn_agent.NO_EB,
                                questions_for_user=["How was the spring released?"])
    all_ok = True
    for name, ode, text in [
            ("forward", forward, "Simulate a spring with m=2 kg, k=0.3 N/m, c=0.3 Ns/m, released "
                                 "from x=1 m at rest, from 0 to 10 s."),
            ("inverse", inverse, "A spring with mass 2 kg and stiffness 0.3 has unknown damping, "
                                 "released from x=1 m at rest. Measured positions are attached. "
                                 "Simulate from 0 to 10 s."),
            ("needs information", needs_info, "Simulate a spring with mass 2 kg, stiffness 0.3, "
                                              "damping 0.3 and nonlinear coefficient 0.2, from 1 "
                                              "to 10 sec.")]:
        say("=" * 70 + f"\nSELFTEST: {name}\n" + "=" * 70)
        res = run_system(text, make_fake_llm(ode), os.path.join(out_dir, name.replace(" ", "_")),
                         max_attempts=2, first_epochs=epochs)
        ok = (res["stopped_at"] is not None) if name == "needs information" else (res["final"] is not None)
        all_ok &= ok
        say(f"\n-> {'OK' if ok else 'BAD'}: {res['final'] or res['stopped_at']}\n")
    say("all four agents are wired correctly" if all_ok else "something is wrong")
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="selftest", choices=["selftest", "gemini", "anthropic", "manual"])
    p.add_argument("--text", default=DEMO_TEXT)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None, help="override the epochs of the first attempt")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join("agent_runs", time.strftime("%Y%m%d_%H%M%S"))
    if args.backend == "selftest":
        sys.exit(0 if selftest(out, args.epochs or 1500) else 1)
    llm = lambda user_text, system: call_llm(user_text, args.backend, system=system)
    res = run_system(args.text, llm, out, args.max_attempts, args.epochs)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run_all_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    say(f"\nEverything is saved in {out}")


if __name__ == "__main__":
    main()
