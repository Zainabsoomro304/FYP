"""
ODE agent: prompt + automatic checker + test runner.

How to run
  python ode_agent.py --backend selftest     # checks the checker itself, needs nothing
  python ode_agent.py --backend manual       # you paste the chatbot's JSON for each test
  python ode_agent.py --backend gemini       # calls Gemini (free key, needs GEMINI_API_KEY)
  python ode_agent.py --backend anthropic    # calls the Claude API (needs ANTHROPIC_API_KEY)
"""
import json
import math
import os
import re
import sys
import time

SYSTEM_PROMPT = """You are the ODE agent in a pipeline that solves differential equations
with physics-informed neural networks (PINNs). You receive a plain-language
problem. Turn it into the JSON format below. Output JSON only, no extra text.

Steps:
1. Identify the system and write its governing equation. Check the order
   (the highest derivative). Use SI units unless told otherwise. Write
   derivatives with primes, e.g. x'' for acceleration.
2. Extract every parameter and mark each one known or unknown.
3. Extract the initial conditions: the time they apply at, the starting
   position, and the starting velocity. Also extract the time interval.
4. Check completeness. Never invent a value silently.
   - If initial conditions are missing, set status to "needs_clarification",
     propose a default in "assumptions" (source: "assumed") and ask the user
     to confirm in "questions_for_user".
   - If a parameter is unknown, set problem_type to "inverse". An unknown
     parameter can only be found from measurements, so if none are given,
     set status to "needs_clarification" and ask for (t, x) data.
   - If the interval starts after t=0 and the text does not say when the
     initial conditions apply, ask.
5. For a linear second-order system (no nonlinear term), compute the damping
   ratio zeta = c / (2*sqrt(k*m)) and the regime (undamped, underdamped,
   critically damped, overdamped, unstable). For a nonlinear system set both
   to null.
6. Re-read the original text. Every number in your output must appear in it
   or be listed under "assumptions".

Output format:
{
  "status": "ok" | "needs_clarification",
  "problem_type": "forward" | "inverse",
  "equation": {"order": 2, "variable": "t", "unknown": "x",
               "form": "m*x'' + c*x' + k*x = 0"},
  "parameters": {"m": {"value": 2, "known": true},
                 "c": {"value": null, "known": false}},
  "initial_conditions": {"time": 0, "x": 1, "x_dot": 0,
                         "source": "given" | "assumed"},
  "time_interval": [0, 10],
  "observed_data": null | {"t": [...], "x": [...]},
  "expected_behavior": {"damping_ratio": 0.19, "regime": "underdamped"},
  "assumptions": [],
  "questions_for_user": []
}
"""

TESTS = [
    {
        "name": "1_complete",
        "text": "Simulate a mass-spring-damper with m=2 kg, k=0.3 N/m, c=0.3 Ns/m, "
                "released from x=1 m at rest, from 0 to 10 s.",
        "expect": {"status": "ok", "problem_type": "forward",
                   "unknown": [], "regime": "underdamped"},
    },
    {
        "name": "2_missing_initial_conditions",
        "text": "Simulate a spring with mass 2 kg, stiffness 0.3, damping 0.3 and "
                "nonlinear coefficient 0.2, from 1 to 10 sec.",
        "expect": {"status": "needs_clarification", "problem_type": "forward",
                   "unknown": []},
    },
    {
        "name": "3_unknown_damping_with_data",
        "text": "A spring with mass 2 kg and stiffness 0.3 has unknown damping. It is "
                "released from x=1 at rest. I measured x = 0.8 at t=1, 0.5 at t=2 and "
                "0.1 at t=3. Simulate to 10 s.",
        "expect": {"status": "ok", "problem_type": "inverse", "unknown": ["c"]},
    },
    {
        "name": "4_unknown_damping_no_data",
        "text": "A spring with mass 2 kg and stiffness 0.3 has unknown damping. It is "
                "released from x=1 at rest. Simulate to 10 s.",
        "expect": {"status": "needs_clarification", "problem_type": "inverse",
                   "unknown": ["c"]},
    },
]

REQUIRED_KEYS = ["status", "problem_type", "equation", "parameters",
                 "initial_conditions", "time_interval", "observed_data",
                 "expected_behavior", "assumptions", "questions_for_user"]


# ---------------------------------------------------------------- the checker
def numbers_in(text):
    return {float(n) for n in re.findall(r"\d+\.?\d*", text)}


def extract_json(raw):
    """Pull the JSON object out of a reply, even if it is wrapped in ``` fences."""
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the reply")
    return json.loads(raw[start:end + 1])


def check(out, test):
    """Return a list of problems. An empty list means the output passed."""
    problems = []
    text, expect = test["text"], test["expect"]

    for key in REQUIRED_KEYS:
        if key not in out:
            problems.append(f"missing field: {key}")
    if problems:
        return problems

    # compare with what we expect for this test case
    if out["status"] != expect["status"]:
        problems.append(f"status is {out['status']}, expected {expect['status']}")
    if out["problem_type"] != expect["problem_type"]:
        problems.append(f"problem_type is {out['problem_type']}, "
                        f"expected {expect['problem_type']}")

    params = out["parameters"]
    unknown = sorted(k for k, v in params.items() if not v.get("known", True))
    if unknown != sorted(expect["unknown"]):
        problems.append(f"unknown parameters are {unknown}, expected {expect['unknown']}")

    # the equation must be second order
    primes = [len(m) for m in re.findall(r"[A-Za-z]('+)", out["equation"].get("form", ""))]
    if not primes or max(primes) != 2 or out["equation"].get("order") != 2:
        problems.append("equation is not clearly second order (need x'' in form and order 2)")

    # no invented numbers: every known parameter must appear in the user's text
    in_text = numbers_in(text)
    for name, p in params.items():
        if p.get("known", True) and p.get("value") is not None:
            if float(p["value"]) not in in_text:
                problems.append(f"parameter {name}={p['value']} does not appear in the text")

    # initial conditions given by the user must really be in the text
    ic = out["initial_conditions"]
    if ic.get("source") == "given":
        for field in ("x", "x_dot"):
            v = ic.get(field)
            if v is not None and float(v) != 0 and float(v) not in in_text:
                problems.append(f"initial condition {field}={v} marked 'given' but not in text")

    # status must agree with questions and data
    if out["status"] == "ok" and out["questions_for_user"]:
        problems.append("status is ok but it still has questions for the user")
    if out["status"] == "needs_clarification" and not out["questions_for_user"]:
        problems.append("needs_clarification but no question was asked")
    if out["problem_type"] == "inverse" and out["status"] == "ok" and not out["observed_data"]:
        problems.append("inverse problem marked ok but no observed data")
    if unknown and out["problem_type"] != "inverse":
        problems.append("a parameter is unknown but problem_type is not inverse")

    # damping ratio: recompute it ourselves and compare
    eb = out["expected_behavior"] or {}
    has_nonlinear = any(k not in ("m", "c", "k") for k in params)
    if not has_nonlinear and not unknown and eb.get("damping_ratio") is not None:
        m, c, k = (params[n]["value"] for n in ("m", "c", "k"))
        zeta = c / (2 * math.sqrt(k * m))
        if abs(zeta - eb["damping_ratio"]) > 0.01:
            problems.append(f"damping_ratio is {eb['damping_ratio']}, should be {zeta:.3f}")
    if has_nonlinear and eb.get("damping_ratio") is not None:
        problems.append("nonlinear system: damping_ratio should be null")
    if "regime" in expect and eb.get("regime") != expect["regime"]:
        problems.append(f"regime is {eb.get('regime')}, expected {expect['regime']}")

    return problems


# ---------------------------------------------------------------- the LLM call
def call_llm(user_text, backend, system=None):
    system = system or SYSTEM_PROMPT
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        reply = client.messages.create(
            model=os.environ.get("ODE_MODEL", "claude-sonnet-5-5"),
            max_tokens=1500, system=system,
            messages=[{"role": "user", "content": user_text}])
        return reply.content[0].text
    if backend == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client()  # reads GEMINI_API_KEY from the environment
        reply = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json"))
        time.sleep(6)  # stay under the free tier's requests-per-minute limit
        return reply.text
    if backend == "manual":
        print("\n" + "=" * 70)
        print("Paste SYSTEM_PROMPT (once per chat), then send this problem:\n")
        print(user_text)
        print("\nPaste the chatbot's JSON reply below, then type END on its own line:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)
    raise ValueError(f"unknown backend {backend}")


# ---------------------------------------------------------------- self test
def good(**over):
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


def selftest():
    t = {x["name"]: x for x in TESTS}
    nonlinear = {"m": {"value": 2, "known": True}, "c": {"value": 0.3, "known": True},
                 "k": {"value": 0.3, "known": True}, "alpha": {"value": 0.2, "known": True}}
    inverse = {"m": {"value": 2, "known": True}, "c": {"value": None, "known": False},
               "k": {"value": 0.3, "known": True}}
    no_eb = {"damping_ratio": None, "regime": None}
    cases = [
        # (label, output, test, should_pass)
        ("good answer to test 1", good(), t["1_complete"], True),
        ("good answer to test 2",
         good(status="needs_clarification", parameters=nonlinear, expected_behavior=no_eb,
              equation={"order": 2, "variable": "t", "unknown": "x",
                        "form": "m*x'' + c*x' + k*x + alpha*x**3 = 0"},
              initial_conditions={"time": 1, "x": 1, "x_dot": 0, "source": "assumed"},
              time_interval=[1, 10], assumptions=["assumed released from x=1 at rest"],
              questions_for_user=["How was the spring released? At t=0 or t=1?"]),
         t["2_missing_initial_conditions"], True),
        ("good answer to test 3",
         good(problem_type="inverse", parameters=inverse, expected_behavior=no_eb,
              observed_data={"t": [1, 2, 3], "x": [0.8, 0.5, 0.1]}),
         t["3_unknown_damping_with_data"], True),
        ("good answer to test 4",
         good(status="needs_clarification", problem_type="inverse", parameters=inverse,
              expected_behavior=no_eb, questions_for_user=["Please give (t, x) measurements."]),
         t["4_unknown_damping_no_data"], True),
        # these are wrong on purpose, the checker must catch each one
        ("invented value (k=0.5)",
         good(parameters={**good()["parameters"], "k": {"value": 0.5, "known": True}}),
         t["1_complete"], False),
        ("wrong damping ratio (0.5)",
         good(expected_behavior={"damping_ratio": 0.5, "regime": "underdamped"}),
         t["1_complete"], False),
        ("silently assumed initial conditions on test 2 (status ok)",
         good(parameters=nonlinear, expected_behavior=no_eb,
              initial_conditions={"time": 0, "x": 1, "x_dot": 0, "source": "assumed"}),
         t["2_missing_initial_conditions"], False),
        ("first order equation",
         good(equation={"order": 1, "variable": "t", "unknown": "x", "form": "c*x' + k*x = 0"}),
         t["1_complete"], False),
        ("unknown c but marked forward",
         good(problem_type="forward", parameters=inverse, expected_behavior=no_eb,
              observed_data={"t": [1], "x": [0.8]}),
         t["3_unknown_damping_with_data"], False),
    ]
    failed = 0
    for label, out, test, should_pass in cases:
        problems = check(out, test)
        ok = (not problems) == should_pass
        failed += (not ok)
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
            out = extract_json(call_llm(test["text"], backend))
            problems = check(out, test)
        except Exception as e:  # bad JSON, API error, ...
            problems = [f"could not read the reply: {e}"]
        results.append((test["name"], problems))
        saved.append({"test": test["name"], "input": test["text"],
                      "output": out, "problems": problems})
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    print("Saved every input and output to results.json")
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
