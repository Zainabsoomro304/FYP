"""
ui_backend.py
=============

The bridge between the web UI (app.py) and the existing agent pipeline.
It contains no PINN and no copy of the workflow: every run calls
run_all.run_system() (agents 1-4 + the PINN code) inside ui_worker.py.

* events      -- run_system(on_event=...) reports each step; the worker appends
                 every event as one JSON line to <run>/events.jsonl
* start_run() -- launches ui_worker.py as its own process (argument list, no
                 shell), so training never freezes the UI
* readers     -- events, training-log progress, the project's own plot files
* build_run_view() -- turns the events into what the UI shows

Event line format:
    {"seq": 7, "time": "2026-10-21T17:03:12", "elapsed_s": 41.2,
     "type": "attempt_finished", "agent": 3, "attempt": 1, "data": {...}}
"""
import json
import os
import re
import subprocess
import sys
import time
import uuid

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_DIR, "agent_runs")
WORKER = os.path.join(PROJECT_DIR, "ui_worker.py")

EVENTS_FILE = "events.jsonl"
REQUEST_FILE = "request.json"
RESULTS_FILE = "run_all_results.json"
CONSOLE_FILE = "pipeline_console.txt"

SECRET_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")

# The real LLM backends of ode_agent.call_llm(). The UI offers only these.
BACKENDS = {
    "gemini": {"label": "Gemini", "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "package": "google.genai"},
    "anthropic": {"label": "Claude", "env": ("ANTHROPIC_API_KEY",), "package": "anthropic"},
}
# Automated tests may replace the LLM with run_all's own fake (the CLI self-test
# AI). The worker accepts it ONLY when this environment variable is set, so the
# UI can never fall back to it.
TEST_BACKEND = "test_fake_llm"
TEST_BACKEND_FLAG = "PINN_UI_ALLOW_TEST_BACKEND"

AGENT_NAMES = {1: "ODE Formulation", 2: "PINN Design", 3: "Code Execution",
               4: "Feedback & Validation"}


# =========================================================================== secrets
def secret_values(env=None):
    env = os.environ if env is None else env
    return [env[name] for name in SECRET_ENV_VARS if len(env.get(name, "")) >= 8]


def redact(obj, secrets):
    """Replace any secret value inside strings (recursively) with '***'."""
    if not secrets:
        return obj
    if isinstance(obj, str):
        for s in secrets:
            obj = obj.replace(s, "***")
        return obj
    if isinstance(obj, dict):
        return {k: redact(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v, secrets) for v in obj]
    return obj


def load_dotenv(path=None):
    """Read KEY=VALUE lines from .env into os.environ (existing values win).

    Returns the NAMES that were loaded, never the values.
    """
    path = path or os.path.join(PROJECT_DIR, ".env")
    loaded = []
    if not os.path.isfile(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if name and value and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return loaded


def _module_available(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def backend_status(name):
    """Can this backend run? Reports only whether a key is SET, never its value."""
    info = BACKENDS[name]
    key_set = any(os.environ.get(v) for v in info["env"])
    package_ok = _module_available(info["package"])
    if not package_ok:
        reason = f"the Python package '{info['package']}' is not installed"
    elif not key_set:
        reason = f"no API key ({' or '.join(info['env'])})"
    else:
        reason = None
    return {"available": reason is None, "reason": reason}


def available_backends():
    return [name for name in BACKENDS if backend_status(name)["available"]]


# =========================================================================== events
class EventWriter:
    """Callable passed to run_all.run_system(on_event=...). Appends JSON lines."""

    def __init__(self, path, secrets=()):
        self.path, self.secrets, self.seq, self.start = path, list(secrets), 0, time.time()

    def __call__(self, event_type, agent=None, attempt=None, **data):
        self.seq += 1
        event = {"seq": self.seq, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "elapsed_s": round(time.time() - self.start, 3), "type": event_type,
                 "agent": agent, "attempt": attempt, "data": redact(_jsonable(data), self.secrets)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        return event


def _jsonable(obj):
    """Round-trip through JSON so numpy numbers etc. become plain values."""
    def default(value):
        try:
            import numpy as np
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except ImportError:
            pass
        return str(value)
    return json.loads(json.dumps(obj, default=default))


def read_events(run_dir):
    """All complete events of a run (a half-written last line is ignored)."""
    path = os.path.join(run_dir, EVENTS_FILE)
    events = []
    if not os.path.isfile(path):
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_text_tail(path, max_chars=20000):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-max_chars:]
    except OSError:
        return ""


# =========================================================================== running
def start_run(request):
    """Write request.json and launch ui_worker.py. Returns (run_dir, Popen).

    request: {"text", "backend", "max_attempts", "first_epochs"} -- the same
    inputs as `python run_all.py --backend ... --text ... --max-attempts ...
    --epochs ...`. It never contains secrets: keys stay in the environment.
    """
    run_dir = os.path.join(RUNS_DIR, "ui_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4])
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, REQUEST_FILE), "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    console = open(os.path.join(run_dir, CONSOLE_FILE), "w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-u", WORKER, "--run-dir", run_dir],
                               cwd=PROJECT_DIR, env=env, stdout=console, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    console.close()                                  # the child keeps its own handle
    return run_dir, process


def worker_alive(events, run_dir):
    """Is this run's worker still running? True / False / None (cannot tell).

    Used when the UI has no process handle (e.g. after a server restart). The PID
    must belong to ui_worker.py for THIS run, so a recycled PID is never mistaken.
    """
    pid = next((e["data"].get("pid") for e in events if e["type"] == "run_started"), None)
    if not pid:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        cmdline = " ".join(psutil.Process(int(pid)).cmdline())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return None
    return "ui_worker.py" in cmdline and os.path.basename(run_dir) in cmdline


def cancel_run(run_dir, process=None):
    """Stop a running worker and record that the user cancelled it."""
    events = read_events(run_dir)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    elif process is None and worker_alive(events, run_dir):
        import psutil
        pid = next(e["data"]["pid"] for e in events if e["type"] == "run_started")
        worker = psutil.Process(int(pid))
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except psutil.TimeoutExpired:
            worker.kill()
    writer = EventWriter(os.path.join(run_dir, EVENTS_FILE))
    writer.seq = events[-1]["seq"] if events else 0
    if events and events[0].get("time"):             # keep elapsed_s on the worker's clock
        writer.start = time.mktime(time.strptime(events[0]["time"], "%Y-%m-%dT%H:%M:%S"))
    writer("run_cancelled", message="cancelled by the user")


# =========================================================================== progress + plots
EPOCH_LINE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+total\s+(?P<total>\S+)\s+\|\s+physics\s+(?P<physics>\S+)"
    r"\s+\|\s+IC\s+(?P<ic>\S+)(?:\s+\|\s+data\s+(?P<data>\S+))?(?:\s+\|\s+c_est\s+(?P<c_est>\S+))?")


def training_progress(attempt_dir):
    """Loss lines that training really printed to training_log.txt (may be empty)."""
    rows = []
    if not attempt_dir:
        return rows
    text = read_text_tail(os.path.join(attempt_dir, "training_log.txt"), max_chars=2_000_000)
    for match in EPOCH_LINE.finditer(text):
        row = {}
        for key, value in match.groupdict().items():
            if value is not None:
                try:
                    row[key] = int(value) if key == "epoch" else float(value)
                except ValueError:
                    pass
        rows.append(row)
    return rows


# Plot files written by main.py / plots.plot_results(), most important first.
PLOT_FILES = [("duffing_solution_comparison.png", "PINN prediction vs numerical reference"),
              ("duffing_absolute_error.png", "Absolute position error"),
              ("physics_residual.png", "Physics residual"),
              ("duffing_training_losses.png", "Training losses")]


def attempt_plots(attempt):
    """[(title, png path)] for one finished attempt, using the project's own plots.

    Forward problems: the PNGs main.py already saved. Inverse problems: the
    pipeline (code_agent._run_inverse) saves no plots, so they are drawn once
    from its saved arrays (solution_data.npz) with the SAME plots.py functions.
    """
    run_dir, summary = attempt.get("run_dir"), attempt.get("summary") or {}
    if not run_dir or not summary:
        return []
    folder = summary.get("plots_folder") or os.path.join(run_dir, "physics_only")
    if summary.get("mode") == "inverse":
        folder = os.path.join(run_dir, "plots")
        if not os.path.isfile(os.path.join(folder, PLOT_FILES[0][0])):
            draw_inverse_plots(attempt, folder)
    return [(title, os.path.join(folder, name)) for name, title in PLOT_FILES
            if os.path.isfile(os.path.join(folder, name))]


def draw_inverse_plots(attempt, folder):
    """Inverse attempt: existing plots.py figures from the saved arrays."""
    import numpy as np
    import plots
    from training import load_config
    summary = attempt["summary"]
    path = summary.get("data_file")
    if not path or not os.path.isfile(path):
        return
    with np.load(path) as saved:
        d = {name: saved[name] for name in saved.files}
    cfg = load_config(attempt["settings"])
    history = {name[len("history_"):]: d[name] for name in d if name.startswith("history_")}
    subtitle = plots.describe_system(
        cfg, damping_text=f"c estimated = {summary['estimated_c']:.4f} N*s/m")
    # The reference for user measurements is SciPy solved WITH THE ESTIMATED c
    # (code_agent._run_inverse); the label says so.
    plots.plot_solution_comparison(d["t"], d["x_pred"], d["x_reference"], subtitle, folder,
                                   reference_label="Numerical reference (SciPy, estimated c)")
    plots.plot_absolute_error(d["t"], d["x_pred"], d["x_reference"], subtitle, folder)
    plots.plot_training_losses(history, cfg, subtitle, folder)
    plots.plot_physics_residual(d["t"], d["residual"], subtitle, folder)


# =========================================================================== view model
def build_run_view(events, process_alive=None):
    """What the UI shows about one run, derived only from its events.

    process_alive: True/False if known, None if it cannot be determined.
    """
    agents = {n: {"number": n, "name": AGENT_NAMES[n], "state": "pending", "output": None,
                  "problems": [], "error": None, "started_s": None, "busy_s": 0.0}
              for n in AGENT_NAMES}
    view = {"agents": agents, "attempts": [], "spec": None, "final": None, "stopped_at": None,
            "questions": [], "error": None, "backend": None, "request": {}, "finished": False,
            "start_time": events[0]["time"] if events else None, "events": events,
            "elapsed_s": events[-1]["elapsed_s"] if events else 0.0}
    attempts = {}

    def rec(n):
        return attempts.setdefault(n, {
            "attempt": n, "state": "training", "settings": None, "summary": None, "run_dir": None,
            "started_s": None, "finished_s": None, "feedback": None, "feedback_problems": [],
            "passed_by_rules": None, "failed_checks": [], "notes": [], "changes": None,
            "used_fallback": False, "change_reason": None, "previous_settings": None,
            "thresholds": None, "amplitude": None, "error": None})

    for e in events:
        kind, n, data = e["type"], e.get("attempt"), e.get("data") or {}
        a = agents.get(e.get("agent"))
        if kind == "run_started":
            view["backend"], view["request"] = data.get("backend"), data.get("request", {})
        elif kind == "agent_started" and a:
            a["state"], a["started_s"] = "running", e["elapsed_s"]
        elif kind == "agent_finished" and a:
            a["busy_s"] += e["elapsed_s"] - (e["elapsed_s"] if a["started_s"] is None else a["started_s"])
            a["output"], a["problems"] = data.get("output"), data.get("problems") or []
            a["state"] = "failed" if a["problems"] and a["number"] != 4 else "completed"
            if a["number"] == 4:
                r = rec(n)
                r.update(feedback=data.get("output"), feedback_problems=data.get("problems") or [],
                         passed_by_rules=data.get("passed_by_rules"),
                         failed_checks=data.get("failed_checks") or [], notes=data.get("notes") or [],
                         thresholds=data.get("thresholds"), amplitude=data.get("amplitude"),
                         state="passed" if data.get("passed_by_rules") else "failed_checks")
        elif kind == "agent_failed" and a:
            a["state"], a["error"] = "failed", data.get("error")
        elif kind == "spec_built":
            view["spec"] = data
        elif kind == "attempt_started":
            rec(n).update(settings=data.get("settings"), run_dir=data.get("run_dir"),
                          started_s=e["elapsed_s"], state="training")
            agents[3]["state"] = "retrying" if n > 1 else "running"
        elif kind == "attempt_finished":
            r = rec(n)
            r.update(summary=data.get("summary"), finished_s=e["elapsed_s"], state="trained")
            agents[3]["busy_s"] += e["elapsed_s"] - (e["elapsed_s"] if r["started_s"] is None else r["started_s"])
            agents[3]["state"] = "completed"
        elif kind == "attempt_failed":
            rec(n).update(state="error", error=data.get("error"), finished_s=e["elapsed_s"])
            agents[3]["state"], agents[3]["error"] = "failed", data.get("error")
        elif kind == "retry_scheduled":
            rec(n).update(changes=data.get("changes"), used_fallback=data.get("used_fallback"),
                          change_reason=data.get("reason"),
                          previous_settings=data.get("previous_settings"), state="retry")
            agents[4]["state"] = "retrying"
        elif kind == "run_finished":
            view.update(finished=True, final=data.get("final"), stopped_at=data.get("stopped_at"),
                        questions=data.get("questions") or [])
            stopped_agent = agents.get(e.get("agent"))
            if stopped_agent and view["stopped_at"] and stopped_agent["state"] in ("running", "completed") \
                    and "more information" not in view["stopped_at"]:
                stopped_agent["state"] = "failed"
        elif kind in ("run_error", "run_cancelled"):
            view["finished"] = True
            view["error"] = dict(data, cancelled=kind == "run_cancelled")
            for agent in agents.values():
                if agent["state"] in ("running", "retrying"):
                    agent["state"] = "failed"
            for r in attempts.values():
                if r["state"] == "training":
                    r["state"] = "error"

    view["attempts"] = [attempts[k] for k in sorted(attempts)]
    view["status"] = run_status(view, process_alive)
    return view


def run_status(view, process_alive):
    """ready | running | retrying | completed | failed | cancelled"""
    if view["error"]:
        return "cancelled" if view["error"].get("cancelled") else "failed"
    if view["finished"]:
        return "completed" if view["final"] in ("pass", "did not pass") else "failed"
    if not view["events"]:
        # A worker that has just been launched has not written its first event
        # yet: it is running, so the UI must keep refreshing.
        return {True: "running", False: "failed"}.get(process_alive, "ready")
    if process_alive is False:
        return "failed"                      # worker ended without reporting a result
    return "retrying" if len(view["attempts"]) > 1 else "running"


def load_run_view(run_dir, process_alive=None):
    events = read_events(run_dir)
    if process_alive is None and events:
        process_alive = worker_alive(events, run_dir)
    view = build_run_view(events, process_alive)
    view["run_dir"] = run_dir
    view["process_alive"] = process_alive
    if not view["request"]:
        view["request"] = _read_json(os.path.join(run_dir, REQUEST_FILE)) or {}
    return view


# =========================================================================== example prompts
def _decimal(value):
    """Plain decimal text that agent 1's checker can read back (no 1e-05 forms)."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def example_prompts():
    """Suggested problem statements. All numbers are written out, as agent 1 requires.

    The measurements in the last example were generated with the SciPy reference
    solver using c = 0.3 N*s/m; that value is deliberately not in the text.
    """
    import numpy as np
    from physics import solve_reference_ode
    t = np.linspace(0.0, 10.0, 21)[1:]
    x, _ = solve_reference_ode(t, 1.0, 1.0, 0.3, 0.0, 1.0, 0.0, 0.0)
    pairs = "; ".join(f"t = {_decimal(round(ti, 4))} s: x = {_decimal(round(xi, 4))} m"
                      for ti, xi in zip(t, x))
    release = "It is released from x = 1 m with velocity 0 m/s at t = 0 s."
    return {
        "Simple harmonic oscillator":
            "Simulate an undamped mass-spring system with mass 1 kg and spring constant 1 N/m. "
            "There is no damping (damping coefficient c = 0 N*s/m) and no nonlinear term. "
            f"{release} Simulate from 0 to 10 s.",
        "Damped oscillator":
            "Simulate a damped mass-spring system with mass 1 kg, spring constant 4 N/m and "
            f"damping coefficient 0.5 N*s/m. The spring is linear. {release} Simulate from 0 to 10 s.",
        "Nonlinear Duffing oscillator":
            "Simulate a nonlinear Duffing oscillator m*x'' + c*x' + k*x + alpha*x^3 = 0 with mass "
            "1 kg, damping coefficient 0.1 N*s/m, spring constant 1 N/m and nonlinear coefficient "
            f"alpha = 0.5 N/m^3. {release} Simulate from 0 to 10 s.",
        "Estimate unknown damping":
            "A linear mass-spring system with mass 1 kg and spring constant 1 N/m has an unknown "
            "damping coefficient c that must be estimated. There is no nonlinear term. "
            f"{release} I measured these positions: {pairs}. Simulate from 0 to 10 s.",
    }
