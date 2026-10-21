"""
app.py -- Agentic PINN (web UI)

    streamlit run app.py

A thin UI over the existing pipeline. Each run is `run_all.run_system()` (agents
1-4 and the PINN code) executed by ui_worker.py in its own process; this file
only starts runs, shows what they report, and displays the plots the project
itself produced.
"""
import html
import os
import time

import pandas as pd
import streamlit as st

import feedback_agent
import ui_backend as backend

st.set_page_config(page_title="Agentic PINN", page_icon=":material/function:", layout="wide",
                   initial_sidebar_state="collapsed")

POLL_SECONDS = 3
STATUS = {"ready": ("Ready", "#6b7280"), "running": ("Running", "#2563eb"),
          "retrying": ("Retrying", "#b45309"), "completed": ("Completed", "#047857"),
          "failed": ("Failed", "#b91c1c"), "cancelled": ("Cancelled", "#6b7280")}
STAGE = {"pending": ("Pending", "#9ca3af"), "running": ("Running", "#2563eb"),
         "retrying": ("Retrying", "#b45309"), "completed": ("Completed", "#047857"),
         "failed": ("Failed", "#b91c1c")}


def css():
    st.markdown("""
<style>
.block-container { max-width: 1080px; padding-top: 2.6rem; padding-bottom: 4rem; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
h1.title { font-size: 1.6rem; font-weight: 650; margin: 0; padding: 0; color: #111827; }
.subtitle { color: #6b7280; margin: 2px 0 0 0; font-size: 0.95rem; }
.status { display: inline-flex; align-items: center; gap: 7px; font-size: 0.85rem;
          font-weight: 600; }
.status .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status .live { animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .3; } }
.muted { color: #6b7280; font-size: 0.88rem; }
.stage { border-left: 3px solid; padding: 2px 0 2px 14px; margin: 18px 0 4px 0; }
.stage .head { display: flex; justify-content: space-between; align-items: baseline; }
.stage .name { font-weight: 600; color: #111827; }
.stage .state { font-size: 0.8rem; font-weight: 600; }
.stage .time { color: #9ca3af; font-size: 0.8rem; margin-left: 10px; font-weight: 400; }
.stage .desc { color: #4b5563; font-size: 0.9rem; margin-top: 2px; }
.loop { color: #92400e; font-size: 0.88rem; margin: 14px 0 0 17px; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; }
</style>""", unsafe_allow_html=True)


# =============================================================================== state
@st.cache_resource
def process_registry():
    """Worker processes started by this server: {run_dir: Popen}."""
    return {}


def current_view():
    run_dir = st.session_state.get("run_dir")
    if not run_dir or not os.path.isdir(run_dir):
        return None
    process = process_registry().get(run_dir)
    return backend.load_run_view(run_dir, None if process is None else process.poll() is None)


def is_active(view):
    return view is not None and view["status"] in ("running", "retrying")


def any_run_active():
    if any(p.poll() is None for p in process_registry().values()):
        return True
    return is_active(current_view())


def init_state():
    ss = st.session_state
    if "initialised" in ss:
        return
    ss.initialised = True
    ss.problem_text = ""
    ss.run_dir = None
    ss.last_status = None
    backend.load_dotenv()
    # ?run=<folder in agent_runs> reopens a saved run (plain folder names only)
    name = st.query_params.get("run")
    if name and name == os.path.basename(name) and os.path.isdir(os.path.join(backend.RUNS_DIR, name)):
        ss.run_dir = os.path.join(backend.RUNS_DIR, name)


def fmt_time(seconds):
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


def sci(value):
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return "—"


def elapsed_now(view):
    if not view or not view.get("start_time"):
        return None
    if view["finished"] or not is_active(view):
        return view["elapsed_s"]
    return time.time() - time.mktime(time.strptime(view["start_time"], "%Y-%m-%dT%H:%M:%S"))


# =============================================================================== header
def status_html(status):
    label, color = STATUS.get(status, STATUS["ready"])
    live = " live" if status in ("running", "retrying") else ""
    return (f'<span class="status" style="color:{color}"><span class="dot{live}" '
            f'style="background:{color}"></span>{label}</span>')


def header():
    view = current_view()
    status = view["status"] if view else "ready"
    left, right = st.columns([4, 1])
    left.markdown('<h1 class="title">Agentic PINN</h1><p class="subtitle">AI agents formulate an '
                  'ODE from your description, solve it with a physics-informed neural network, '
                  'and validate the result against a numerical solver.</p>', unsafe_allow_html=True)
    right.markdown(f'<div style="text-align:right;margin-top:8px">{status_html(status)}</div>',
                   unsafe_allow_html=True)
    ss = st.session_state
    if ss.last_status in ("running", "retrying") and status not in ("running", "retrying"):
        ss.last_status = status
        st.rerun(scope="app")                        # refresh everything once a run ends
    ss.last_status = status


# =============================================================================== simulation
def set_prompt(text):
    st.session_state.problem_text = text


def simulation_tab():
    ss = st.session_state
    prompts = backend.example_prompts()
    st.markdown('<span class="muted">Suggestions</span>', unsafe_allow_html=True)
    cols = st.columns(len(prompts))
    for col, (name, text) in zip(cols, prompts.items()):
        col.button(name, key=f"prompt_{name}", on_click=set_prompt, args=(text,), type="tertiary",
                   help="The measurements in this example were generated with the SciPy solver "
                        "using c = 0.3 N·s/m (not stated in the text)."
                   if name == "Estimate unknown damping" else None)
    st.text_area("Problem description", key="problem_text", height=150,
                 placeholder="Describe the oscillator: mass, stiffness, damping, any nonlinear "
                             "term, the initial position and velocity, and the time interval.")

    configured = backend.available_backends()
    with st.expander("Advanced", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Maximum attempts", 1, 5, 3, key="max_attempts",
                        help="How many training runs agent 4 may request (CLI: --max-attempts).")
        c2.number_input("Epochs for the first attempt", 0, 30000, 0, step=500, key="first_epochs",
                        help="0 = use agent 2's choice. Same as the CLI's --epochs.")
        if len(configured) > 1:
            c3.selectbox("Language model", configured, key="backend_name",
                         format_func=lambda n: backend.BACKENDS[n]["label"])
    backend_name = ss.get("backend_name") if ss.get("backend_name") in configured else \
        (configured[0] if configured else None)

    if not configured:
        st.info("No language model is configured. Set `GEMINI_API_KEY` (Gemini) or "
                "`ANTHROPIC_API_KEY` (Claude) in the environment or in a `.env` file in the "
                "project folder, then restart the app.", icon=":material/key:")

    busy = any_run_active()
    c1, c2, _ = st.columns([1.3, 1, 4])
    run = c1.button("Run simulation", type="primary", width="stretch",
                    disabled=busy or not configured or not ss.problem_text.strip())
    if busy and c2.button("Cancel", width="stretch"):
        for run_dir, process in list(process_registry().items()):
            if process.poll() is None:
                backend.cancel_run(run_dir, process)
        view = current_view()
        if is_active(view):
            backend.cancel_run(view["run_dir"], None)
        st.rerun()
    if run and not any_run_active():                 # re-check: never two runs at once
        request = {"text": ss.problem_text, "backend": backend_name,
                   "max_attempts": int(ss.max_attempts), "first_epochs": int(ss.first_epochs) or None}
        try:
            run_dir, process = backend.start_run(request)
        except OSError as e:
            st.error(f"Could not start the run: {e}")
            return
        process_registry()[run_dir] = process
        ss.run_dir, ss.last_status = run_dir, "running"
        st.rerun()

    view = current_view()
    if view is None:
        return
    st.fragment(run_every=POLL_SECONDS if is_active(view) else None)(results_section)()


def results_section():
    view = current_view()
    if view is None:
        return
    st.divider()
    attempts = [a for a in view["attempts"] if a.get("summary")]
    if is_active(view):
        n = len(view["attempts"])
        st.markdown(f'<span class="muted">Attempt {max(n, 1)} in progress · '
                    f'{fmt_time(elapsed_now(view))} elapsed. See Agent Activity for details.</span>',
                    unsafe_allow_html=True)
    if view["error"]:
        st.error("Run cancelled." if view["error"].get("cancelled") else
                 f"{view['error'].get('error_type', 'Error')}: {view['error'].get('message')}")
    elif view["stopped_at"]:
        message = view["stopped_at"]
        if view["questions"]:
            message += "\n\n" + "\n".join(f"- {q}" for q in view["questions"])
        st.warning(message)
    elif view["process_alive"] is False and not view["finished"]:
        st.error("The run stopped without reporting a result. See the debug log in Agent Activity.")
    if not attempts:
        return

    shown = attempts[-1]
    if len(attempts) > 1 and not is_active(view):
        labels = {a["attempt"]: f"Attempt {a['attempt']}" for a in attempts}
        choice = st.segmented_control("Attempt", list(labels), default=shown["attempt"],
                                      format_func=labels.get, key="result_attempt",
                                      label_visibility="collapsed")
        shown = next((a for a in attempts if a["attempt"] == choice), shown)
    s = shown["summary"]
    if shown.get("passed_by_rules") is None:
        result = "Judging…"
    else:
        result = "Passed" if shown["passed_by_rules"] else "Not passed"
    cols = st.columns(5)
    cols[0].metric("Validation", result, help="Decided by agent 4's fixed threshold rules.")
    cols[1].metric("Position RMSE", f"{sci(s.get('position_rmse_m'))} m", help=s.get("position_rmse_is"))
    cols[2].metric("Residual MSE", f"{sci(s.get('physics_residual_mse'))} N²")
    if s.get("mode") == "inverse":
        cols[3].metric("Estimated c", f"{s['estimated_c']:.4f} N·s/m")
        cols[4].metric("Data MSE", f"{sci(s.get('data_mse'))} m²")
    else:
        cols[3].metric("IC errors (x, v)", f"{sci(s.get('ic_position_error'))}, "
                                           f"{sci(s.get('ic_velocity_error'))}")
        cols[4].metric("Training time", fmt_time(s.get("training_time_s")))

    plots = backend.attempt_plots(shown)
    if not plots:
        st.caption("No plots were saved for this attempt.")
        return
    note = ("Reference: SciPy solve_ivp solved with the estimated c (a numerical approximation). "
            "Plots drawn from the saved results with the project's plots.py."
            if s.get("mode") == "inverse" else
            "Reference: SciPy solve_ivp (a high-accuracy numerical approximation, not an exact "
            "solution). Plots as saved by the project.")
    st.caption(note)
    st.image(plots[0][1], width="stretch")
    rest = plots[1:]
    for i in range(0, len(rest), 2):
        cols = st.columns(2)
        for col, (_, path) in zip(cols, rest[i:i + 2]):
            col.image(path, width="stretch")


# =============================================================================== agent activity
def agent_tab():
    view = current_view()
    if view is None:
        st.markdown('<p class="muted" style="margin-top:12px">Run a simulation to see the four '
                    'agents at work.</p>', unsafe_allow_html=True)
        return
    st.fragment(run_every=POLL_SECONDS if is_active(view) else None)(agent_timeline)()


def agent_timeline():
    view = current_view()
    if view is None:
        return
    attempts = view["attempts"]
    summary_line = [f"Attempt {len(attempts)}" if attempts else "",
                    f"{fmt_time(elapsed_now(view))} elapsed" if view.get("start_time") else ""]
    st.markdown(f'<span class="muted">{" · ".join(p for p in summary_line if p)}</span>',
                unsafe_allow_html=True)
    for n in (1, 2, 3, 4):
        stage(view, n)
    for a in [a for a in attempts if a.get("changes")]:
        prev = a.get("previous_settings") or {}
        change = ", ".join(f"{k} {prev.get(k)} → {v}" for k, v in a["changes"].items())
        who = "the fallback rule (agent 4's proposal was invalid)" if a.get("used_fallback") \
            else "agent 4"
        st.markdown(f'<div class="loop">↻ Retry after attempt {a["attempt"]}: {html.escape(change)}'
                    f' — requested by {who}.</div>', unsafe_allow_html=True)
    with st.expander("Debug log", expanded=False):
        console = backend.read_text_tail(os.path.join(view["run_dir"], backend.CONSOLE_FILE), 15000)
        st.code(console or "(no console output yet)", language="text")


def stage(view, n):
    a = view["agents"][n]
    state = a["state"]
    label, color = STAGE.get(state, STAGE["pending"])
    busy = a["busy_s"]
    if n == 3 and state in ("running", "retrying") and view["attempts"]:
        last = view["attempts"][-1]
        if last.get("started_s") is not None and last["state"] == "training":
            busy += max((elapsed_now(view) or 0) - last["started_s"], 0)
    timing = f'<span class="time">{fmt_time(busy)}</span>' if busy >= 1 else ""
    st.markdown(f'<div class="stage" style="border-color:{color}"><div class="head">'
                f'<span class="name">{n} · {backend.AGENT_NAMES[n]}</span>'
                f'<span class="state" style="color:{color}">{label}{timing}</span></div>'
                f'<div class="desc">{html.escape(describe(view, n))}</div></div>',
                unsafe_allow_html=True)
    if n == 3:
        live_progress(view)
    if state != "pending":
        with st.expander("Details", expanded=False):
            DETAILS[n](view)


def describe(view, n):
    """One line of what the agent actually did, built from its real output."""
    a = view["agents"][n]
    out = a.get("output") or {}
    if a.get("error"):
        return a["error"]
    if a["problems"] and n != 4:
        return "Its answer was rejected: " + a["problems"][0]
    if n == 1:
        if not out:
            return "Reads the problem text and writes the governing equation." \
                if a["state"] == "pending" else "Reading the problem text…"
        unknown = [k for k, p in (out.get("parameters") or {}).items() if not p.get("known", True)]
        text = f"{out['equation']['form']} · {out.get('problem_type')} problem"
        if unknown:
            text += f" · unknown: {', '.join(unknown)}"
        if out.get("status") != "ok":
            text += " · needs more information"
        return text
    if n == 2:
        if not out:
            return "Chooses the network, collocation points and losses." \
                if a["state"] == "pending" else "Designing the PINN…"
        if out.get("status") != "ok":
            return f"Did not design a network ({out.get('status')})."
        arch, opt = out["architecture"], out["optimizer"]
        text = (f"MLP {arch['hidden_layers']}×{arch['neurons_per_layer']} {arch['activation']}, "
                f"{out['collocation']['n_points']} collocation points, {opt['epochs']:,} epochs, "
                f"learning rate {opt['learning_rate']:g}")
        if out.get("trainable_parameters"):
            text += f"; learns {', '.join(out['trainable_parameters'])}"
        return text
    if n == 3:
        if not view["attempts"]:
            return "Maps the design onto the PINN code and trains it." \
                if a["state"] == "pending" else "Preparing the solver settings…"
        last = view["attempts"][-1]
        epochs = (last.get("settings") or {}).get("EPOCHS") or 0
        if last["state"] == "training":
            return f"Training attempt {last['attempt']} ({epochs:,} epochs)…"
        if last["state"] == "error":
            return f"Attempt {last['attempt']} failed: {last.get('error')}"
        done = [r for r in view["attempts"] if r.get("summary")]
        return f"Trained {len(done)} attempt(s); the last one used {epochs:,} epochs."
    judged = [r for r in view["attempts"] if r.get("passed_by_rules") is not None]
    if not judged:
        return "Checks the errors against fixed thresholds and decides: pass or retry." \
            if a["state"] == "pending" else "Judging the results…"
    last = judged[-1]
    if last["passed_by_rules"]:
        return f"Attempt {last['attempt']}: all checks passed."
    tail = " Retry requested." if last.get("changes") else \
        (" No attempts left." if view["final"] == "did not pass" else "")
    return f"Attempt {last['attempt']}: {'; '.join(last['failed_checks'])}.{tail}"


def live_progress(view):
    """Genuine training progress: the epochs that training itself printed to its log."""
    if not view["attempts"] or view["attempts"][-1]["state"] != "training":
        return
    last = view["attempts"][-1]
    rows = backend.training_progress(last.get("run_dir"))
    settings = last.get("settings") or {}
    epochs, every = settings.get("EPOCHS") or 0, settings.get("PRINT_EVERY")
    if rows and epochs:
        r = rows[-1]
        st.progress(min(r["epoch"] / epochs, 1.0),
                    text=f"epoch {r['epoch']:,} / {epochs:,} · total loss {r['total']:.2e}"
                         + (f" · c estimate {r['c_est']:.4f}" if "c_est" in r else "")
                         + (f" · the log reports every {every:,} epochs" if every else ""))
    else:
        st.caption("Training started; progress appears after the first logged epoch.")


# ----------------------------------------------------------------------- stage details
def details_agent1(view):
    out = view["agents"][1].get("output") or {}
    st.markdown("**Problem statement**")
    st.text(view["request"].get("text", ""))
    problems(view["agents"][1])
    if not out:
        return
    st.markdown(f"**Equation** ({out.get('problem_type')} problem)")
    st.code(out["equation"]["form"], language="text")
    params = out.get("parameters") or {}
    st.dataframe(pd.DataFrame([{"parameter": k, "value": "unknown" if not p.get("known", True)
                                else p.get("value")} for k, p in params.items()]).astype(str),
                 hide_index=True)
    ic = out.get("initial_conditions") or {}
    st.markdown(f"Initial conditions: x({ic.get('time')}) = {ic.get('x')} m, "
                f"x'({ic.get('time')}) = {ic.get('x_dot')} m/s · interval {out.get('time_interval')} s")
    if out.get("observed_data"):
        st.markdown(f"Measurements extracted: {len(out['observed_data'].get('t', []))} points")
    for item in out.get("assumptions") or []:
        st.markdown(f"- Assumption: {item}")
    for item in out.get("questions_for_user") or []:
        st.markdown(f"- Question: {item}")


def details_agent2(view):
    out = view["agents"][2].get("output") or {}
    problems(view["agents"][2])
    if out.get("status") != "ok":
        return
    losses = out["losses"]
    rows = [{"loss": "physics residual", "weight": losses["physics"]["weight"]},
            {"loss": "initial conditions", "weight": losses["initial_condition"]["weight"]}]
    if losses.get("data"):
        rows.append({"loss": "measurement data", "weight": losses["data"]["weight"]})
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True)
    st.markdown(f"Residual: `{losses['physics']['residual']}` · derivatives by "
                f"{losses['physics'].get('derivatives')} · optimizer {out['optimizer']['name']}")
    if out.get("initial_guesses"):
        st.markdown(f"Starting guess: {out['initial_guesses']} (kept positive with "
                    f"{out.get('positive_parameterization')})")
    for item in out.get("notes") or []:
        st.markdown(f"- {item}")


SOLVER_KEYS = ["EXPERIMENT_MODE", "MASS", "SPRING_CONSTANT", "DAMPING_COEFFICIENT",
               "NONLINEAR_COEFFICIENT", "INITIAL_DAMPING_ESTIMATE", "INITIAL_POSITION",
               "INITIAL_VELOCITY", "TIME_START", "TIME_END", "N_COLLOCATION", "HIDDEN_SIZE",
               "NUM_HIDDEN_LAYERS", "EPOCHS", "LEARNING_RATE", "LAMBDA_PHYSICS", "LAMBDA_IC",
               "LAMBDA_DATA"]


def details_agent3(view):
    spec = view.get("spec")
    if not spec:
        return
    settings = spec.get("settings") or {}
    st.markdown("Settings passed to the PINN code (`config.py` names) for attempt 1:")
    st.dataframe(pd.DataFrame([{"setting": k, "value": settings[k]} for k in SOLVER_KEYS
                               if k in settings]).astype(str), hide_index=True)
    if spec.get("first_epochs"):
        st.caption(f"Epochs for attempt 1 set to {spec['first_epochs']:,} in Advanced "
                   f"(agent 2 chose {spec.get('planned_epochs'):,}).")
    for w in spec.get("warnings") or []:
        st.warning(w)
    for a in view["attempts"]:
        s = a.get("summary")
        if s:
            st.markdown(f"Attempt {a['attempt']}: {s.get('epochs'):,} epochs in "
                        f"{fmt_time(s.get('training_time_s'))}, best epoch {s.get('best_epoch')}")


def details_agent4(view):
    for a in [a for a in view["attempts"] if a.get("passed_by_rules") is not None]:
        st.markdown(f"**Attempt {a['attempt']}**: fixed rules say "
                    f"{'pass' if a['passed_by_rules'] else 'fail'}")
        st.dataframe(threshold_rows(a), hide_index=True)
        fb = a.get("feedback") or {}
        if fb.get("diagnosis"):
            st.markdown(f"Agent's diagnosis: {fb['diagnosis']}")
        if fb.get("changes") and fb.get("reason"):
            st.markdown(f"Agent's reason for the change: {fb['reason']}")
        for p in a.get("feedback_problems") or []:
            st.markdown(f"- Proposal rejected: {p}")
        for item in a.get("notes") or []:
            st.markdown(f"- {item}")


def threshold_rows(a):
    s, t = a.get("summary") or {}, a.get("thresholds") or feedback_agent.THRESHOLDS
    rows = [("position RMSE [m]", s.get("position_rmse_m"),
             t["position_error_vs_amplitude"] * (a.get("amplitude") or 1.0)),
            ("residual MSE [N²]", s.get("physics_residual_mse"), t["physics_residual_mse"]),
            ("IC position error [m]", s.get("ic_position_error"), t["ic_error"]),
            ("IC velocity error [m/s]", s.get("ic_velocity_error"), t["ic_error"])]
    if s.get("mode") == "inverse":
        rows.append(("data MSE [m²]", s.get("data_mse"), t["data_mse"]))
    return pd.DataFrame([{"check": name, "value": sci(v), "threshold": sci(limit),
                          "result": "pass" if v is not None and v <= limit else "fail"}
                         for name, v, limit in rows])


def problems(agent):
    for p in agent.get("problems") or []:
        st.markdown(f"- Rejected by the automatic check: {p}")


DETAILS = {1: details_agent1, 2: details_agent2, 3: details_agent3, 4: details_agent4}


# =============================================================================== main
def main():
    css()
    init_state()
    view = current_view()
    st.fragment(run_every=POLL_SECONDS if is_active(view) else None)(header)()
    tab_sim, tab_agents = st.tabs(["Simulation", "Agent Activity"])
    with tab_sim:
        simulation_tab()
    with tab_agents:
        agent_tab()


main()
