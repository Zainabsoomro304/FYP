"""
ui_worker.py
============

Runs ONE pipeline run for the web UI, in its own process:

    python ui_worker.py --run-dir agent_runs/ui_20261021_170000_ab12

It reads <run-dir>/request.json and calls the existing run_all.run_system()
exactly as `python run_all.py` does (same text, backend, --max-attempts,
--epochs), adding only an event callback. It writes:
    events.jsonl          every step, one JSON object per line (see ui_backend.py)
    run_all_results.json  the same results file the CLI writes
    attempt_N/            the PINN outputs (plots, metrics.json, training_log.txt)

API keys come from the environment (or .env); they are never written to any
file and every event is scrubbed of their values.
"""
import argparse
import json
import os
import sys
import time
import traceback

import ui_backend


def make_llm(request, emit):
    """The llm(user_text, system) function that run_system() expects."""
    import feedback_agent
    import ode_agent
    import pinn_agent
    import run_all

    backend = request["backend"]
    if backend == ui_backend.TEST_BACKEND:
        # Automated tests only: run_all's own self-test fake AI, fed with an
        # agent-1 JSON from the test. Refused unless the test flag is set.
        if os.environ.get(ui_backend.TEST_BACKEND_FLAG) != "1":
            raise RuntimeError("the test backend is disabled outside automated tests")
        inner = run_all.make_fake_llm(request["test_ode_json"])
    elif backend in ui_backend.BACKENDS:
        status = ui_backend.backend_status(backend)
        if not status["available"]:
            raise RuntimeError(f"{ui_backend.BACKENDS[backend]['label']} is not configured: "
                               f"{status['reason']}")

        def inner(user_text, system):
            return ode_agent.call_llm(user_text, backend, system=system)
    else:
        raise RuntimeError(f"unknown backend {backend!r}")

    agent_of_prompt = {ode_agent.SYSTEM_PROMPT: 1, pinn_agent.SYSTEM_PROMPT: 2,
                       feedback_agent.SYSTEM_PROMPT: 4}

    def llm(user_text, system):
        agent = agent_of_prompt.get(system)
        start = time.time()
        try:
            reply = inner(user_text, system)
        except Exception as e:
            emit("llm_call", agent=agent, ok=False, duration_s=round(time.time() - start, 2),
                 error=f"{type(e).__name__}: {e}")
            raise
        emit("llm_call", agent=agent, ok=True, duration_s=round(time.time() - start, 2),
             reply=(reply or "")[:6000])
        return reply

    return llm


def run(run_dir):
    request = ui_backend._read_json(os.path.join(run_dir, ui_backend.REQUEST_FILE))
    if request is None:
        raise SystemExit(f"no {ui_backend.REQUEST_FILE} in {run_dir}")
    ui_backend.load_dotenv()
    emit = ui_backend.EventWriter(os.path.join(run_dir, ui_backend.EVENTS_FILE),
                                  secrets=ui_backend.secret_values())
    emit("run_started", backend=request.get("backend"), request=request, pid=os.getpid())
    try:
        llm = make_llm(request, emit)
        import run_all
        result = run_all.run_system(request["text"], llm, run_dir,
                                    max_attempts=int(request.get("max_attempts", 3)),
                                    first_epochs=request.get("first_epochs") or None,
                                    on_event=emit)
        with open(os.path.join(run_dir, ui_backend.RESULTS_FILE), "w", encoding="utf-8") as f:
            json.dump(ui_backend.redact(result, ui_backend.secret_values()), f, indent=2, default=str)
        # Inverse attempts save no plots of their own; draw them now (with plots.py)
        # so they are ready when the UI shows the result.
        for attempt in ui_backend.load_run_view(run_dir, process_alive=True)["attempts"]:
            try:
                ui_backend.attempt_plots(attempt)
            except Exception as e:                   # a plot problem must not fail the run
                print(f"could not draw plots for attempt {attempt['attempt']}: {e}", flush=True)
        return 0
    except Exception as e:
        emit("run_error", error_type=type(e).__name__, message=str(e),
             traceback=traceback.format_exc()[-4000:])
        print(f"run failed: {type(e).__name__}: {e}", flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run one pipeline run for the web UI")
    parser.add_argument("--run-dir", required=True)
    sys.exit(run(os.path.abspath(parser.parse_args().run_dir)))


if __name__ == "__main__":
    main()
