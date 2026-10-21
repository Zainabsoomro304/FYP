"""
Tests for the web-UI integration: run_all.run_system(on_event=...), the event
log, the run view, the worker process and the reuse of the project's plots.

Most tests replace the LLM with run_all's own self-test fake and the training
call with a stub, so they run in seconds. The worker tests run the real
worker process with real PINN training for a few hundred epochs.

    python -m unittest tests.test_ui_backend -v
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

import numpy as np  # noqa: E402

import code_agent  # noqa: E402
import feedback_agent  # noqa: E402
import pinn_agent  # noqa: E402
import run_all  # noqa: E402
import ui_backend  # noqa: E402
import ui_worker  # noqa: E402
from physics import solve_reference_ode  # noqa: E402

TEXT = ("Simulate a mass-spring-damper with m=2 kg, k=0.3 N/m, c=0.3 Ns/m, "
        "released from x=1 m at rest, from 0 to 10 s.")
GOOD = {"position_rmse_m": 1e-3, "physics_residual_mse": 1e-5, "ic_position_error": 1e-5,
        "ic_velocity_error": 1e-5}
BAD = {"position_rmse_m": 0.5, "physics_residual_mse": 0.05, "ic_position_error": 1e-5,
       "ic_velocity_error": 1e-5}


def summary(metrics, epochs=1000):
    return dict(metrics, mode="forward", position_rmse_is="PINN vs SciPy", best_epoch=epochs,
                training_time_s=1.0, epochs=epochs, log_file="none")


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event_type, agent=None, attempt=None, **data):
        self.events.append({"type": event_type, "agent": agent, "attempt": attempt, "data": data,
                            "seq": len(self.events) + 1, "elapsed_s": float(len(self.events)),
                            "time": "2026-10-21T10:00:00"})

    def types(self):
        return [e["type"] for e in self.events]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TestRunSystemEvents(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out, True)
        self.llm = run_all.make_fake_llm(pinn_agent.ode())

    def run_pipeline(self, metrics_per_attempt, llm=None, **kwargs):
        summaries = iter(summary(m) for m in metrics_per_attempt)
        rec = Recorder()
        with mock.patch.object(code_agent, "run_spec", side_effect=lambda spec, d: next(summaries)):
            result = quiet(run_all.run_system, TEXT, llm or self.llm, self.out, on_event=rec,
                           **kwargs)
        return result, rec

    def test_cli_output_is_unchanged_with_epochs_override(self):
        """Regression: `run_all.py --epochs N` must not add keys or lines (it once did)."""
        console = io.StringIO()
        with mock.patch.object(code_agent, "run_spec", return_value=summary(GOOD)), \
                contextlib.redirect_stdout(console):
            result = run_all.run_system(TEXT, self.llm, self.out, first_epochs=600)
        self.assertEqual(set(result), {"input", "stopped_at", "attempts", "final", "ode", "pinn"})
        self.assertNotIn("user settings", console.getvalue())
        self.assertEqual(result["attempts"][0]["settings"]["EPOCHS"], 600)

    def test_forward_pass_event_sequence(self):
        result, rec = self.run_pipeline([GOOD])
        self.assertEqual(result["final"], "pass")
        self.assertEqual(rec.types(), [
            "agent_started", "agent_finished", "agent_started", "agent_finished",
            "agent_started", "spec_built", "attempt_started", "attempt_finished",
            "agent_started", "agent_finished", "run_finished"])
        self.assertEqual(rec.events[1]["data"]["output"], pinn_agent.ode())   # real agent-1 JSON
        view = ui_backend.build_run_view(rec.events, process_alive=False)
        self.assertEqual(view["status"], "completed")
        self.assertTrue(all(a["state"] == "completed" for a in view["agents"].values()))
        self.assertGreater(view["agents"][1]["busy_s"], 0)        # real elapsed time per agent

    def test_failed_validation_triggers_retry_with_recorded_changes(self):
        result, rec = self.run_pipeline([BAD, GOOD])
        self.assertEqual(result["final"], "pass")
        retry = next(e for e in rec.events if e["type"] == "retry_scheduled")
        self.assertEqual(retry["data"]["changes"], {"EPOCHS": 20000})   # fake agent doubles 10000
        self.assertEqual(retry["data"]["previous_settings"]["EPOCHS"], 10000)
        second = [e for e in rec.events if e["type"] == "attempt_started"][1]
        self.assertEqual(second["data"]["settings"]["EPOCHS"], 20000)   # the change really applied
        view = ui_backend.build_run_view(rec.events[:-4], process_alive=True)   # up to attempt 2 start
        self.assertEqual(view["status"], "retrying")                    # attempt 2 running
        self.assertEqual(view["agents"][3]["state"], "retrying")
        view = ui_backend.build_run_view(rec.events, process_alive=False)
        self.assertEqual(view["attempts"][0]["state"], "retry")
        self.assertTrue(view["attempts"][0]["failed_checks"])
        self.assertEqual(view["status"], "completed")

    def test_out_of_attempts_is_completed_but_not_passed(self):
        result, rec = self.run_pipeline([BAD, BAD], max_attempts=2)
        self.assertEqual(result["final"], "did not pass")
        view = ui_backend.build_run_view(rec.events, process_alive=False)
        self.assertEqual(view["status"], "completed")
        self.assertFalse(view["attempts"][-1]["passed_by_rules"])

    def test_invalid_feedback_proposal_uses_fallback_and_says_so(self):
        base = run_all.make_fake_llm(pinn_agent.ode())

        def llm(user_text, system):
            if system == feedback_agent.SYSTEM_PROMPT:
                return json.dumps({"verdict": "retry", "diagnosis": "x",
                                   "changes": {"SEED": 7}, "reason": "not allowed"})
            return base(user_text, system)

        _, rec = self.run_pipeline([BAD, GOOD], llm=llm)
        retry = next(e for e in rec.events if e["type"] == "retry_scheduled")
        self.assertTrue(retry["data"]["used_fallback"])
        self.assertIsNone(retry["data"]["reason"])                      # no invented reason

    def test_api_failure_is_reported_not_hidden(self):
        def llm(user_text, system):
            raise ConnectionError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred")

        result, rec = self.run_pipeline([], llm=llm)
        self.assertEqual(result["stopped_at"], "agent 1 could not produce a valid answer")
        self.assertIn("SSL", rec.events[1]["data"]["problems"][0])
        view = ui_backend.build_run_view(rec.events, process_alive=False)
        self.assertEqual(view["status"], "failed")
        self.assertEqual(view["agents"][1]["state"], "failed")
        self.assertEqual(view["agents"][2]["state"], "pending")         # never reached

    def test_unsupported_equation_stops_at_agent_3(self):
        params = {k: v for k, v in pinn_agent.NONLINEAR.items() if k != "alpha"}
        params["beta"] = pinn_agent.NONLINEAR["alpha"]
        ode = pinn_agent.ode(parameters=params, expected_behavior=pinn_agent.NO_EB,
                             equation={"order": 2, "variable": "t", "unknown": "x",
                                       "form": "m*x'' + c*x' + k*x + beta*x^3 = 0"})
        text = ("Simulate a spring with m=2 kg, k=0.3 N/m, c=0.3 Ns/m and nonlinear coefficient "
                "0.2, released from x=1 m at rest, from 0 to 10 s.")
        rec = Recorder()
        result = quiet(run_all.run_system, text, run_all.make_fake_llm(ode), self.out, on_event=rec)
        self.assertTrue(result["stopped_at"].startswith("agent 3 cannot run this"))
        view = ui_backend.build_run_view(rec.events, process_alive=False)
        self.assertEqual(view["status"], "failed")
        self.assertEqual(view["agents"][3]["state"], "failed")

    def test_needs_clarification_shows_the_questions(self):
        ode = pinn_agent.ode(status="needs_clarification", questions_for_user=["When released?"])
        rec = Recorder()
        quiet(run_all.run_system, TEXT, run_all.make_fake_llm(ode), self.out, on_event=rec)
        view = ui_backend.build_run_view(rec.events, process_alive=False)
        self.assertEqual(view["questions"], ["When released?"])
        self.assertEqual(view["agents"][1]["state"], "completed")      # it answered, with questions

    def test_training_crash_is_reported(self):
        rec = Recorder()
        with mock.patch.object(code_agent, "run_spec", side_effect=RuntimeError("loss became nan")):
            result = quiet(run_all.run_system, TEXT, self.llm, self.out, on_event=rec)
        self.assertEqual(result["stopped_at"], "training failed: loss became nan")
        self.assertIn("loss became nan", next(e for e in rec.events
                                             if e["type"] == "attempt_failed")["data"]["error"])
        self.assertEqual(ui_backend.build_run_view(rec.events, False)["status"], "failed")


class TestEventsAndConfiguration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_event_writer_redacts_secrets_and_survives_partial_lines(self):
        path = os.path.join(self.dir, ui_backend.EVENTS_FILE)
        writer = ui_backend.EventWriter(path, secrets=["sk-test-SECRET-123456"])
        writer("run_error", message="bad key sk-test-SECRET-123456 rejected", value=np.float64(1.5))
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"seq": 2, "type": "half-writ')                 # a line still being written
        events = ui_backend.read_events(self.dir)
        self.assertEqual(len(events), 1)
        self.assertNotIn("SECRET", json.dumps(events))
        self.assertEqual(events[0]["data"]["value"], 1.5)

    def test_missing_keys_mean_no_backend_and_no_fallback(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
                                          "ANTHROPIC_API_KEY": ""}):
            self.assertEqual(ui_backend.available_backends(), [])
            self.assertIn("no API key", ui_backend.backend_status("gemini")["reason"])
        self.assertNotIn(ui_backend.TEST_BACKEND, ui_backend.BACKENDS)

    def test_backend_status_never_contains_the_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "real-key-value-123"}):
            self.assertNotIn("real-key-value-123", json.dumps(ui_backend.backend_status("gemini")))

    def test_load_dotenv_does_not_override(self):
        path = os.path.join(self.dir, ".env")
        with open(path, "w") as f:
            f.write("# comment\nUI_TEST_A=from_file\nUI_TEST_B='quoted'\n")
        with mock.patch.dict(os.environ, {"UI_TEST_A": "already"}):
            names = ui_backend.load_dotenv(path)
            self.assertEqual(os.environ["UI_TEST_A"], "already")
            self.assertEqual(os.environ["UI_TEST_B"], "quoted")
        self.assertEqual(names, ["UI_TEST_B"])

    def test_training_progress_reads_real_log_format(self):
        with open(os.path.join(self.dir, "training_log.txt"), "w") as f:
            f.write("Epoch      1 | total 1.0e+01 | physics 2.0e-02 | IC 1.0e+00\n"
                    "Epoch   1000 | total 3.0e-01 | physics 2.0e-01 | IC 1.0e-02 | data 4.0e-03 | c_est 0.7000\n")
        rows = ui_backend.training_progress(self.dir)
        self.assertEqual([r["epoch"] for r in rows], [1, 1000])
        self.assertEqual(rows[1]["c_est"], 0.7)
        self.assertEqual(ui_backend.training_progress(os.path.join(self.dir, "missing")), [])

    def test_run_status_values(self):
        start = {"seq": 1, "type": "run_started", "agent": None, "attempt": None, "elapsed_s": 0,
                 "time": "2026-10-21T10:00:00", "data": {"backend": "gemini"}}
        self.assertEqual(ui_backend.build_run_view([], None)["status"], "ready")
        # Regression: a just-launched worker has no events yet but IS running,
        # otherwise the page never starts refreshing (header stuck on "Ready").
        self.assertEqual(ui_backend.build_run_view([], True)["status"], "running")
        self.assertEqual(ui_backend.build_run_view([], False)["status"], "failed")
        self.assertEqual(ui_backend.build_run_view([start], True)["status"], "running")
        self.assertEqual(ui_backend.build_run_view([start], False)["status"], "failed")
        cancel = dict(start, seq=2, type="run_cancelled", data={})
        self.assertEqual(ui_backend.build_run_view([start, cancel], False)["status"], "cancelled")

    def test_example_prompts_pass_agent_1s_own_checker(self):
        """A correct agent-1 answer for each suggestion passes the pipeline's self-check."""
        prompts = ui_backend.example_prompts()
        linear = "m*x'' + c*x' + k*x = 0"
        t = np.linspace(0.0, 10.0, 21)[1:]
        x, _ = solve_reference_ode(t, 1.0, 1.0, 0.3, 0.0, 1.0, 0.0, 0.0)

        def p(m, c, k, **extra):
            out = {"m": {"value": m, "known": True}, "k": {"value": k, "known": True},
                   "c": {"value": c, "known": c is not None}}
            out.update({n: {"value": v, "known": True} for n, v in extra.items()})
            return out

        answers = {
            "Simple harmonic oscillator": pinn_agent.ode(
                parameters=p(1, 0, 1), expected_behavior={"damping_ratio": 0, "regime": "undamped"}),
            "Damped oscillator": pinn_agent.ode(
                parameters=p(1, 0.5, 4), expected_behavior={"damping_ratio": 0.125,
                                                            "regime": "underdamped"}),
            "Nonlinear Duffing oscillator": pinn_agent.ode(
                parameters=p(1, 0.1, 1, alpha=0.5), expected_behavior=pinn_agent.NO_EB,
                equation={"order": 2, "variable": "t", "unknown": "x",
                          "form": "m*x'' + c*x' + k*x + alpha*x^3 = 0"}),
            "Estimate unknown damping": pinn_agent.ode(
                problem_type="inverse", parameters=p(1, None, 1), expected_behavior=pinn_agent.NO_EB,
                observed_data={"t": [round(v, 4) for v in t], "x": [round(v, 4) for v in x]}),
        }
        self.assertEqual(set(prompts), set(answers))
        for name, ode in answers.items():
            ode["equation"].setdefault("form", linear)
            self.assertEqual(run_all.ode_self_check(prompts[name])(ode), [], name)
        self.assertNotIn("0.3 N", prompts["Estimate unknown damping"])   # hidden value not leaked

    def test_feedback_prompt_never_contains_local_file_paths(self):
        s = dict(summary(GOOD), data_file=r"C:\secret\path\solution_data.npz", plots_folder="x")
        prompt = feedback_agent.agent_input(pinn_agent.ode(), {"EPOCHS": 1}, s, 1)
        self.assertNotIn("solution_data", prompt)


class TestWorker(unittest.TestCase):
    def make_run(self, request):
        run_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, run_dir, True)
        with open(os.path.join(run_dir, ui_backend.REQUEST_FILE), "w") as f:
            json.dump(request, f)
        return run_dir

    def test_missing_api_key_gives_a_clear_error_not_a_fallback(self):
        run_dir = self.make_run({"text": TEXT, "backend": "gemini", "max_attempts": 1})
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}), \
                mock.patch.object(ui_backend, "load_dotenv", return_value=[]):
            self.assertEqual(quiet(ui_worker.run, run_dir), 1)
        view = ui_backend.load_run_view(run_dir)
        self.assertEqual(view["status"], "failed")
        self.assertIn("not configured", view["error"]["message"])
        self.assertEqual(view["agents"][1]["state"], "pending")         # nothing ran instead

    def test_test_backend_is_refused_without_the_test_flag(self):
        run_dir = self.make_run({"text": TEXT, "backend": ui_backend.TEST_BACKEND,
                                 "max_attempts": 1, "test_ode_json": pinn_agent.ode()})
        with mock.patch.dict(os.environ, {ui_backend.TEST_BACKEND_FLAG: ""}):
            self.assertEqual(quiet(ui_worker.run, run_dir), 1)
        self.assertIn("disabled", ui_backend.load_run_view(run_dir)["error"]["message"])

    def test_unexpected_exception_becomes_run_error_and_secret_is_scrubbed(self):
        run_dir = self.make_run({"text": TEXT, "backend": ui_backend.TEST_BACKEND, "max_attempts": 1,
                                 "test_ode_json": pinn_agent.ode()})
        secret = "AIza-fake-secret-key-987654321"
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": secret, ui_backend.TEST_BACKEND_FLAG: "1"}), \
                mock.patch.object(run_all, "run_system", side_effect=RuntimeError(f"boom {secret}")):
            self.assertEqual(quiet(ui_worker.run, run_dir), 1)
        with open(os.path.join(run_dir, ui_backend.EVENTS_FILE), encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn(secret, raw)
        self.assertIn("boom ***", raw)

    def real_run(self, ode_json, text, epochs):
        request = {"text": text, "backend": ui_backend.TEST_BACKEND, "max_attempts": 1,
                   "first_epochs": epochs, "test_ode_json": ode_json}
        runs = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, runs, True)
        with mock.patch.object(ui_backend, "RUNS_DIR", runs), \
                mock.patch.dict(os.environ, {ui_backend.TEST_BACKEND_FLAG: "1"}):
            run_dir, process = ui_backend.start_run(request)
        self.assertEqual(process.wait(timeout=900), 0,
                         ui_backend.read_text_tail(os.path.join(run_dir, ui_backend.CONSOLE_FILE)))
        with open(os.path.join(run_dir, ui_backend.REQUEST_FILE)) as f:
            self.assertNotIn("KEY", f.read())
        return ui_backend.load_run_view(run_dir, process_alive=False)

    def test_real_forward_run_shows_the_projects_own_plots(self):
        ode = pinn_agent.ode(parameters=pinn_agent.NONLINEAR, expected_behavior=pinn_agent.NO_EB,
                             equation={"order": 2, "variable": "t", "unknown": "x",
                                       "form": "m*x'' + c*x' + k*x + alpha*x^3 = 0"})
        text = ("Simulate a spring with mass 2 kg, stiffness 0.3, damping 0.3 and nonlinear "
                "coefficient 0.2, released from x=1 m at rest, from 0 to 10 s.")
        view = self.real_run(ode, text, 500)
        self.assertEqual(view["status"], "completed")
        attempt = view["attempts"][0]
        self.assertEqual(attempt["settings"]["EPOCHS"], 500)
        plots = ui_backend.attempt_plots(attempt)
        names = [os.path.basename(p) for _, p in plots]
        self.assertEqual(names, [n for n, _ in ui_backend.PLOT_FILES])     # displacement first
        folder = os.path.join(attempt["run_dir"], "physics_only")
        self.assertTrue(all(os.path.dirname(p) == folder for _, p in plots))  # main.py's own files
        # code_agent logs epoch 1 and every PRINT_EVERY=1000 epochs (unchanged from the CLI)
        rows = ui_backend.training_progress(attempt["run_dir"])
        self.assertEqual([r["epoch"] for r in rows], [1])
        self.assertEqual(attempt["settings"]["PRINT_EVERY"], 1000)

    def test_real_inverse_run_plots_are_drawn_with_plots_py(self):
        t = np.linspace(0, 10, 25)
        x, _ = solve_reference_ode(t, 2.0, 0.3, 0.3, 0.0, 1.0, 0.0, 0.0)
        ode = pinn_agent.ode(problem_type="inverse", parameters=pinn_agent.INVERSE,
                             expected_behavior=pinn_agent.NO_EB,
                             observed_data={"t": [round(v, 4) for v in t],
                                            "x": [round(v, 5) for v in x]})
        text = ("A spring with mass 2 kg and stiffness 0.3 has unknown damping, released from x=1 m "
                "at rest. Measured positions are attached. Simulate from 0 to 10 s.")
        view = self.real_run(ode, text, 300)
        attempt = view["attempts"][0]
        self.assertGreater(attempt["summary"]["estimated_c"], 0)
        plots = ui_backend.attempt_plots(attempt)
        self.assertEqual(len(plots), 4)
        self.assertTrue(all(os.path.getsize(p) > 10000 for _, p in plots))


if __name__ == "__main__":
    unittest.main()
