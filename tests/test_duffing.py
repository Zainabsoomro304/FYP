"""
Tests for the Duffing oscillator, the SciPy reference solver and the
unknown-damping experiment. Run from the project folder with:

    python -m unittest discover -s tests -v

Everything here is fast (smoke tests use a few dozen epochs). The full
damping-recovery test trains for 10,000 epochs (a few minutes) and only runs
when the environment variable RUN_SLOW_TESTS=1 is set.
"""

import contextlib
import csv
import io
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import damping_estimation  # noqa: E402
import main  # noqa: E402
import training  # noqa: E402
from physics import analytical_solution, analytical_velocity, solve_reference_ode  # noqa: E402
from pinn_model import PINN, TrainableDampingPINN, compute_physics_residual  # noqa: E402

QUIET = contextlib.redirect_stdout  # usage: with QUIET(io.StringIO()): ...


def square_model(**kwargs):
    """A PINN whose layers are replaced by s -> s^2, for exact residual checks."""
    class Square(torch.nn.Module):
        def forward(self, s):
            return s ** 2
    model = PINN(hidden_size=1, num_hidden_layers=1, **kwargs)
    model.net = Square()
    return model


# 1. ---------------------------------------------------------------------------
class TestLinearLimit(unittest.TestCase):
    """With NONLINEAR_COEFFICIENT = 0 everything reduces to the linear mass-spring ODE."""

    def test_residual_includes_cubic_term(self):
        # x = t^2, x' = 2t, x'' = 2  ->  r = 2m + 2ct + k t^2 + alpha t^6
        t = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
        r = compute_physics_residual(square_model(), t, mass=2.0, damping_coefficient=0.5,
                                     spring_constant=3.0, nonlinear_coefficient=0.25)
        expected = [4.0, 4.0 + 1.0 + 3.0 + 0.25, 4.0 + 2.0 + 12.0 + 0.25 * 64]
        np.testing.assert_allclose(r.detach().numpy().ravel(), expected, rtol=1e-6)

    def test_alpha_zero_residual_equals_linear_residual(self):
        t = torch.linspace(0, 3, 7).reshape(-1, 1).requires_grad_(True)
        model = PINN(8, 2)
        r_linear = compute_physics_residual(model, t, 1.5, 0.2, 2.0)
        r_alpha0 = compute_physics_residual(model, t, 1.5, 0.2, 2.0, nonlinear_coefficient=0.0)
        torch.testing.assert_close(r_alpha0, r_linear)

    def test_alpha_zero_reference_matches_analytical_solution(self):
        cases = [(1.0, 1.0, 0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.4, 1.0, 0.0, 0.0),
                 (1.0, 1.0, 2.0, 1.0, 0.0, 0.0), (1.0, 1.0, 5.0, 1.0, 0.0, 0.0),
                 (2.0, 8.0, 1.0, -0.5, 3.0, 1.5)]
        for m, k, c, x0, v0, t0 in cases:
            with self.subTest(c=c, t0=t0):
                t = np.linspace(t0, t0 + 10.0, 400)
                x, v = solve_reference_ode(t, m, k, c, 0.0, x0, v0, t0)
                np.testing.assert_allclose(x, analytical_solution(t, m, k, c, x0, v0, t0), atol=1e-8)
                np.testing.assert_allclose(v, analytical_velocity(t, m, k, c, x0, v0, t0), atol=1e-8)


# 2. and 3. --------------------------------------------------------------------
class TestReferenceSolver(unittest.TestCase):

    def test_initial_conditions_respected(self):
        for alpha, x0, v0, t0 in [(0.2, 1.0, 0.0, 0.0), (0.5, -0.3, 1.5, 2.0), (1.0, 0.0, 2.0, -1.0)]:
            with self.subTest(alpha=alpha, t0=t0):
                t = np.linspace(t0, t0 + 5.0, 50)
                x, v = solve_reference_ode(t, 1.0, 1.0, 0.3, alpha, x0, v0, t0)
                self.assertAlmostEqual(x[0], x0, places=12)
                self.assertAlmostEqual(v[0], v0, places=12)

    def test_duffing_solution_satisfies_the_ode(self):
        # Finite differences are fine HERE: this checks the reference, not the PINN.
        m, k, c, alpha = 1.0, 1.0, 0.3, 0.2
        t = np.linspace(0.0, 10.0, 20001)
        x, v = solve_reference_ode(t, m, k, c, alpha, 1.0, 0.0, 0.0)
        a = np.gradient(v, t[1] - t[0])
        residual = m * a + c * v + k * x + alpha * x ** 3
        self.assertLess(np.max(np.abs(residual[2:-2])), 1e-5)

    def test_solver_failure_is_reported(self):
        # A softening spring (alpha < 0) with a large amplitude blows up in finite time.
        t = np.linspace(0.0, 10.0, 100)
        with self.assertRaises(RuntimeError) as context:
            solve_reference_ode(t, 1.0, 1.0, 0.0, -5.0, 2.0, 0.0, 0.0)
        self.assertIn("failed", str(context.exception))

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):            # time before TIME_START
            solve_reference_ode([-1.0, 0.0, 1.0], 1, 1, 0, 0.2, 1, 0, 0.0)
        with self.assertRaises(ValueError):            # unsorted times
            solve_reference_ode([0.0, 2.0, 1.0], 1, 1, 0, 0.2, 1, 0, 0.0)
        with self.assertRaises(ValueError):            # invalid physics
            solve_reference_ode([0.0, 1.0], 0.0, 1, 0, 0.2, 1, 0, 0.0)
        with self.assertRaises(ValueError):
            solve_reference_ode([0.0, 1.0], 1, 1, 0, float("nan"), 1, 0, 0.0)


# 4. ---------------------------------------------------------------------------
class TestPhysicsOnlyMode(unittest.TestCase):

    def test_training_uses_no_reference_and_no_data_loss(self):
        boom = mock.Mock(side_effect=AssertionError("reference used during physics-only training!"))
        cfg = training.load_config({"EPOCHS": 5, "PRINT_EVERY": 100, "NONLINEAR_COEFFICIENT": 0.2,
                                    "ENABLE_TRAINING_VISUALIZATION": True, "VISUALIZATION_INTERVAL": 2})
        with mock.patch("physics.solve_reference_ode", boom), \
             mock.patch("physics.make_reference_solver", boom), \
             mock.patch.object(main, "make_reference_solver", boom), \
             mock.patch.object(damping_estimation, "make_reference_solver", boom), \
             mock.patch("pinn_model.compute_data_loss", boom), \
             mock.patch.object(training, "compute_data_loss", boom):
            training.set_seed(0)
            model = training.build_model(cfg)
            with QUIET(io.StringIO()):
                history, _, _ = training.train_pinn(model, cfg, observations=None)
        boom.assert_not_called()
        self.assertNotIn("data", history)
        self.assertEqual(set(history), {"total", "physics", "ic", "ic_position", "ic_velocity"})


# 5., 6. and 7. ------------------------------------------------------------------
class TestTrainableDamping(unittest.TestCase):

    def make(self, **overrides):
        settings = {"EXPERIMENT_MODE": "damping_estimation", "EPOCHS": 30, "PRINT_EVERY": 1000,
                    "ENABLE_TRAINING_VISUALIZATION": False, "N_COLLOCATION": 50}
        settings.update(overrides)
        return training.load_config(settings)

    def test_initial_estimate_is_the_configured_guess_not_the_true_value(self):
        cfg = self.make()
        known = damping_estimation.make_known_parameter_config(cfg)
        self.assertFalse(hasattr(known, "TRUE_DAMPING_COEFFICIENT"))
        self.assertFalse(hasattr(known, "DAMPING_COEFFICIENT"))
        model = training.build_model(known, learn_damping=True)
        self.assertAlmostEqual(model.damping_coefficient().item(), cfg.INITIAL_DAMPING_ESTIMATE, places=6)
        self.assertNotAlmostEqual(model.damping_coefficient().item(), cfg.TRUE_DAMPING_COEFFICIENT,
                                  places=3)

    def test_damping_receives_gradients_and_is_updated(self):
        cfg = self.make()
        t_obs, x_obs, _ = damping_estimation.generate_synthetic_observations(cfg)
        observations = damping_estimation.observations_to_tensors(t_obs, x_obs)
        known = damping_estimation.make_known_parameter_config(cfg)
        training.set_seed(0)
        model = training.build_model(known, learn_damping=True)
        self.assertIn(model.raw_damping, list(model.parameters()))     # in the optimizer
        with QUIET(io.StringIO()):
            history, _, _ = training.train_pinn(model, known, observations)
        self.assertIsNotNone(model.raw_damping.grad)
        self.assertNotEqual(model.raw_damping.grad.item(), 0.0)
        estimates = history["damping_estimate"]
        self.assertAlmostEqual(estimates[0], cfg.INITIAL_DAMPING_ESTIMATE, places=6)
        self.assertNotAlmostEqual(estimates[-1], estimates[0], places=4)

    def test_estimate_stays_nonnegative(self):
        model = TrainableDampingPINN(4, 1, initial_damping=0.8)
        for raw in [-100.0, -10.0, 0.0, 10.0]:
            with torch.no_grad():
                model.raw_damping.fill_(raw)
            self.assertGreaterEqual(model.damping_coefficient().item(), 0.0)
        # Training towards a true value of 0 (undamped data) must never go negative.
        cfg = self.make(TRUE_DAMPING_COEFFICIENT=0.0, INITIAL_DAMPING_ESTIMATE=0.05,
                        LEARNING_RATE=1e-2, EPOCHS=60)
        t_obs, x_obs, _ = damping_estimation.generate_synthetic_observations(cfg)
        known = damping_estimation.make_known_parameter_config(cfg)
        training.set_seed(0)
        model = training.build_model(known, learn_damping=True)
        with QUIET(io.StringIO()):
            history, _, _ = training.train_pinn(
                model, known, damping_estimation.observations_to_tensors(t_obs, x_obs))
        self.assertTrue(all(c >= 0 for c in history["damping_estimate"]))

    def test_invalid_initial_estimate_rejected(self):
        for bad in [0.0, -0.5, float("nan")]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TrainableDampingPINN(4, 1, initial_damping=bad)


# 8. and 10. ---------------------------------------------------------------------
class TestExperimentsEndToEnd(unittest.TestCase):

    def test_both_modes_run_and_write_outputs(self):
        with tempfile.TemporaryDirectory() as out:
            common = {"EPOCHS": 30, "PRINT_EVERY": 1000, "VISUALIZATION_INTERVAL": 10,
                      "VISUALIZATION_POINTS": 50, "N_COLLOCATION": 50, "OUTPUT_DIR": out}
            with QUIET(io.StringIO()):
                metrics_a = main.main(dict(common, EXPERIMENT_MODE="physics_only"))
                metrics_b = main.main(dict(common, EXPERIMENT_MODE="damping_estimation"))

            dir_a, dir_b = os.path.join(out, "physics_only"), os.path.join(out, "damping_estimation")
            for name in ["duffing_solution_comparison.png", "duffing_absolute_error.png",
                         "duffing_training_losses.png", "physics_residual.png", "metrics.json",
                         "training_snapshots.png", "solution_evolution.gif"]:
                self.assertGreater(os.path.getsize(os.path.join(dir_a, name)), 0, name)
            for name in ["duffing_solution_comparison.png", "duffing_absolute_error.png",
                         "duffing_training_losses.png", "damping_estimation_history.png",
                         "damping_observations.png", "observations.csv", "metrics.json"]:
                self.assertGreater(os.path.getsize(os.path.join(dir_b, name)), 0, name)

            # Mode A: no damping estimation, no data loss.
            self.assertNotIn("estimated_damping_coefficient_N_s_per_m", metrics_a)
            self.assertNotIn("final_model_loss_data_m2", metrics_a)

            # Mode B: evaluated against the HIDDEN true value and the reference.
            cfg = training.load_config(dict(common, EXPERIMENT_MODE="damping_estimation"))
            self.assertEqual(metrics_b["true_damping_coefficient_N_s_per_m"], cfg.TRUE_DAMPING_COEFFICIENT)
            error = abs(metrics_b["estimated_damping_coefficient_N_s_per_m"] - cfg.TRUE_DAMPING_COEFFICIENT)
            self.assertAlmostEqual(metrics_b["damping_absolute_error_N_s_per_m"], error, places=12)
            self.assertAlmostEqual(metrics_b["damping_relative_error"],
                                   error / cfg.TRUE_DAMPING_COEFFICIENT, places=12)
            self.assertAlmostEqual(metrics_b["rmse_position_m"] ** 2, metrics_b["mse_position_m2"], places=12)
            for key in ["data_mse_at_observations_m2", "physics_residual_mse_eval_grid_N2",
                        "initial_position_error_m", "initial_velocity_error_m_per_s"]:
                self.assertIn(key, metrics_b)

            with open(os.path.join(dir_b, "observations.csv"), encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["time_s", "observed_position_m", "noise_free_position_m"])
            self.assertEqual(len(rows) - 1, cfg.N_OBSERVATION_POINTS)
            with open(os.path.join(dir_b, "metrics.json"), encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["experiment_mode"], "damping_estimation")

    def test_estimation_training_never_sees_true_damping(self):
        boom = mock.Mock(side_effect=AssertionError("reference used during estimation training!"))
        cfg = training.load_config({"EXPERIMENT_MODE": "damping_estimation", "EPOCHS": 5,
                                    "PRINT_EVERY": 1000, "ENABLE_TRAINING_VISUALIZATION": False})
        t_obs, x_obs, _ = damping_estimation.generate_synthetic_observations(cfg)
        known = damping_estimation.make_known_parameter_config(cfg)
        training.set_seed(0)
        model = training.build_model(known, learn_damping=True)
        with mock.patch("physics.solve_reference_ode", boom), \
             mock.patch.object(damping_estimation, "make_reference_solver", boom):
            with QUIET(io.StringIO()):
                training.train_pinn(model, known,
                                    damping_estimation.observations_to_tensors(t_obs, x_obs))
        boom.assert_not_called()


# 9. ---------------------------------------------------------------------------
class TestObservationGeneration(unittest.TestCase):

    def test_noise_free_and_noisy_data_are_reproducible(self):
        clean_cfg = training.load_config({"ADD_OBSERVATION_NOISE": False})
        noisy_cfg = training.load_config({"ADD_OBSERVATION_NOISE": True, "OBSERVATION_NOISE_STD": 0.01})
        t1, x1, clean1 = damping_estimation.generate_synthetic_observations(clean_cfg)
        t2, x2, clean2 = damping_estimation.generate_synthetic_observations(clean_cfg)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(x1, clean1)              # no noise -> exact reference values

        tn1, xn1, cleann1 = damping_estimation.generate_synthetic_observations(noisy_cfg)
        tn2, xn2, _ = damping_estimation.generate_synthetic_observations(noisy_cfg)
        np.testing.assert_array_equal(xn1, xn2)                # same seed -> same noise
        np.testing.assert_array_equal(cleann1, clean1)
        noise = xn1 - cleann1
        self.assertGreater(np.std(noise), 0.005)
        self.assertLess(np.std(noise), 0.015)

        other_seed = training.load_config({"ADD_OBSERVATION_NOISE": True, "SEED": 7})
        _, xn3, _ = damping_estimation.generate_synthetic_observations(other_seed)
        self.assertFalse(np.allclose(xn3, xn1))                # different seed -> different noise

    def test_observation_times_cover_interval_and_differ_from_evaluation_grid(self):
        cfg = training.load_config({"N_OBSERVATION_POINTS": 50})
        t_obs, _, _ = damping_estimation.generate_synthetic_observations(cfg)
        self.assertEqual(len(t_obs), 50)
        self.assertAlmostEqual(t_obs[0], cfg.TIME_START)
        self.assertAlmostEqual(t_obs[-1], cfg.TIME_END)
        self.assertLess(len(t_obs), cfg.N_EVALUATION_POINTS)


# Optional full training ---------------------------------------------------------
@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS") == "1", "set RUN_SLOW_TESTS=1 to run")
class TestDampingRecoveryFullTraining(unittest.TestCase):

    def test_recovers_moderate_damping_from_clean_data(self):
        with tempfile.TemporaryDirectory() as out:
            with QUIET(io.StringIO()):
                metrics = main.main({"EXPERIMENT_MODE": "damping_estimation", "EPOCHS": 10000,
                                     "ENABLE_TRAINING_VISUALIZATION": False, "OUTPUT_DIR": out})
        self.assertLess(metrics["damping_relative_error"], 0.05)
        self.assertLess(metrics["rmse_position_m"], 0.01)


if __name__ == "__main__":
    unittest.main()
