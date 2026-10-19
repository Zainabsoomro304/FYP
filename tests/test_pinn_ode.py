"""
Lightweight unit tests. Run from the project folder with:

    python -m unittest discover -s tests -v

They check the analytical solution, parameter validation, the initial-condition
loss, and that training never touches the analytical solution. A full PINN
training run is NOT part of these tests (it takes about a minute); see README.
"""

import contextlib
import copy
import io
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from physics import (analytical_solution, analytical_velocity,  # noqa: E402
                     classify_damping_regime, validate_physical_parameters)
from pinn_model import PINN, compute_initial_condition_loss, compute_physics_residual  # noqa: E402

# (mass, spring, damping, x0, v0, t0) -- one case per regime plus extra ICs / start times
CASES = {
    "undamped":            (1.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    "undamped_v0_t0":      (2.0, 8.0, 0.0, -0.5, 3.0, 1.5),
    "underdamped":         (1.0, 1.0, 0.4, 1.0, 0.0, 0.0),
    "underdamped_v0_t0":   (0.5, 4.0, 1.0, 0.2, -1.0, 2.0),
    "critically_damped":   (1.0, 1.0, 2.0, 1.0, 0.0, 0.0),
    "critical_v0_t0":      (2.0, 3.0, 2 * math.sqrt(6.0), 0.0, 1.0, -1.0),
    "overdamped":          (1.0, 1.0, 5.0, 1.0, 0.0, 0.0),
    "overdamped_v0_t0":    (3.0, 2.0, 10.0, 1.0, 2.0, 4.0),
}


class TestAnalyticalSolution(unittest.TestCase):

    def test_original_configuration_is_cos(self):
        t = np.linspace(0.0, 2 * math.pi, 1001)
        x = analytical_solution(t, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        np.testing.assert_allclose(x, np.cos(t), atol=1e-14)

    def test_regime_classification(self):
        self.assertEqual(classify_damping_regime(1, 1, 0.0), "undamped")
        self.assertEqual(classify_damping_regime(1, 1, 1.0), "underdamped")
        self.assertEqual(classify_damping_regime(1, 1, 2.0), "critically damped")
        self.assertEqual(classify_damping_regime(2, 3, 2 * math.sqrt(6.0)), "critically damped")
        self.assertEqual(classify_damping_regime(1, 1, 3.0), "overdamped")

    def test_initial_conditions_hold_at_time_start(self):
        for name, (m, k, c, x0, v0, t0) in CASES.items():
            with self.subTest(name):
                x = analytical_solution(np.array([t0]), m, k, c, x0, v0, t0)
                v = analytical_velocity(np.array([t0]), m, k, c, x0, v0, t0)
                self.assertAlmostEqual(float(x[0]), x0, places=12)
                self.assertAlmostEqual(float(v[0]), v0, places=12)

    def test_solution_satisfies_the_ode(self):
        # Finite differences are fine HERE: this only checks the reference solution.
        for name, (m, k, c, x0, v0, t0) in CASES.items():
            with self.subTest(name):
                t = np.linspace(t0, t0 + 10.0, 20001)
                h = t[1] - t[0]
                x = analytical_solution(t, m, k, c, x0, v0, t0)
                v = analytical_velocity(t, m, k, c, x0, v0, t0)
                np.testing.assert_allclose(np.gradient(x, h)[2:-2], v[2:-2], atol=1e-5)
                a = np.gradient(v, h)
                residual = m * a + c * v + k * x
                self.assertLess(np.max(np.abs(residual[2:-2])), 1e-4)

    def test_damped_solutions_decay(self):
        for name in ["underdamped", "critically_damped", "overdamped"]:
            m, k, c, x0, v0, t0 = CASES[name]
            x = analytical_solution(np.array([t0 + 50.0]), m, k, c, x0, v0, t0)
            self.assertLess(abs(float(x[0])), 1e-3, name)


class TestValidation(unittest.TestCase):

    def test_invalid_parameters_raise(self):
        for m, k, c in [(0, 1, 0), (-1, 1, 0), (1, 0, 0), (1, -2, 0), (1, 1, -0.1),
                        (float("nan"), 1, 0), (1, float("inf"), 0)]:
            with self.subTest(m=m, k=k, c=c), self.assertRaises(ValueError):
                validate_physical_parameters(m, k, c)

    def test_valid_parameters_pass(self):
        validate_physical_parameters(1.0, 1.0, 0.0)
        validate_physical_parameters(2.5, 40.0, 3.0)

    def test_bad_config_is_rejected(self):
        for overrides in [{"MASS": 0.0}, {"DAMPING_COEFFICIENT": -1.0},
                          {"TIME_END": 0.0}, {"N_COLLOCATION": 1}, {"NOT_A_SETTING": 1}]:
            with self.subTest(overrides), self.assertRaises((ValueError, KeyError)):
                main.load_config(overrides)


class TestPINNLosses(unittest.TestCase):

    @staticmethod
    def square_model(**kwargs):
        """A PINN whose layers are replaced by s -> s^2 (s = network input), for exact checks."""
        class Square(torch.nn.Module):
            def forward(self, s):
                return s ** 2
        model = PINN(hidden_size=1, num_hidden_layers=1, **kwargs)
        model.net = Square()
        return model

    def test_initial_condition_loss_uses_time_start(self):
        # x = t^2  ->  x(t0=2) = 4,  x'(t0=2) = 4
        model = self.square_model()
        loss, pos_err, vel_err = compute_initial_condition_loss(model, time_start=2.0,
                                                                initial_position=0.5,
                                                                initial_velocity=1.0)
        self.assertAlmostEqual(pos_err.item(), 4.0 - 0.5, places=5)
        self.assertAlmostEqual(vel_err.item(), 4.0 - 1.0, places=5)
        self.assertAlmostEqual(loss.item(), 3.5 ** 2 + 3.0 ** 2, places=4)

    def test_physics_residual_formula(self):
        # x = t^2, x' = 2t, x'' = 2  ->  r = 2m + 2ct + kt^2
        model = self.square_model()
        t = torch.tensor([[0.0], [1.0], [3.0]], requires_grad=True)
        r = compute_physics_residual(model, t, mass=2.0, damping_coefficient=0.5, spring_constant=3.0)
        np.testing.assert_allclose(r.detach().numpy().ravel(), [4.0, 8.0, 34.0], rtol=1e-6)

    def test_time_normalization_keeps_physical_derivative(self):
        # Scaling [10, 14] -> [-1, 1] gives s = (t - 12)/2 and x = s^2.
        # At t = 13: x = 0.25, dx/dt = 2s * ds/dt = 0.5, d2x/dt2 = 2 * (1/2)^2 = 0.5
        model = self.square_model(t_min=10.0, t_max=14.0)
        t = torch.tensor([[13.0]], requires_grad=True)
        x, v, a = main.predict_with_derivatives(model, t)
        self.assertAlmostEqual(x.item(), 0.25, places=6)
        self.assertAlmostEqual(v.item(), 0.5, places=6)
        self.assertAlmostEqual(a.item(), 0.5, places=6)

    def test_training_never_uses_analytical_solution(self):
        """If training touched the analytical solution, these mocks would raise."""
        boom = mock.Mock(side_effect=AssertionError("analytical solution used in training!"))
        with tempfile.TemporaryDirectory() as out:
            cfg = main.load_config({"EPOCHS": 5, "PRINT_EVERY": 5, "DAMPING_COEFFICIENT": 0.5,
                                    "ENABLE_TRAINING_VISUALIZATION": True, "VISUALIZATION_INTERVAL": 2,
                                    "TIME_START": 1.0, "TIME_END": 5.0, "OUTPUT_DIR": out})
            with mock.patch.object(main, "analytical_solution", boom), \
                 mock.patch("physics.analytical_solution", boom), \
                 mock.patch("physics.analytical_velocity", boom):
                main.set_seed(0)
                model = main.build_model(cfg)
                history, best_epoch, snapshots = main.train_pinn(model, cfg)
            self.assertEqual(len(history["total"]), 5)
            self.assertTrue(all(math.isfinite(v) for v in history["total"]))
            boom.assert_not_called()



class TestTrainingVisualization(unittest.TestCase):
    """Snapshots must show the REAL network at each epoch and must not change training."""

    BASE = {"EPOCHS": 20, "PRINT_EVERY": 1000, "VISUALIZATION_INTERVAL": 5,
            "VISUALIZATION_POINTS": 50, "N_COLLOCATION": 50, "DAMPING_COEFFICIENT": 0.3}

    def run_training(self, **overrides):
        cfg = main.load_config(dict(self.BASE, **overrides))
        main.set_seed(cfg.SEED)
        model = main.build_model(cfg)
        untrained = copy.deepcopy(model)
        with contextlib.redirect_stdout(io.StringIO()):
            history, best_epoch, snapshots = main.train_pinn(model, cfg)
        return cfg, model, untrained, history, best_epoch, snapshots

    def test_snapshot_epochs_are_adapted_to_training_length(self):
        from visualization import select_snapshot_epochs
        self.assertEqual(sorted(select_snapshot_epochs(100, 50)), [0, 1, 10, 50, 100])
        self.assertEqual(max(select_snapshot_epochs(1234, 100)), 1234)

    def test_epoch_0_is_the_untrained_network(self):
        cfg, _, untrained, _, _, snapshots = self.run_training(ENABLE_TRAINING_VISUALIZATION=True)
        first = snapshots[0]
        self.assertEqual((first["epoch"], first["kind"]), (0, "training"))
        t_visual = main.make_visualization_points(cfg)
        with torch.no_grad():
            expected = untrained(t_visual).numpy().ravel()
        np.testing.assert_array_equal(first["x_pred"], expected)

    def test_snapshots_match_independent_shorter_runs(self):
        """Snapshot at epoch N == final weights of a separate run stopped after N epochs
        (with visualization disabled). This proves the snapshots show the real weights
        and that capturing them does not alter training."""
        _, _, _, history_vis, _, snapshots = self.run_training(ENABLE_TRAINING_VISUALIZATION=True)
        by_epoch = {s["epoch"]: s for s in snapshots if s["kind"] == "training"}
        for n in [5, 10]:
            cfg, model_n, _, history_n, _, snaps_n = self.run_training(
                EPOCHS=n, ENABLE_TRAINING_VISUALIZATION=False, KEEP_BEST_WEIGHTS=False)
            self.assertEqual(snaps_n, [])
            with torch.no_grad():
                expected = model_n(main.make_visualization_points(cfg)).numpy().ravel()
            np.testing.assert_allclose(by_epoch[n]["x_pred"], expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(history_vis["total"][:n], history_n["total"], rtol=1e-6)

    def test_final_and_best_are_kept_separate(self):
        cfg, model, _, history, best_epoch, snapshots = self.run_training(
            ENABLE_TRAINING_VISUALIZATION=True)
        kinds = [s["kind"] for s in snapshots]
        self.assertEqual(kinds.count("final"), 1)
        self.assertEqual(kinds.count("best"), 1)
        best = next(s for s in snapshots if s["kind"] == "best")
        self.assertEqual(best["epoch"], best_epoch - 1)
        self.assertEqual(best["loss_total"], history["total"][best_epoch - 1])
        with torch.no_grad():   # model now holds the restored best weights
            expected = model(main.make_visualization_points(cfg)).numpy().ravel()
        np.testing.assert_array_equal(best["x_pred"], expected)

    def test_output_files_are_created(self):
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                main.main(dict(self.BASE, ENABLE_TRAINING_VISUALIZATION=True, OUTPUT_DIR=out))
            for name in ["solution_comparison.png", "training_losses.png", "absolute_error.png",
                         "physics_residual.png", "metrics.json", "training_snapshots.png",
                         "solution_evolution.gif"]:
                self.assertTrue(os.path.getsize(os.path.join(out, name)) > 0, name)

    def test_disabled_visualization_creates_no_extra_files(self):
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                main.main(dict(self.BASE, ENABLE_TRAINING_VISUALIZATION=False, OUTPUT_DIR=out))
            self.assertTrue(os.path.exists(os.path.join(out, "metrics.json")))
            self.assertFalse(os.path.exists(os.path.join(out, "training_snapshots.png")))
            self.assertFalse(os.path.exists(os.path.join(out, "solution_evolution.gif")))


if __name__ == "__main__":
    unittest.main()
