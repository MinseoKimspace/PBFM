"""Analytic checks for the new preflight; no learned weights or external data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.contact_flow.io import load_config
from src.multiplier_flow.data import release_problem
from src.multiplier_flow.homotopy import Homotopy, euler, recovery, reference_path, solve_center
from src.multiplier_flow.homotopy_preflight import movement, preflight
from src.multiplier_flow.problem import pack


def scalar(c=-.1, **kwargs):
    return Homotopy.create(torch.ones(1, 1, dtype=torch.float64),
                          torch.tensor([c], dtype=torch.float64), **kwargs)


class HomotopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_identity_and_geometric_derivative(self):
        p = pack([release_problem()])
        for mode in ("epsilon", "diagonal_scaled", "isolated_root"):
            f = Homotopy.create(p["D"][0], p["c"][0], initialization=mode)
            torch.testing.assert_close(f.residual(f.start, 0), torch.zeros_like(f.c))
            value, derivative = f.mu(.4)
            fd = (f.mu(.400001)[0] - f.mu(.399999)[0]) / .000002
            self.assertAlmostEqual(derivative, fd, delta=1e-10)
            self.assertGreater(value, f.mu_min)
            self.assertAlmostEqual(f.mu(1)[0], f.mu_min)
            if mode == "diagonal_scaled":
                off = f.D - torch.diag(f.D.diag())
                torch.testing.assert_close(f.r0, f.c + off @ f.start)

    def test_scalar_center_and_inactive_limit(self):
        for c in (-.1, 0., .1):
            f = scalar(c, initialization="isolated_root")
            for t in (0., .5, 1.):
                result = solve_center(f, t, f.start)
                self.assertTrue(result["completed"], result)
                mu = f.mu(t)[0]
                effective_c = c - (1-t) * float(f.r0)
                exact = (-effective_c + (effective_c**2 + 4*mu)**.5) / 2
                self.assertAlmostEqual(float(result["final"]), exact, delta=1e-9)
        f = scalar(.1)
        end = solve_center(f, 1., f.start)["final"]
        self.assertGreater(float(end), 0.)  # finite mu_min is NOT exact KKT.
        self.assertAlmostEqual(float(end * (end+.1)), f.mu_min, delta=1e-10)

    def test_field_derivative_and_frozen_r0_recovery(self):
        p = pack([release_problem()])
        f = Homotopy.create(p["D"][0], p["c"][0])
        original = f.r0.clone()
        point = solve_center(f, .5, f.start)["final"]
        perturbed = point * torch.tensor([1.3, .8], dtype=point.dtype)
        A, rhs, H, H_t = f.system(perturbed, .5)
        u = torch.linalg.solve(A, rhs)
        torch.testing.assert_close(A @ u + H_t, -f.kappa * H)
        eps = 1e-7
        derivative = (f.residual(perturbed + eps*u, .5+eps) - H) / eps
        torch.testing.assert_close(derivative, -f.kappa*H, atol=1e-6, rtol=1e-4)
        result = recovery(f, point, steps=128)
        self.assertTrue(result["completed"])
        self.assertLess(result["relative_decay_error"], 1e-4)
        torch.testing.assert_close(f.r0, original)

    def test_same_clock_raw_and_explicit_guard_budget(self):
        f = scalar(initialization="epsilon")
        raw = euler(f, 4)
        self.assertEqual(raw["times"], [i/4 for i in range(len(raw["times"]))])
        self.assertLessEqual(raw["nfe"], 4)
        guarded = euler(f, 4, guarded=True, max_nfe_multiplier=2)
        self.assertLessEqual(guarded["nfe"], 8)
        self.assertTrue((guarded["states"] > 0).all())
        self.assertEqual(guarded["linear_solves"], guarded["nfe"])
        # A deliberately outward field exercises failure instead of clipping.
        f.field = lambda lam, t: -100 * torch.ones_like(lam)
        raw = euler(f, 4)
        self.assertFalse(raw["completed"])
        self.assertEqual(raw["reason"], "nonpositive_or_nonfinite_state")
        torch.testing.assert_close(raw["final"], f.start)
        guarded = euler(f, 4, guarded=True, max_nfe_multiplier=2)
        self.assertFalse(guarded["completed"])
        self.assertGreater(guarded["interventions"], 0)

    def test_uniform_mass_scaling(self):
        p = pack([release_problem()])
        f = Homotopy.create(p["D"][0], p["c"][0])
        heavy = Homotopy.create(p["D"][0] / 7, p["c"][0])
        torch.testing.assert_close(heavy.start, f.start * 7)
        torch.testing.assert_close(heavy.r0, f.r0)
        torch.testing.assert_close(heavy.field(heavy.start, .2), f.field(f.start, .2) * 7)

    def test_empty_and_invalid_settings_rejected(self):
        for kwargs in (dict(mu_min_ratio=0), dict(kappa=0), dict(schedule="arc_length"),
                       dict(initialization="centered"), dict(gap_scale=float("nan"))):
            with self.assertRaises(ValueError):
                scalar(**kwargs)
        with self.assertRaises(ValueError):
            scalar().field(torch.zeros(1, dtype=torch.float64), 0)

    def test_path_keeps_original_time_and_reference_cost(self):
        f = scalar(schedule="linear", initialization="diagonal_scaled")
        result = reference_path(f, points=9)
        self.assertTrue(result["completed"])
        self.assertEqual(result["times"], [i/8 for i in range(9)])
        self.assertGreater(result["linear_solves"], 0)
        stats = movement(result["times"], result["states"], f)
        self.assertAlmostEqual(sum(stats["movement_fraction"]), 1.)

    def test_small_report_is_strict_json_and_training_config_unchanged(self):
        config = load_config("configs/multiplier_homotopy_preflight.yaml")
        config["homotopy_preflight"].update(stack_sizes=[3], initializations=["diagonal_scaled"],
            schedules=["geometric"], calls=[4], reference_points=9, recovery_steps=32,
            guard_max_nfe_multiplier=2)
        original = copy.deepcopy(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            result = preflight(config, path)
            saved = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: self.fail(x))
        self.assertEqual(original, config)
        self.assertEqual(saved["format"], "multiplier_homotopy_preflight_v1")
        self.assertIn("stack_3", result["cases"])
        self.assertTrue(result["cases"]["isolated_separated"]["initially_solved"])
        self.assertEqual(next(iter(result["summary"].values()))["nontrivial_scenes"], 4)


if __name__ == "__main__":
    unittest.main()
