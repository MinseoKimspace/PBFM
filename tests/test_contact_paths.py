from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.contact_flow.dynamics import free_position, make_projection_condition
from src.contact_flow.io import load_states
from src.contact_flow.paths import build_path_buffer, compute_path_weights
from src.contact_flow.physics import FlowConfig, PhysicsConfig, geometry
from src.contact_flow.solver import SolverConfig


class PathBufferTests(unittest.TestCase):
    def setUp(self):
        self.state = torch.tensor([[[0.0, 1.0, 0.0, -0.2]],
                                   [[1.0, 1.2, 0.0, 0.1]]])
        self.radius = torch.full((2, 1), 0.5)
        self.physics = PhysicsConfig()
        self.solver = SolverConfig()
        self.dynamics = {"time_step": 1 / 60, "gravity_y": -9.8, "linear_damping": 0.1}
        self.paths = {"candidates": 3, "rank": 2, "start_noise_std": 0.02}

    @staticmethod
    def fake_integrate(field, z, radius, physics, solver, *, record, stop_on_tolerance):
        assert record and not stop_on_tolerance
        count = z.shape[0]
        tau = z.new_tensor([0.0, 0.25])[:, None].expand(2, count)
        velocity = torch.stack([field(z, tau[0]), field(z, tau[1])])
        return {"final": z, "time": torch.ones(count),
                "completed": torch.ones(count, dtype=torch.bool),
                "failed": torch.zeros(count, dtype=torch.bool),
                "converged": torch.ones(count, dtype=torch.bool),
                "nfe": 2, "backtracks": 0, "energy_evals": 3,
                "history": {"z": torch.stack([z, z]), "tau": tau, "u": velocity,
                            "dt": z.new_tensor([0.25, 0.75])[:, None].expand(2, count)}}

    def build(self, **changes):
        config = dict(self.paths, **changes)
        return build_path_buffer(self.state, self.radius, self.physics,
                                 self.dynamics, config, self.solver, seed=17)

    def test_weights_are_per_source_and_temperature_controls_concentration(self):
        scores = torch.tensor([[0.0, 1.0, 2.0], [1000.0, 1001.0, 1002.0]])
        cold = compute_path_weights(scores, 0.01)
        warm = compute_path_weights(scores, 100.0)
        torch.testing.assert_close(cold[0], cold[1])
        torch.testing.assert_close(cold.sum(dim=1), torch.ones(2, dtype=torch.float64))
        self.assertGreater(cold[0, 0].item(), 0.999)
        self.assertLess(warm[0].max().item(), 0.34)
        with self.assertRaises(ValueError):
            compute_path_weights(scores, 0)

    @patch("src.contact_flow.paths.integrate", side_effect=fake_integrate)
    def test_quadrature_preserves_path_weight_and_equal_anchor_mass(self, _):
        data = self.build()
        self.assertEqual(data["format"], "contact_paths_v2")
        self.assertEqual(data["tau"].shape, (12, 1))
        totals = torch.zeros(6, dtype=torch.float64)
        totals.scatter_add_(0, data["path_index"], data["weight"])
        torch.testing.assert_close(totals.reshape(2, 3), data["path_weights"])
        anchor_totals = torch.zeros(2, dtype=torch.float64)
        anchor_totals.scatter_add_(0, data["context_index"], data["weight"])
        torch.testing.assert_close(anchor_totals, torch.ones_like(anchor_totals))
        first = data["weight"][data["tau"].flatten() == 0]
        second = data["weight"][data["tau"].flatten() == 0.25]
        torch.testing.assert_close(second, 3 * first)
        self.assertTrue((data["dt"] > 0).all())

    @patch("src.contact_flow.paths.integrate", side_effect=fake_integrate)
    def test_fixed_condition_is_not_noisy_start_and_seed_is_reproducible(self, _):
        data = self.build()
        repeated = self.build()
        proposal = free_position(self.state, **self.dynamics)
        torch.testing.assert_close(data["condition"], make_projection_condition(self.state, proposal))
        self.assertFalse(torch.equal(data["start"], proposal))
        torch.testing.assert_close(data["z"][:6], data["start"].repeat_interleave(3, dim=0))
        for name in ("z", "velocity", "weight", "condition"):
            torch.testing.assert_close(data[name], repeated[name])
            self.assertEqual(data[name].device.type, "cpu")
            self.assertFalse(data[name].requires_grad)
        self.assertNotIn("target", data)

    def test_failed_paths_raise_instead_of_selectively_dropping_sources(self):
        def failed(*args, **kwargs):
            result = self.fake_integrate(*args, **kwargs)
            result["completed"][0] = False
            result["failed"][0] = True
            return result
        with patch("src.contact_flow.paths.integrate", side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, "No failed source was silently removed"):
                self.build()

    def test_incomplete_time_quadrature_is_rejected(self):
        def truncated(*args, **kwargs):
            result = self.fake_integrate(*args, **kwargs)
            result["history"]["dt"] = result["history"]["dt"] * 0.5
            return result
        with patch("src.contact_flow.paths.integrate", side_effect=truncated):
            with self.assertRaisesRegex(RuntimeError, "entire tau interval"):
                self.build()

    @patch("src.contact_flow.paths.integrate", side_effect=fake_integrate)
    def test_completed_but_unsolved_paths_are_retained(self, _):
        self.state[..., 1] = 0.3
        data = self.build(start_noise_std=0)
        self.assertTrue((data["diagnostics"]["terminal_energy"] > 0).all())
        self.assertEqual(data["path_scores"].shape, (2, 3))
        self.assertTrue(torch.isfinite(data["weight"]).all())

    def test_real_reference_integration_reduces_contact_energy(self):
        state = torch.tensor([[[0.0, 0.49, 0.0, -1.0]]])
        radius = torch.tensor([[0.5]])
        data = build_path_buffer(state, radius, self.physics, self.dynamics,
                                 {"candidates": 2, "rank": 1, "start_noise_std": 0},
                                 SolverConfig(steps=8, max_steps=128), seed=29)
        initial = geometry(data["start"], radius, self.physics)["energy"]
        self.assertTrue((data["diagnostics"]["terminal_energy"] <= initial[:, None]).all())
        self.assertTrue(torch.isfinite(data["velocity"]).all())
        torch.testing.assert_close(data["weight"].sum(), torch.tensor(1.0, dtype=torch.float64),
                                   atol=2e-5, rtol=2e-5)

    def test_float32_paths_cover_time_and_report_roundoff_policy(self):
        state, radius = load_states("", "train", 4, 42, 5)
        data = build_path_buffer(state, radius, self.physics, self.dynamics,
                                 {"candidates": 8, "rank": 4, "start_noise_std": 0.01},
                                 SolverConfig(), seed=42, flow=FlowConfig(normalization="none"))
        self.assertEqual(data["diagnostics"]["accepted_roundoff"].shape, (4, 8))
        self.assertTrue((data["diagnostics"]["accepted_roundoff"] >= 0).all())
        self.assertEqual(data["metadata"]["roundoff_policy"],
                         "within_tolerance_subresolution_nonincrease_only")
        totals = torch.zeros(32, dtype=torch.float64)
        totals.scatter_add_(0, data["path_index"], data["dt"])
        torch.testing.assert_close(totals, torch.ones_like(totals), atol=2e-5, rtol=2e-5)

    def test_uniform_weights_are_exact_and_ignore_score_ranking(self):
        scores = torch.tensor([[0.0, 1e6, 2.0], [-1000.0, -1.0, 2.0]])
        expected = torch.full((2, 3), 1 / 3, dtype=torch.float64)
        torch.testing.assert_close(compute_path_weights(scores, 1e-30, "uniform"), expected)
        with self.assertRaises(ValueError):
            compute_path_weights(scores, 1, "invalid")

    @patch("src.contact_flow.paths.integrate", side_effect=fake_integrate)
    def test_v2_stores_flow_and_final_states_and_rejects_old_cap(self, _):
        data = self.build(rank=0, weighting="uniform", terminal_violation_weight=3.0)
        self.assertEqual(data["final_positions"].shape, (2, 3, 1, 2))
        self.assertEqual(data["metadata"]["flow"], {
            "gain": 1.0, "normalization": "diagonal", "mobility_bound": 16.0,
            "normalization_eps": 0.0001})
        self.assertEqual(data["diagnostics"]["nfe"].shape, (2, 3))
        self.assertGreaterEqual(data["diagnostics"]["generation_seconds"], 0)
        with self.assertRaisesRegex(ValueError, "obsolete"):
            self.build(mobility_bound=16)

    @patch("src.contact_flow.paths.integrate", side_effect=fake_integrate)
    def test_terminal_violation_score_is_explicit_ablation(self, _):
        self.state[..., 1] = 0.3
        plain = self.build(start_noise_std=0)
        penalty = self.build(start_noise_std=0, terminal_violation_weight=4.0)
        torch.testing.assert_close(penalty["path_scores"] - plain["path_scores"],
                                   4 * plain["diagnostics"]["terminal_violation"].double())

    @patch("src.contact_flow.paths.integrate", side_effect=fake_integrate)
    def test_gain_is_applied_to_stored_tangents_not_only_inference(self, _):
        self.state[..., 1] = 0.3
        baseline = build_path_buffer(self.state, self.radius, self.physics,
                                     self.dynamics, self.paths, self.solver, 17,
                                     flow=FlowConfig(gain=1))
        faster = build_path_buffer(self.state, self.radius, self.physics,
                                   self.dynamics, self.paths, self.solver, 17,
                                   flow=FlowConfig(gain=4))
        self.assertGreater(baseline["velocity"].abs().sum(), 0)
        torch.testing.assert_close(faster["velocity"], 4 * baseline["velocity"])
        self.assertEqual(faster["metadata"]["flow"]["gain"], 4)


if __name__ == "__main__":
    unittest.main()
