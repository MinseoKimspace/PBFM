"""Large-scene geometry parity, unchanged PGS equations and honest comparisons."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

import eval_multiplier_stress
from test_multiplier_flow import EndpointFunction, tiny_config
from src.contact_flow.dynamics import free_position
from src.contact_flow.physics import PhysicsConfig
from src.multiplier_flow.model import ConditionalField
from src.multiplier_flow.problem import circle_problem, pack, move
from src.multiplier_flow.solvers import pgs, run_cfm
from src.multiplier_flow.stress import (benchmark_snapshot, evaluate_stress, prepare_problem,
                                        simulate_stress, solve, validate_stress)
from src.multiplier_flow.stress_pgs import SparsePGS
from src.multiplier_flow.stress_problem import circle_gaps, compact_problem, large_scene


def config():
    cfg = tiny_config()
    cfg["model"].update(communication="global", feature_version="residual", head_type="analytic",
                        attention_heads=2, attention_layers=1)
    cfg["stress"] = dict(mode="snapshot", pgs_backend="sparse_cpu", sizes=[8],
        scenes=["pile_drop"], seeds=[42], hybrid_calls=[2], tolerance=.001,
        max_sweeps=2000, timing_repeats=2, steps=3, radius=.45, spacing_gap=.002,
        max_speed=100., max_position=100., render=False, image_size=160, frame_stride=1)
    return cfg


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_compact_qp_matches_original_geometry_and_exact_components(self):
        physics = PhysicsConfig()
        positions = torch.tensor([[0., .44], [0., 1.34], [4., .44], [9.56, 3.]], dtype=torch.float64)
        radius = torch.full((4,), .45, dtype=torch.float64)
        compact, _, info = compact_problem(positions, radius, physics, .9)
        original = pack([circle_problem(positions, radius, physics, .9)])
        for key in ("p", "w", "c", "J", "D", "mask", "reach"):
            torch.testing.assert_close(compact[key], original[key], atol=0, rtol=0)
        self.assertLessEqual(float(compact["eta"]), float(original["eta"])+1e-14)
        self.assertGreater(info["components"], 1)
        self.assertLess(info["contacts"], 4*3//2+4*3)
        # Empty contact world keeps one masked row, without inventing a component.
        empty, _, meta = compact_problem(torch.tensor([[0., 5.]], dtype=torch.float64), radius[:1], physics, .9)
        self.assertFalse(empty["mask"].any())
        self.assertFalse(empty["reach"].any())
        self.assertEqual(meta["components"], 0)
        self.assertTrue(SparsePGS(empty).solve(torch.zeros_like(empty["c"]), .001, 10)["converged"].all())

    def test_scenes_have_requested_counts_no_initial_overlap_and_reproducible_seeds(self):
        physics = PhysicsConfig()
        for count in (256, 512, 1024, 2048):
            for kind in ("pile_drop", "pile_impact", "pile_shear"):
                state, radius = large_scene(kind, count, 42, physics)
                self.assertEqual(state.shape, (count, 4))
                gaps = circle_gaps(state[:, :2], radius, physics)
                self.assertGreater(float(gaps["gap"].min()), 0)
                other, _ = large_scene(kind, count, 42, physics)
                torch.testing.assert_close(state, other, atol=0, rtol=0)
                different, _ = large_scene(kind, count, 43, physics)
                self.assertFalse(torch.equal(state[:, 2:], different[:, 2:]))

    def test_sparse_pgs_matches_sequential_dense_updates_and_sweep_cap(self):
        cfg = config()
        physics = PhysicsConfig()
        state, radius = large_scene("pile_drop", 12, 42, physics)
        proposal = free_position(state.double()[None], **cfg["dynamics"])[0]
        problem, _, _ = compact_problem(proposal, radius.double(), physics, .9)
        for dtype, tolerance, atol in ((torch.float64, 1e-9, 1e-12), (torch.float32, 1e-4, 1e-6)):
            qp = move(problem, torch.device("cpu"), dtype)
            backend = SparsePGS(qp)
            for cap in (1, 3, 2000):
                start = torch.rand_like(qp["c"])*.01
                expected = pgs(qp, start, tolerance, cap)
                actual = backend.solve(start, tolerance, cap)
                torch.testing.assert_close(actual["final"], expected["final"], atol=atol, rtol=atol*100)
                self.assertEqual(actual["sweeps"].tolist(), expected["sweeps"].tolist())
                self.assertEqual(actual["converged"].tolist(), expected["converged"].tolist())
                self.assertEqual(actual["contact_evals"].tolist(), expected["contact_evals"].tolist())

    def test_hybrid_passes_exact_fm_endpoint_to_same_backend_and_failed_prefix_stops(self):
        cfg = config()
        state, radius = large_scene("pile_drop", 8, 42, PhysicsConfig())
        proposal = free_position(state[None], **cfg["dynamics"])[0]
        problem, _, _, backend = prepare_problem(proposal, radius, PhysicsConfig(), .9, cfg["stress"])
        model = ConditionalField(**cfg["model"]).eval()
        prefix = run_cfm(model, problem, torch.zeros_like(problem["c"]), 2, collect_diagnostics=False)
        with patch.object(backend, "solve", wraps=backend.solve) as finish:
            result = solve(problem, model, 2, cfg["stress"], backend)
        torch.testing.assert_close(finish.call_args.args[0], prefix["final"], atol=0, rtol=0)
        self.assertTrue(result["converged"].all())
        broken = EndpointFunction(lambda x, tau, qp: torch.full_like(x, float("nan")))
        with patch.object(backend, "solve", wraps=backend.solve) as finish:
            failed = solve(problem, broken, 2, cfg["stress"], backend)
        finish.assert_not_called()
        self.assertFalse(failed["completed"].any())
        self.assertFalse(failed["converged"].any())

    def test_snapshot_reporting_includes_transfer_backend_and_capped_failures(self):
        cfg = config()
        model = ConditionalField(**cfg["model"]).eval()
        state, radius = large_scene("pile_drop", 8, 42, PhysicsConfig())
        with patch("src.multiplier_flow.stress.timed", side_effect=lambda device, op: (op(), .02)):
            report, predictions = benchmark_snapshot(model, state, radius, cfg, torch.device("cpu"), 42)
        self.assertEqual(set(report["methods"]), {"pgs", "hybrid_k2"})
        for row in report["methods"].values():
            self.assertEqual(row["timing_seconds"], [.02, .02])
            self.assertEqual(row["solver_seconds"], .02)
            self.assertEqual(row["setup_plus_solver_seconds"], .04)
            self.assertEqual(row["speedup_vs_pgs"], 1.)
            self.assertTrue(row["solver_converged"])
        self.assertEqual(predictions["pgs"].shape, (8, 2))
        cfg["stress"]["max_sweeps"] = 1
        failed, _ = benchmark_snapshot(model, state, radius, cfg, torch.device("cpu"), 42)
        self.assertFalse(failed["methods"]["pgs"]["solver_converged"])
        self.assertIsNone(failed["methods"]["hybrid_k2"]["speedup_vs_pgs"])

    def test_rollout_failure_never_advances_and_success_preserves_frame_metrics(self):
        cfg = config()
        state, radius = large_scene("pile_drop", 8, 42, PhysicsConfig())
        model = ConditionalField(**cfg["model"]).eval()
        result = simulate_stress(model, state, radius, cfg, torch.device("cpu"), 2)
        self.assertEqual(result["completed_steps"], 3)
        self.assertTrue(result["summary"]["completed"])
        self.assertTrue(all(row["solver_converged"] for row in result["frames"]))
        cfg["stress"]["max_sweeps"] = 1
        failed = simulate_stress(model, state, radius, cfg, torch.device("cpu"), 0)
        self.assertEqual(failed["completed_steps"], 0)
        self.assertEqual(len(failed["states"]), 1)
        self.assertEqual(failed["summary"]["attempted_steps"], 1)
        self.assertGreater(failed["summary"]["total_attempt_seconds"], 0)

    def test_complete_evaluation_json_markdown_and_two_panel_gif(self):
        cfg = config()
        cfg["stress"].update(mode="rollout", render=True)
        model = ConditionalField(**cfg["model"]).eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"report"
            report = evaluate_stress(model, cfg, torch.device("cpu"), output)
            saved = json.loads((output/"stress.json").read_text())
            self.assertEqual(saved["pgs_backend"], "sparse_cpu")
            self.assertEqual(set(report["cases"][0]["methods"]), {"pgs", "hybrid_k2"})
            self.assertTrue((output/"stress.md").exists())
            with Image.open(next(output.glob("*.gif"))) as image:
                self.assertEqual(image.width, 320)
            with self.assertRaises(FileExistsError):
                evaluate_stress(model, cfg, torch.device("cpu"), output)

    def test_invalid_settings_are_rejected(self):
        for key, value in (("sizes", [0]), ("hybrid_calls", [2, 2]), ("tolerance", 0),
                           ("mode", "unknown"), ("scenes", []), ("pgs_backend", "unknown")):
            cfg = config()
            cfg["stress"][key] = value
            with self.assertRaises(ValueError):
                validate_stress(cfg)

    def test_cli_keeps_training_spec_while_overriding_large_evaluation(self):
        cfg = config()
        original = copy.deepcopy(cfg)
        model = ConditionalField(**cfg["model"])
        checkpoint = dict(updates=12, selection_metric="validation", config=cfg)
        with tempfile.TemporaryDirectory() as directory, \
                patch("sys.argv", ["eval_multiplier_stress.py", "--output", directory,
                                    "--sizes", "256", "2048", "--tolerance", "0.0001", "--device", "cpu"]), \
                patch.object(eval_multiplier_stress, "load_config", return_value=cfg), \
                patch.object(eval_multiplier_stress, "load_model", return_value=(model, checkpoint)) as loader, \
                patch.object(eval_multiplier_stress, "evaluate_stress") as evaluate:
            eval_multiplier_stress.main()
        selected = loader.call_args.args[1]
        self.assertEqual(cfg, original)
        for key in ("seed", "data", "physics", "dynamics", "reference"):
            self.assertEqual(selected[key], original[key])
        self.assertEqual(selected["stress"]["sizes"], [256, 2048])
        self.assertEqual(selected["stress"]["tolerance"], .0001)
        self.assertEqual(evaluate.call_args.kwargs["metadata"]["updates"], 12)


if __name__ == "__main__":
    unittest.main()
