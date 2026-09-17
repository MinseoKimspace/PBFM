import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import eval_multiplier_rollout
from test_multiplier_flow import EndpointFunction, tiny_config
from src.contact_flow.dynamics import free_position, finite_difference_state
from src.contact_flow.physics import PhysicsConfig
from src.multiplier_flow.model import ConditionalField
from src.multiplier_flow.problem import converged, decode
from src.multiplier_flow.rollout import (build_frame_problem, evaluate_motion, motion_scenes,
                                         render_motion, rollout_budgets, simulate, swept_pairs)
from src.multiplier_flow.solvers import pgs, run_cfm


def config():
    cfg = tiny_config()
    cfg["rollout"] = dict(steps=3, calls=[1], guarded=False, max_backtracks=2,
        frame_stride=1, image_size=160, render=False, max_speed=100., max_position=100.)
    return cfg


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_free_flight_matches_analytic_motion_for_both_solvers(self):
        cfg = config()
        _, initial, radius = motion_scenes(PhysicsConfig())[0]
        net = ConditionalField(**cfg["model"]).eval()
        for model in (None, net):
            run = simulate(initial, radius, cfg, torch.device("cpu"), model)
            expected = initial[None]
            for k in range(cfg["rollout"]["steps"]):
                proposal = free_position(expected, **cfg["dynamics"])
                expected = finite_difference_state(expected, proposal, cfg["dynamics"]["time_step"])
                torch.testing.assert_close(run["states"][k+1], expected[0])
            self.assertIsNone(run["failure"])
            self.assertEqual(run["summary"]["unconverged_steps"], 0)

    def test_resting_floor_projection_and_fd(self):
        cfg = config()
        initial = torch.tensor([[0., .45-cfg["physics"]["slop"], 0., 0.]])
        for model in (None, ConditionalField(**cfg["model"]).eval()):
            for calls in (1, 8):
                run = simulate(initial, torch.tensor([.45]), cfg, torch.device("cpu"), model, calls)
                self.assertIsNone(run["failure"])
                torch.testing.assert_close(run["states"][-1], initial, atol=1e-4, rtol=0)
                self.assertLess(run["summary"]["max_geometric_penetration"], .001)

    def test_swept_diagnostic_catches_endpoint_invisible_crossing(self):
        start = torch.tensor([[-1., 1.], [1., 1.]])
        end = start.flip(0)
        result = swept_pairs(start, end, torch.tensor([.4, .4]), 0.)
        self.assertEqual(result["endpoint_missed_count"], 1)
        self.assertAlmostEqual(result["max_penetration"], .8, places=6)

    def test_guard_failure_is_not_used_as_next_frame(self):
        model = EndpointFunction(lambda x, tau, problem: torch.full_like(x, 1000))
        cfg = config()
        run = simulate(torch.tensor([[0., .44, 0., 0.]]), torch.tensor([.45]), cfg,
                       torch.device("cpu"), model, guarded=True)
        self.assertEqual(run["completed_steps"], 0)
        self.assertEqual(len(run["states"]), 1)
        self.assertEqual(run["failure"]["reason"], "incomplete_solver_clock")

    def test_report_and_renderer(self):
        cfg = config()
        cfg["rollout"]["render"] = True
        model = ConditionalField(**cfg["model"]).eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"rollout.json"
            report = evaluate_motion({"cfm": model}, cfg, torch.device("cpu"), output)
            self.assertEqual(len(report["scenes"]), 4)
            for name, scene in report["scenes"].items():
                self.assertTrue((Path(directory)/f"{name}.gif").exists())
                self.assertEqual(scene["pgs"]["completed_steps"], 3)
                self.assertEqual(scene["cfm_k1_raw"]["pgs_comparison_frames"], 3)


class HybridMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_each_frame_matches_manual_fm_pgs_composition_for_both_heads(self):
        cfg = config()
        physics = PhysicsConfig(**cfg["physics"])
        _, initial, radius = motion_scenes(physics)[2]
        for head in ("analytic", "direct"):
            model = ConditionalField(**cfg["model"], head_type=head).eval()
            if head == "direct":
                with torch.no_grad():
                    model.head.bias.fill_(-1.)
            state, expected, finishes = initial.clone(), [initial.clone()], []
            for _ in range(cfg["rollout"]["steps"]):
                proposal = free_position(state[None], **cfg["dynamics"])[0]
                problem, _ = build_frame_problem(proposal, radius, physics, cfg["reference"]["eta_fraction"])
                start = torch.zeros_like(problem["c"])
                prefix = run_cfm(model, problem, start, 4)
                finish = pgs(problem, prefix["final"], cfg["evaluation"]["tolerance"],
                             cfg["reference"]["max_sweeps"])
                finishes.append((finish, bool(converged(problem, prefix["final"], cfg["evaluation"]["tolerance"])[0])))
                position = decode(problem, finish["final"]).reshape(1, len(radius), 2)
                state = finite_difference_state(state[None], position, cfg["dynamics"]["time_step"])[0]
                expected.append(state.clone())
            # Only four actual operations are timed per successful frame:
            # free dynamics, geometry, continuous solve, and velocity update.
            with patch("src.multiplier_flow.rollout.timed", side_effect=lambda device, op: (op(), .125)):
                run = simulate(initial, radius, cfg, torch.device("cpu"), model, 4, hybrid=True)
            self.assertIsNone(run["failure"])
            torch.testing.assert_close(run["states"], torch.stack(expected), atol=0, rtol=0)
            for row, (finish, cfm_converged) in zip(run["frames"], finishes):
                self.assertTrue(row["solver_converged"])
                self.assertEqual(row["cfm_converged"], cfm_converged)
                self.assertEqual(row["nfe"], 4)
                self.assertEqual(row["sweeps"], int(finish["sweeps"][0]))
                self.assertEqual(row["contact_evals"], int(finish["contact_evals"][0]))
                self.assertEqual(row["solver_seconds"], .125)
                self.assertEqual(row["simulation_seconds"], .5)
                self.assertNotIn("unclipped_diagnostic", row)
            self.assertEqual(run["summary"]["total_nfe"], 12)
            self.assertEqual(run["summary"]["total_pgs_sweeps"], sum(int(f[0]["sweeps"][0]) for f in finishes))
            if head == "direct":
                self.assertGreater(sum(row["clipped_entries"] for row in run["frames"]), 0)

    def test_free_flight_and_isolated_contact_need_no_pgs_updates(self):
        cfg = config()
        model = ConditionalField(**cfg["model"]).eval()
        for height in (3., .45-cfg["physics"]["slop"]):
            initial = torch.tensor([[0., height, 0., 0.]])
            radius = torch.tensor([.45])
            baseline = simulate(initial, radius, cfg, torch.device("cpu"))
            run = simulate(initial, radius, cfg, torch.device("cpu"), model, 8, hybrid=True)
            torch.testing.assert_close(run["states"], baseline["states"], atol=1e-6, rtol=0)
            self.assertIsNone(run["failure"])
            for row in run["frames"]:
                self.assertTrue(row["cfm_completed"] and row["cfm_converged"])
                self.assertEqual(row["nfe"], 8)
                self.assertEqual(row["sweeps"], 0)

    def test_failed_fm_is_not_promoted_to_a_physical_frame(self):
        model = EndpointFunction(lambda x, tau, problem: torch.full_like(x, float("nan")))
        initial = torch.tensor([[0., .44, 0., 0.]])
        run = simulate(initial, torch.tensor([.45]), config(), torch.device("cpu"), model, 4, hybrid=True)
        self.assertEqual(run["completed_steps"], 0)
        torch.testing.assert_close(run["states"], initial[None], atol=0, rtol=0)
        failure = run["failure"]
        self.assertEqual(failure["reason"], "incomplete_solver_clock")
        self.assertFalse(failure["cfm_completed"] or failure["solver_converged"])
        self.assertFalse(failure["pgs_budget_exhausted"])
        self.assertEqual(failure["counters"]["nfe"], 1)
        self.assertEqual(failure["counters"]["sweeps"], 0)
        json.dumps(failure, allow_nan=False)

    def test_pgs_cap_failure_aborts_baseline_and_hybrid_without_advancing(self):
        cfg = config()
        cfg["reference"]["max_sweeps"] = 1
        cfg["evaluation"]["tolerance"] = 1e-8
        initial = torch.tensor([[0., .4, 0., 0.], [0., 1.25, 0., 0.], [0., 2.1, 0., 0.]])
        model = EndpointFunction(lambda x, tau, problem: torch.zeros_like(x))
        for hybrid in (False, True):
            run = simulate(initial, torch.full((3,), .45), cfg, torch.device("cpu"),
                           model if hybrid else None, 4, hybrid=hybrid)
            self.assertEqual(run["completed_steps"], 0)
            self.assertEqual(len(run["states"]), 1)
            failure = run["failure"]
            self.assertEqual(failure["reason"], "pgs_budget_exhausted")
            self.assertTrue(failure["pgs_budget_exhausted"])
            self.assertFalse(failure["solver_converged"])
            self.assertEqual(failure["counters"]["sweeps"], 1)
            self.assertGreater(failure["projected_gradient"], cfg["evaluation"]["tolerance"])
            json.dumps(failure, allow_nan=False)

    def test_hybrid_report_and_three_panel_gifs_include_matching_raw_budgets(self):
        cfg = config()
        cfg["rollout"].update(render=True, hybrid={"enabled": True, "calls": [4, 8, 16]})
        original = copy.deepcopy(cfg)
        model = ConditionalField(**cfg["model"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"rollout.json"
            with patch("src.multiplier_flow.rollout.render_motion", wraps=render_motion) as renderer:
                report = evaluate_motion({"cfm": model}, cfg, torch.device("cpu"), output)
            self.assertFalse(model.training)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["hybrid"]["calls"], [4, 8, 16])
            self.assertEqual(saved["standalone_neural_calls"], [1, 4, 8, 16])
            self.assertEqual(renderer.call_count, 12)
            for call in renderer.call_args_list:
                keys = list(call.args[0])
                self.assertEqual(keys[0], "pgs")
                self.assertTrue(keys[1].endswith("_raw"))
                self.assertEqual(keys[2], keys[1].replace("_raw", "_hybrid"))
            for name, scene in report["scenes"].items():
                self.assertEqual(len(saved["render_files"][name]), 3)
                for calls in (4, 8, 16):
                    row = scene[f"cfm_k{calls}_hybrid"]
                    self.assertEqual(row["fm_calls"], calls)
                    self.assertEqual(row["completed_steps"], 3)
                    self.assertEqual(row["pgs_comparison_frames"], 3)
                    for frame in row["frames"]:
                        self.assertTrue(frame["solver_converged"])
                        for metric in ("sweeps", "nfe", "solver_seconds", "simulation_seconds",
                                       "geometric_penetration", "linear_penetration", "cfm_converged"):
                            self.assertIn(metric, frame)
                    path = Path(directory)/f"{name}_cfm_k{calls}_hybrid.gif"
                    with Image.open(path) as gif:
                        self.assertEqual(gif.width, 3*cfg["rollout"]["image_size"])
                        self.assertGreater(gif.n_frames, 1)
                    self.assertTrue(path.with_suffix(".png").exists())
        self.assertEqual(cfg, original)

    def test_renderer_shows_failure_even_if_no_solver_advances(self):
        cfg = config()
        initial, radius = torch.tensor([[0., .44, 0., 0.]]), torch.tensor([.45])
        model = EndpointFunction(lambda x, tau, problem: torch.full_like(x, float("nan")))
        run = simulate(initial, radius, cfg, torch.device("cpu"), model, 4, hybrid=True)
        settings = dict(cfg["rollout"], time_step=cfg["dynamics"]["time_step"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"failed.gif"
            render_motion({"pgs": run, "hybrid": run}, radius, PhysicsConfig(), settings, path)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 2)

    def test_invalid_budgets_and_hybrid_modes_are_rejected(self):
        cfg = config()
        for calls in ([], [0], [True], [4, 4]):
            for key in ("calls", "hybrid"):
                settings = copy.deepcopy(cfg["rollout"])
                settings[key] = calls if key == "calls" else {"enabled": True, "calls": calls}
                with self.assertRaisesRegex(ValueError, "rollout"):
                    rollout_budgets(settings)
        initial, radius = torch.tensor([[0., .44, 0., 0.]]), torch.tensor([.45])
        for model, guarded in ((None, False), (ConditionalField(**cfg["model"]), True)):
            with self.assertRaisesRegex(ValueError, "unguarded"):
                simulate(initial, radius, cfg, torch.device("cpu"), model, guarded=guarded, hybrid=True)

    def test_pgs_only_ignores_configured_hybrid_runs(self):
        cfg = config()
        cfg["rollout"]["hybrid"] = {"enabled": True, "calls": [4]}
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_motion({}, cfg, torch.device("cpu"), Path(directory)/"rollout.json")
        self.assertFalse(report["hybrid"]["enabled"])
        self.assertEqual(report["hybrid"]["calls"], [])
        for scene in report["scenes"].values():
            self.assertEqual(list(scene), ["pgs"])

    def test_cli_selects_d_best_solver_and_independent_rollout_budgets(self):
        cfg = config()
        cfg["outdir"] = "runs/test"
        checkpoint = dict(epoch=1, updates=2, format="test", solver="test", config=cfg, selection_metric="test")
        model = ConditionalField(**cfg["model"]).eval()
        with tempfile.TemporaryDirectory() as directory:
            argv = ["eval_multiplier_rollout.py", "--variant", "D", "--device", "cpu",
                    "--calls", "4", "--hybrid-calls", "4", "8", "16", "--output", directory]
            with patch("sys.argv", argv), patch.object(eval_multiplier_rollout, "load_config", return_value=cfg), \
                    patch.object(eval_multiplier_rollout, "load_model", return_value=(model, checkpoint)) as loader, \
                    patch.object(eval_multiplier_rollout, "evaluate_motion") as evaluate:
                eval_multiplier_rollout.main()
            self.assertEqual(loader.call_args.args[0], Path("runs/test/D/cfm/best_solver.pt"))
            selected = evaluate.call_args.args[1]
            self.assertEqual(selected["rollout"]["calls"], [4])
            self.assertEqual(selected["rollout"]["hybrid"], {"enabled": True, "calls": [4, 8, 16]})
            self.assertEqual(selected["evaluation"], cfg["evaluation"])

    def test_cli_rejects_conflicts_and_bad_budgets_before_loading_checkpoint(self):
        for args in (["--no-hybrid", "--hybrid-calls", "4"], ["--pgs-only", "--hybrid"],
                     ["--hybrid-calls", "0"], ["--hybrid-calls", "4", "4"]):
            with self.subTest(args=args), patch("sys.argv", ["eval_multiplier_rollout.py", *args]), \
                    patch.object(eval_multiplier_rollout, "load_config", return_value=config()), \
                    patch.object(eval_multiplier_rollout, "load_model") as loader, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                eval_multiplier_rollout.main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
