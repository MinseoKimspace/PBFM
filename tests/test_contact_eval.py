import json
import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import eval_projection as evaluation
from src.contact_flow.physics import FlowConfig, PhysicsConfig, make_baseline
from src.contact_flow.solver import SolverConfig


def outcome(final, converged=True, failed=False):
    batch = final.shape[0]
    flag = lambda value: torch.full((batch,), value, dtype=torch.bool, device=final.device)
    number = torch.ones(batch, dtype=torch.long, device=final.device)
    return {"final": final, "time": number.float(), "converged": flag(converged),
            "completed": flag(not failed), "failed": flag(failed),
            "nfe": number, "backtracks": number * 0, "energy_evals": number}


class ContactEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.physics = PhysicsConfig()
        self.dynamics = {"time_step": 0.1, "gravity_y": 0.0, "linear_damping": 0.0}
        self.solver = SolverConfig(steps=4)
        self.radius = torch.tensor([[0.5]])
        self.state = torch.tensor([[[0.0, 3.0, 0.2, 0.0]]])
        self.factory = lambda old, proposal, r: make_baseline("gradient", r, self.physics)

    def test_linear_newton_and_flow_are_distinct(self):
        diagnostic = evaluation.linear_residual_diagnostic(32)
        self.assertEqual(diagnostic["newton_unit_step_residual_ratio"], 0.0)
        self.assertAlmostEqual(diagnostic["continuous_unit_time_residual_ratio"], math.exp(-1))
        self.assertGreater(diagnostic["euler_unit_time_residual_ratio"], 0.3)
        self.assertLess(diagnostic["euler_unit_time_residual_ratio"], math.exp(-1))
        with self.assertRaises(ValueError):
            evaluation.linear_residual_diagnostic(0)

    def test_actual_fixed_contact_matches_regularized_euler(self):
        for steps in (1, 4, 16):
            diagnostic = evaluation.fixed_contact_diagnostic(steps)
            self.assertFalse(diagnostic["failed"])
            self.assertLess(diagnostic["absolute_error"], 1e-12)

    def test_named_preflight_exercises_contact_activation(self):
        report = evaluation.physical_preflight(self.physics, self.solver,
                                               ["gradient"], torch.device("cpu"), gains=[1],
                                               normalizations=["none"], pbd_sweeps=[4])
        self.assertIn("single_ground_contact/gradient/gain1/classical", report)
        self.assertIn("fixed_normal_three_stack/gradient/gain1/classical", report)
        self.assertIn("fixed_normal_five_stack/gradient/gain1/classical", report)
        changing = report["changing_contact_chain/gradient/gain1/classical"]
        self.assertGreater(changing["active_contact_changes"], 0)
        self.assertLess(changing["energy_final_mean"], changing["energy_initial_mean"])

    def test_finite_json_policy(self):
        self.assertIsNone(evaluation.finite_number(float("inf")))
        self.assertIsNone(evaluation.finite_number(float("nan")))
        json.dumps({"value": evaluation.finite_number(float("inf"))}, allow_nan=False)

    def test_real_free_flight_rollout_preserves_fd_state(self):
        metrics = evaluation.rollout_metrics(self.state, self.radius, self.factory, self.physics,
                                              self.dynamics, self.solver, 4)
        self.assertEqual(metrics["completed_steps_per_world"], [4])
        self.assertEqual(metrics["failed_world_count"], 0)
        self.assertEqual(metrics["unconverged_projection_steps"], 0)
        self.assertAlmostEqual(metrics["projection_kinetic_energy_change_mean"], 0, places=6)
        self.assertEqual(metrics["initial_active_pair_count"], 0)
        self.assertIsNone(metrics["initial_active_pair_gap_mean"])
        json.dumps(metrics, allow_nan=False)

    def test_finite_tolerance_miss_continues_physical_rollout(self):
        seen = []

        def fake(field, proposal, radius, physics, solver, **kwargs):
            seen.append(proposal.clone())
            return outcome(proposal, converged=False)

        with patch.object(evaluation, "integrate", side_effect=fake):
            metrics = evaluation.rollout_metrics(self.state, self.radius, self.factory, self.physics,
                                                  self.dynamics, self.solver, 3)
        self.assertEqual(metrics["completed_steps_per_world"], [3])
        self.assertEqual(metrics["unconverged_projection_steps"], 3)
        self.assertEqual(metrics["first_unconverged_step_per_world"], [0])
        self.assertAlmostEqual(float(seen[-1][0, 0, 0]), 0.06, places=6)

    def test_failed_world_is_never_restarted_or_replaced(self):
        state = self.state.repeat(2, 1, 1)
        state[1, 0, 0] = 1
        radius = self.radius.repeat(2, 1)
        calls = []

        def fake(field, proposal, radius, physics, solver, **kwargs):
            calls.append(proposal.shape[0])
            result = outcome(proposal)
            if len(calls) == 1:
                result["failed"][0] = True
                result["completed"][0] = False
                result["converged"][0] = False
                result["final"][0] = float("nan")
            return result

        with patch.object(evaluation, "integrate", side_effect=fake):
            metrics = evaluation.rollout_metrics(state, radius, self.factory, self.physics,
                                                  self.dynamics, self.solver, 3)
        self.assertEqual(calls, [2, 1, 1])
        self.assertEqual(metrics["completed_steps_per_world"], [0, 3])
        self.assertEqual(metrics["failed_world_count"], 1)
        self.assertEqual(sum(metrics["failure_reasons"].values()), 1)
        json.dumps(metrics, allow_nan=False)

    def test_metrics_use_float64_and_global_penetration_max(self):
        state = self.state.repeat(2, 1, 1)
        radius = self.radius.repeat(2, 1)
        final = state[..., :2].clone()
        final[0, 0, 1] = -1e20
        final[1, 0, 1] = -1
        with patch.object(evaluation, "integrate", return_value=outcome(final, converged=False)):
            report, _ = evaluation.evaluate_projection(state, radius, self.factory, self.physics,
                                                        self.dynamics, self.solver)
        self.assertTrue(math.isfinite(report["energy_final_mean"]))
        self.assertGreater(report["penetration_max"], 9e19)
        self.assertLess(report["penetration_max"], 1.1e20)
        json.dumps(report, allow_nan=False)

    def test_old_projection_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"projection_type": "fm", "model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                evaluation.load_checkpoint(path, torch.device("cpu"))

    def test_cli_honors_output_and_render_settings_without_clobbering_eval(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            normal = base / "custom.json"
            normal.write_text("keep me", encoding="utf-8")
            cfg = {"eval": {"count": 1, "num_objects": 1, "methods": ["inverse"], "solver_steps": [4], "pbd_sweeps": []},
                   "runtime": {"device": "cpu", "cpu_threads": 1},
                   "output": {"json": str(normal)}, "render": {"dir": str(base / "pictures"), "size": 321}}
            with patch("sys.argv", ["eval_projection.py", "--preflight"]), \
                 patch("src.contact_flow.io.load_config", return_value=cfg), \
                 patch.object(evaluation, "physical_preflight", return_value={"checked": True}), \
                 patch.object(evaluation, "_render", return_value=True) as render, redirect_stdout(io.StringIO()):
                evaluation.main()
            report = json.loads((base / "custom_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(normal.read_text(encoding="utf-8"), "keep me")
            self.assertEqual(report["rollout_steps"], 0)
            self.assertEqual(report["render_size"], 321)
            self.assertEqual(report["format"], "contact_flow_eval_v2")
            self.assertEqual(report["start_noise_std"], 0.0)
            self.assertEqual(render.call_args.args[2], base / "pictures" / "preflight" /
                             "sample_solvers" / "inverse_k4.png")
            self.assertEqual(render.call_args.args[4], 321)

    def test_cli_energy_checkpoint_uses_saved_nondefault_dynamics(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "energy_eval.json"
            cfg = {"eval": {"checkpoint": "unused.pt", "count": 1, "num_objects": 1,
                            "methods": [], "solver_steps": [1], "rollout_steps": 0, "pbd_sweeps": []},
                   "runtime": {"device": "cpu", "cpu_threads": 1}, "output": {"json": str(path)}}
            saved_dynamics = {"time_step": 0.01, "gravity_y": -1.0, "linear_damping": 0.0}
            checkpoint = {"objective": "energy", "physics": asdict(self.physics),
                          "dynamics": saved_dynamics, "solver": asdict(self.solver), "flow": asdict(FlowConfig(gain=4))}
            model = lambda z, tau, radius, condition, physics: torch.zeros_like(z)
            with patch("sys.argv", ["eval_projection.py"]), \
                 patch("src.contact_flow.io.load_config", return_value=cfg), \
                 patch("src.contact_flow.io.load_states", return_value=(self.state, self.radius, ["free_flight"])) as load_states, \
                 patch.object(evaluation, "load_checkpoint", return_value=(model, checkpoint)), \
                 patch.object(evaluation, "_render", return_value=False), redirect_stdout(io.StringIO()):
                evaluation.main()
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["checkpoint_objective"], "energy")
            self.assertIn("energy_k1", report["results"])
            self.assertNotIn("fm_k1", report["results"])
            self.assertEqual(report["dynamics"], saved_dynamics)
            self.assertEqual(report["flow"]["gain"], 4)
            self.assertEqual(report["scene_types"], ["free_flight"])
            self.assertTrue(load_states.call_args.kwargs["return_scene_types"])
            self.assertEqual(load_states.call_args.kwargs["physics"], self.physics)
            self.assertEqual(report["results"]["energy_k1"]["convergence_breakdown"]
                             ["scene_types"]["free_flight"]["samples"], 1)

    def test_v1_checkpoint_rejected_and_v2_flow_restored(self):
        from src.contact_flow.model import ContactFlowNet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"format": "contact_flow_v1"}, path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                evaluation.load_checkpoint(path, torch.device("cpu"))
            flow = FlowConfig(gain=4, normalization="none", mobility_bound=8)
            kwargs = {"hidden_dim": 8, "time_dim": 4, "message_steps": 1, "rank": 1}
            model = ContactFlowNet(**kwargs, flow=flow)
            torch.save({"format": "contact_flow_v2", "flow": asdict(flow),
                        "model_kwargs": kwargs, "model_state_dict": model.state_dict()}, path)
            restored, checkpoint = evaluation.load_checkpoint(path, torch.device("cpu"))
            self.assertFalse(restored.training)
            self.assertEqual(checkpoint["flow"], asdict(flow))

    def test_cli_rejects_inference_only_flow_overrides(self):
        checkpoint = {"objective": "fm", "physics": asdict(self.physics), "dynamics": self.dynamics,
                      "solver": asdict(self.solver), "flow": asdict(FlowConfig(gain=4))}
        for changed in ({"gain": 16}, {"normalization": "none"}, {"mobility_bound": 32}):
            with patch("sys.argv", ["eval_projection.py"]), \
                 patch("src.contact_flow.io.load_config", return_value={
                     "eval": {"checkpoint": "unused.pt"}, "flow": changed}), \
                 patch.object(evaluation, "load_checkpoint", return_value=(object(), checkpoint)), \
                 patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main()

    def test_preflight_sweeps_gain_normalization_and_native_pbd(self):
        report = evaluation.physical_preflight(self.physics, self.solver, ["isotropic"],
            torch.device("cpu"), gains=[1, 4], normalizations=["none", "diagonal"], pbd_sweeps=[1, 4])
        self.assertEqual(len(report), 6 * (2 * 2 + 2))
        self.assertIn("fixed_normal_eight_stack/isotropic/gain4/diagonal/cap16", report)
        self.assertIn("fixed_normal_ten_stack/isotropic/gain4/diagonal/cap16", report)
        low = report["single_ground_contact/isotropic/gain1/diagonal/cap16"]
        high = report["single_ground_contact/isotropic/gain4/diagonal/cap16"]
        self.assertLess(high["max_violation"], low["max_violation"])
        native = report["single_ground_contact/pbd/sweeps1"]
        self.assertEqual(native["converged_count"], 1)
        self.assertEqual(native["field_evaluations"], 0)
        self.assertGreater(native["contact_evals"], 0)
        self.assertIsNone(native["tau_mean"])
        self.assertIsNone(native["flow"])
        json.dumps(report, allow_nan=False)

    def test_preflight_mobility_cap_exposes_five_stack_slow_mode(self):
        report = evaluation.physical_preflight(self.physics, SolverConfig(steps=64, max_steps=1024),
            ["inverse"], torch.device("cpu"), gains=[32], normalizations=["diagonal"],
            pbd_sweeps=[1], mobility_bounds=[16, 128])
        low = report["fixed_normal_five_stack/inverse/gain32/diagonal/cap16"]
        high = report["fixed_normal_five_stack/inverse/gain32/diagonal/cap128"]
        self.assertEqual(low["flow"]["mobility_bound"], 16)
        self.assertEqual(high["flow"]["mobility_bound"], 128)
        self.assertLess(high["max_violation"], low["max_violation"])
        self.assertEqual(high["converged_count"], 1)
        self.assertEqual(low["converged_count"], 0)

    def test_preflight_classical_methods_do_not_duplicate_irrelevant_caps(self):
        report = evaluation.physical_preflight(self.physics, self.solver, ["gradient", "jacobi"],
            torch.device("cpu"), gains=[1, 4], normalizations=["none", "diagonal"],
            pbd_sweeps=[1], mobility_bounds=[16, 128])
        self.assertEqual(len(report), 6 * (2 * 2 + 1))
        self.assertIn("fixed_normal_five_stack/gradient/gain4/classical", report)
        self.assertFalse(any("cap" in key for key in report))
        with self.assertRaisesRegex(ValueError, "mobility_bounds"):
            evaluation.physical_preflight(self.physics, self.solver, ["inverse"],
                torch.device("cpu"), mobility_bounds=[])
        with self.assertRaisesRegex(ValueError, "stack_sizes"):
            evaluation.physical_preflight(self.physics, self.solver, ["inverse"],
                torch.device("cpu"), stack_sizes=[5, 5])

    def test_one_step_initial_contact_groups_and_scene_labels(self):
        state, radius = self.state.repeat(2, 1, 1), self.radius.repeat(2, 1)
        state[0, 0, 1] = 0.4
        metrics, _ = evaluation.evaluate_projection(state, radius, None, self.physics,
            self.dynamics, self.solver, projection_integrator=evaluation.pbd_integrator(2),
            scene_types=["contact", "free_flight"])
        groups = metrics["convergence_breakdown"]
        self.assertEqual(groups["initially_violating"]["samples"], 1)
        self.assertEqual(groups["initially_violating"]["converged_count"], 1)
        self.assertEqual(groups["initially_solved"]["samples"], 1)
        self.assertEqual(groups["scene_types"]["contact"]["samples"], 1)
        self.assertEqual(metrics["start_noise_std"], 0.0)

    def test_native_pbd_rollout_counts_sweeps_and_preserves_free_motion(self):
        metrics = evaluation.rollout_metrics(self.state, self.radius, None, self.physics,
            self.dynamics, self.solver, 3, projection_integrator=evaluation.pbd_integrator(2))
        self.assertEqual(metrics["completed_steps_per_world"], [3])
        self.assertEqual(metrics["field_evaluations"], 0)
        self.assertEqual(metrics["native_pbd_sweeps"], 0)
        self.assertAlmostEqual(metrics["projection_kinetic_energy_change_mean"], 0, places=6)

    def test_native_pbd_budget_miss_is_reported_and_rollout_continues(self):
        z = torch.tensor([[[0.0, 0.42 + 0.94 * index] for index in range(5)]])
        state = torch.cat([z, torch.zeros_like(z)], -1)
        radius = z.new_full(z.shape[:2], 0.5)
        native = evaluation.pbd_integrator(1)
        report, _ = evaluation.evaluate_projection(state, radius, None, self.physics,
            self.dynamics, self.solver, projection_integrator=native)
        self.assertEqual(report["integration_failed_count"], 0)
        self.assertEqual(report["budget_exhausted_count"], 1)
        self.assertEqual(report["tolerance_miss_count"], 1)
        rollout = evaluation.rollout_metrics(state, radius, None, self.physics,
            self.dynamics, self.solver, 3, projection_integrator=native)
        self.assertEqual(rollout["completed_steps_per_world"], [3])
        self.assertEqual(rollout["failed_world_count"], 0)
        self.assertGreater(rollout["budget_exhausted_projection_steps"], 0)

    def test_cache_oracle_summary_preserves_full_candidate_accounting(self):
        flow = FlowConfig(gain=4)
        buffer = {"metadata": {"physics": asdict(self.physics), "flow": asdict(flow),
                               "paths": {"temperature": 0.1, "weighting": "uniform"}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.pt"
            torch.save({"format": "contact_path_cache_v2", "spec": {},
                        "buffers": {"train": buffer, "val": buffer}}, path)
            summary = {"oracle": {"all_candidate_field_evaluations": 80}}
            with patch("src.contact_flow.paths.path_summary", return_value=summary) as summarize:
                result = evaluation.summarize_path_cache(path, self.physics, flow)
            self.assertEqual(summarize.call_count, 2)
            self.assertEqual(summarize.call_args.args[1:], (0.1, "uniform"))
            self.assertEqual(result["splits"]["train"]["oracle"]["all_candidate_field_evaluations"], 80)
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluation.summarize_path_cache(path, self.physics, FlowConfig(gain=1))
            torch.save({"format": "contact_path_cache_v1", "buffers": {"train": buffer}}, path)
            with self.assertRaisesRegex(ValueError, "v2 cache"):
                evaluation.summarize_path_cache(path, self.physics, flow)

    def test_actual_v2_training_cache_wrapper_is_evaluable(self):
        from src.contact_flow.paths import build_path_buffer
        flow = FlowConfig(gain=4)
        buffer = build_path_buffer(self.state, self.radius, self.physics, self.dynamics,
            {"candidates": 2, "rank": 1, "start_noise_std": 0.0}, self.solver, seed=3, flow=flow)
        buffer["scene_type"] = ["free_flight"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.pt"
            torch.save({"format": "contact_path_cache_v2", "spec": {},
                        "buffers": {"train": buffer, "val": buffer}}, path)
            report = evaluation.summarize_path_cache(path, self.physics, flow)
            self.assertEqual(set(report["splits"]), {"train", "val"})
            train = report["splits"]["train"]
            self.assertEqual(train["groups"]["scene:free_flight"]["scene_success_rate"], 1.0)
            self.assertEqual(train["oracle"]["charged_cost"], train["candidate_generation_cost"])
            self.assertEqual(report["temperature_source"], "cache_generation_initial")
            json.dumps(report, allow_nan=False)

    def test_cache_provenance_and_checkpoint_temperature_are_enforced(self):
        flow = FlowConfig(gain=4)
        spec = {"seed": 42, "dynamics": self.dynamics, "paths": {"candidate_pool": "informed"}}
        buffer = {"metadata": {"physics": asdict(self.physics), "flow": asdict(flow),
                               "paths": {"temperature": 0.1, "weighting": "score"}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.pt"
            torch.save({"format": "contact_path_cache_v2", "spec": spec,
                        "buffers": {"train": buffer, "val": buffer}}, path)
            with patch("src.contact_flow.paths.path_summary", return_value={}) as summary:
                result = evaluation.summarize_path_cache(path, self.physics, flow,
                    expected_spec=spec, temperature=0.00003, weighting="uniform")
            self.assertEqual(summary.call_args.args[1:], (0.00003, "uniform"))
            self.assertTrue(result["checkpoint_spec_verified"])
            self.assertEqual(result["temperature_source"], "checkpoint_epoch")
            self.assertEqual(result["weighting_source"], "checkpoint_epoch")
            for changed in ({**spec, "seed": 43},
                            {**spec, "paths": {"candidate_pool": "random"}},
                            {**spec, "dynamics": {**self.dynamics, "time_step": 0.2}}):
                with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
                    evaluation.summarize_path_cache(path, self.physics, flow, expected_spec=changed)

    def test_cli_path_cache_override_uses_checkpoint_provenance(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = {"eval": {"checkpoint": "unused.pt", "path_cache": "wrong_default.pt", "count": 1,
                            "num_objects": 1, "methods": [], "solver_steps": [1], "pbd_sweeps": [],
                            "rollout_steps": 0}, "runtime": {"device": "cpu", "cpu_threads": 1},
                   "output": {"json": str(base / "eval.json")}}
            spec = {"seed": 7, "paths": {"candidate_pool": "random"}}
            checkpoint = {"objective": "fm", "physics": asdict(self.physics), "dynamics": self.dynamics,
                          "solver": asdict(self.solver), "flow": asdict(FlowConfig(gain=4)),
                          "reference_cache_spec": spec, "temperature": 0.00003, "weighting": "uniform"}
            model = lambda z, tau, radius, condition, physics: torch.zeros_like(z)
            with patch("sys.argv", ["eval_projection.py", "--path-cache", "matching.pt"]), \
                 patch("src.contact_flow.io.load_config", return_value=cfg), \
                 patch("src.contact_flow.io.load_states", return_value=(self.state, self.radius, ["free_flight"])), \
                 patch.object(evaluation, "load_checkpoint", return_value=(model, checkpoint)), \
                 patch.object(evaluation, "summarize_path_cache", return_value={"verified": True}) as summary, \
                 patch.object(evaluation, "_render", return_value=False), redirect_stdout(io.StringIO()):
                evaluation.main()
            self.assertEqual(summary.call_args.args[0], "matching.pt")
            self.assertEqual(summary.call_args.kwargs, {
                "expected_spec": spec, "temperature": 0.00003, "weighting": "uniform"})
            report = json.loads((base / "eval.json").read_text(encoding="utf-8"))
            self.assertTrue(report["reference_path_diagnostics"]["verified"])

    def test_cli_runs_real_preflight_and_native_pbd(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = {"eval": {"count": 1, "num_objects": 1, "methods": ["isotropic"],
                            "solver_steps": [4], "pbd_sweeps": [1]},
                   "preflight": {"gains": [1, 4], "normalizations": ["none", "diagonal"],
                                 "mobility_bounds": [16, 128], "pbd_sweeps": [1]},
                   "flow": asdict(FlowConfig(gain=4)),
                   "runtime": {"device": "cpu", "cpu_threads": 1},
                   "output": {"json": str(base / "eval.json")},
                   "render": {"dir": str(base / "renders"), "size": 96}}
            with patch("sys.argv", ["eval_projection.py", "--preflight"]), \
                 patch("src.contact_flow.io.load_config", return_value=cfg), redirect_stdout(io.StringIO()):
                evaluation.main()
            result = json.loads((base / "eval_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(result["flow"]["gain"], 4)
            self.assertEqual(len(result["physical_preflight"]), 54)
            self.assertEqual(result["preflight_stack_sizes"], [3, 5, 8, 10])
            self.assertIn("fixed_normal_five_stack/isotropic/gain4/diagonal/cap128",
                          result["physical_preflight"])
            self.assertIn("fixed_normal_ten_stack/isotropic/gain4/diagonal/cap128",
                          result["physical_preflight"])
            self.assertIn("pbd_sweeps1", result["results"])
            self.assertEqual(result["results"]["pbd_sweeps1"]["field_evaluations"], 0)
            self.assertTrue((base / "renders" / "preflight" / "sample_solvers" /
                             "pbd_sweeps1.png").is_file())
            self.assertTrue((base / "renders" / "preflight" / "cases" /
                             "single_ground_contact" /
                             "isotropic__gain4__diagonal__cap16.png").is_file())
            self.assertTrue((base / "renders" / "preflight" / "cases" /
                             "fixed_normal_ten_stack" / "pbd__sweeps1.png").is_file())
            self.assertEqual(result["scene_types"], ["vertical_stack"])
            self.assertIn("vertical_stack", result["results"]["pbd_sweeps1"]
                          ["convergence_breakdown"]["scene_types"])

    def test_preflight_ignores_configured_missing_training_cache(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = {"eval": {"count": 1, "num_objects": 1, "methods": ["gradient"],
                            "solver_steps": [1], "pbd_sweeps": [],
                            "path_cache": str(base / "missing" / "paths.pt")},
                   "runtime": {"device": "cpu", "cpu_threads": 1},
                   "output": {"json": str(base / "eval.json")}}
            with patch("sys.argv", ["eval_projection.py", "--preflight"]), \
                 patch("src.contact_flow.io.load_config", return_value=cfg), \
                 patch.object(evaluation, "physical_preflight", return_value={}), \
                 patch.object(evaluation, "summarize_path_cache") as summary, \
                 patch.object(evaluation, "_render", return_value=False), redirect_stdout(io.StringIO()):
                evaluation.main()
            summary.assert_not_called()
            report = json.loads((base / "eval_preflight.json").read_text(encoding="utf-8"))
            self.assertNotIn("reference_path_diagnostics", report)

    def test_cli_rejects_empty_or_invalid_budgets(self):
        for budgets in ([], [0], [-1]):
            with patch("sys.argv", ["eval_projection.py", "--preflight"]), \
                 patch("src.contact_flow.io.load_config", return_value={"eval": {"solver_steps": budgets}}), \
                 patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main()


if __name__ == "__main__":
    unittest.main()
