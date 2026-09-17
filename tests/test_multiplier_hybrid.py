"""Explicit FM-to-PGS handoff, failure accounting and continuous timing."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.multiplier_flow.benchmark import hybrid_budgets, hybrid_comparison, hybrid_summary, work_summary
from src.multiplier_flow.evaluation import evaluate, hybrid_stage_timings
from src.multiplier_flow.model import ConditionalField
from src.multiplier_flow.problem import make_problem, pack
from src.multiplier_flow.solvers import pgs, run_cfm, run_hybrid
from test_multiplier_flow import EndpointFunction, scalar_problem, tiny_config


def coupled_problem():
    return pack([make_problem(torch.zeros(2, dtype=torch.float64),
        torch.tensor([[1., 0.], [.5, math.sqrt(.75)]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64), -torch.ones(2, dtype=torch.float64))])


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_handoff_matches_manual_composition_for_both_heads(self):
        problem = coupled_problem()
        original_problem = {k: v.clone() for k, v in problem.items()}
        start = torch.zeros_like(problem["c"])
        for head in ("analytic", "direct"):
            model = ConditionalField(hidden_dim=8, message_steps=1, head_type=head).double()
            with torch.no_grad():
                model.head.bias.fill_(.3)
            for calls in (4, 8, 16):
                prefix = run_cfm(model, problem, start, calls, collect_diagnostics=False)
                expected = pgs(problem, prefix["final"], 1e-8, 100)
                actual = run_hybrid(model, problem, start, calls, 1e-8, 100)
                torch.testing.assert_close(actual["cfm_final"], prefix["final"], atol=0, rtol=0)
                for key in ("final", "sweeps", "contact_evals", "converged"):
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                self.assertEqual(actual["nfe"].tolist(), [calls])
                self.assertTrue(actual["completed"].all())
                self.assertTrue(actual["converged"].all())
                self.assertFalse(actual["pgs_budget_exhausted"].any())
        for key in problem:
            torch.testing.assert_close(problem[key], original_problem[key], atol=0, rtol=0)
        self.assertEqual(float(start.abs().sum()), 0.)

    def test_solved_fm_endpoint_and_empty_world_need_no_pgs_updates(self):
        problem = scalar_problem()
        model = ConditionalField(hidden_dim=8, message_steps=1).double()
        result = run_hybrid(model, problem, torch.zeros_like(problem["c"]), 8)
        self.assertTrue(result["cfm_converged"].all())
        self.assertTrue(result["converged"].all())
        self.assertEqual(result["sweeps"].tolist(), [0])
        self.assertEqual(result["contact_evals"].tolist(), [0])
        cost = work_summary(result, problem, model)
        self.assertEqual(cost["executed_batch_neural_calls"], 8)
        self.assertEqual(cost["executed_batch_pgs_sweeps"], 0)

        empty = pack([make_problem(torch.zeros(1, dtype=torch.float64),
            torch.zeros(0, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.empty(0, dtype=torch.float64))])
        result = run_hybrid(model, empty, torch.zeros_like(empty["c"]), 4)
        self.assertTrue(result["converged"].all())
        self.assertEqual(result["sweeps"].tolist(), [0])
        self.assertEqual(result["nfe"].tolist(), [4])

    def test_failed_fm_row_is_not_finished_or_silently_counted_as_success(self):
        problem = scalar_problem()
        problem = {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in problem.items()}
        start = torch.tensor([[.1], [0.]], dtype=torch.float64)

        def endpoint(state, tau, context):
            predicted = torch.full_like(state, .05)
            predicted[0] = float("nan")
            return predicted

        result = run_hybrid(EndpointFunction(endpoint), problem, start, 4)
        # Failed row retains its last valid state even though that state is KKT.
        self.assertEqual(result["completed"].tolist(), [False, True])
        self.assertEqual(result["converged"].tolist(), [False, True])
        self.assertEqual(result["sweeps"].tolist(), [0, 1])
        self.assertEqual(result["cfm_converged"].tolist(), [False, False])
        self.assertEqual(result["pgs_budget_exhausted"].tolist(), [False, False])
        self.assertEqual(result["failure_code"].tolist(), [3, 0])
        self.assertEqual(float(result["final"][0]), .1)

    def test_pgs_cap_miss_is_explicit_without_extra_finish(self):
        problem = coupled_problem()
        model = EndpointFunction(lambda state, tau, context: torch.zeros_like(state))
        result = run_hybrid(model, problem, torch.zeros_like(problem["c"]), 4, 1e-10, 1)
        self.assertTrue(result["completed"].all())
        self.assertFalse(result["converged"].any())
        self.assertTrue(result["pgs_budget_exhausted"].all())
        self.assertEqual(result["sweeps"].tolist(), [1])
        torch.testing.assert_close(result["final"], torch.tensor([[1., .5]], dtype=torch.float64))

    def test_stage_instrumentation_does_not_change_result_or_reuse_cached_fm(self):
        problem = coupled_problem()
        start = torch.zeros_like(problem["c"])
        seen = []

        def endpoint(state, tau, context):
            seen.append(tau.clone())
            return torch.zeros_like(state)

        model = EndpointFunction(endpoint)
        raw = run_hybrid(model, problem, start, 4)
        intervals = iter((.25, .5))
        instrumented = run_hybrid(model, problem, start, 4,
                                  stage_timer=lambda operation: (operation(), next(intervals)))
        self.assertEqual(len(seen), 8)
        self.assertEqual(instrumented["stage_seconds"], {"cfm": .25, "pgs": .5})
        torch.testing.assert_close(raw["final"], instrumented["final"], atol=0, rtol=0)
        torch.testing.assert_close(raw["sweeps"], instrumented["sweeps"], atol=0, rtol=0)

    def test_stage_replay_rejects_a_different_result(self):
        problem = scalar_problem()
        start = torch.zeros_like(problem["c"])
        model = ConditionalField(hidden_dim=8, message_steps=1).double()
        result = run_hybrid(model, problem, start, 4)
        result["final"] = torch.zeros_like(start)
        with self.assertRaisesRegex(RuntimeError, "replay changed"):
            hybrid_stage_timings(model, problem, start, 4, .001, 100,
                                  torch.device("cpu"), 1, result)

    def test_configuration_and_no_false_full_success_speedup(self):
        self.assertEqual(hybrid_budgets({}), [])
        self.assertEqual(hybrid_budgets({"hybrid": {"enabled": True}}), [4, 8, 16])
        for options in (None, {"enabled": "yes"}, {"enabled": True, "calls": []},
                        {"calls": [True]}, {"calls": [0]}, {"calls": [4, 4]}):
            with self.assertRaises(ValueError):
                hybrid_budgets({"hybrid": options})
        rows = [{"name": "stack_0", "success_count": 1, "sweeps": 5}]
        baseline = dict(per_scene=rows, success_count=1, timing={"median_seconds": 2.})
        failed = dict(per_scene=[dict(rows[0], success_count=0, sweeps=1)],
                      success_count=0, timing={"median_seconds": 1.})
        result = hybrid_comparison({"pgs": baseline, "hybrid_k4_pgs": failed}, [4])["4"]
        self.assertEqual(result["pgs_over_hybrid_time_ratio"], 2.)
        self.assertIsNone(result["speedup_at_full_success"])
        self.assertEqual(result["pgs_only_success"], 1)

    def test_report_tracks_prefix_finish_groups_and_measures_continuous_total(self):
        config = tiny_config()
        config["evaluation"].update(calls=[1], start_modes=["zero"], guarded=False,
            timing_repeats=2, hybrid={"enabled": True, "calls": [4, 8, 16]})
        problem = scalar_problem()
        split = dict(problem=problem, optimum=torch.full_like(problem["c"], .1), names=["floor_0"])
        for head in ("analytic", "direct"):
            model = ConditionalField(hidden_dim=8, message_steps=1, head_type=head)
            if head == "direct":
                with torch.no_grad():
                    model.head.bias.fill_(-1.)  # Clipping occurs during FM, then PGS solves.
            original = copy.deepcopy(config)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "hybrid.json"
                # A total solve gets .123s; each separately timed stage also
                # gets .123s. Report must keep .123 total, not their .246 sum.
                with patch("src.multiplier_flow.evaluation.timed", side_effect=lambda device, operation: (operation(), .123)):
                    report = evaluate(model, split, config, torch.device("cpu"), path)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["hybrid"]["calls"], [4, 8, 16])
                self.assertIn("FM + PGS to tolerance", path.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertEqual(config, original)
            for calls in (4, 8, 16):
                row = report["modes"]["zero"][f"hybrid_k{calls}_pgs"]
                h = row["hybrid"]
                self.assertEqual(row["nfe"], calls)
                self.assertEqual(row["success_count"], 1)
                self.assertEqual(h["cfm_success_count"], int(head == "analytic"))
                self.assertEqual(h["recovered_count"], int(head == "direct"))
                self.assertEqual(h["pgs_sweeps_per_qp"]["max"], int(head == "direct"))
                self.assertEqual(row["cost"]["parameter_count"], sum(p.numel() for p in model.parameters()))
                self.assertEqual(row["groups"]["floor"]["hybrid"], h)
                self.assertAlmostEqual(row["timing"]["median_seconds"], .123)
                stages = row["timing"]["stage_diagnostics"]
                self.assertAlmostEqual(stages["cfm"]["median_seconds"] + stages["pgs"]["median_seconds"], .246)
                comparison = report["hybrid_comparison"]["zero"][str(calls)]
                self.assertTrue(comparison["both_solve_all"])
                self.assertEqual(comparison["speedup_at_full_success"], 1.)
                if head == "direct":
                    self.assertGreater(row["clipped_entries"], 0)
                self.assertEqual(row["per_scene"][0]["cfm_completed"], True)


if __name__ == "__main__":
    unittest.main()
