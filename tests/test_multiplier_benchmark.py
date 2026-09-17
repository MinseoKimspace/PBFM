"""Fixed-budget semantics, physical reference accuracy and report integration."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.multiplier_flow.benchmark import (reference_errors, timing_summary,
                                            validate_budgets, work_summary)
from src.multiplier_flow.evaluation import evaluate
from src.multiplier_flow.model import ConditionalField
from src.multiplier_flow.problem import make_problem, pack
from src.multiplier_flow.solvers import pgs, run_cfm
from test_multiplier_flow import scalar_problem, tiny_config


class FixedBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_pgs_is_sequential_and_ignores_early_tolerance(self):
        # D=[[1,.5],[.5,1]], c=[-1,-1]. Exact solution is [2/3,2/3].
        problem = pack([make_problem(torch.zeros(2, dtype=torch.float64),
            torch.tensor([[1., 0.], [.5, math.sqrt(.75)]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64), -torch.ones(2, dtype=torch.float64))])
        start = torch.zeros_like(problem["c"])
        expected = {1: [1., .5], 2: [.75, .625], 3: [.6875, .65625]}
        for budget, coordinates in expected.items():
            # A loose tolerance would stop immediately with the old default.
            result = pgs(problem, start, tolerance=100., max_sweeps=budget,
                         stop_at_tolerance=False)
            torch.testing.assert_close(result["final"], start.new_tensor([coordinates]))
            self.assertEqual(result["sweeps"].tolist(), [budget])
            self.assertEqual(result["contact_evals"].tolist(), [2 * budget])
        torch.testing.assert_close(start, torch.zeros_like(start))
        stopped = pgs(problem, start, tolerance=100., max_sweeps=3)
        self.assertEqual(stopped["sweeps"].tolist(), [0])

    def test_solved_and_empty_worlds_preserve_exact_fixed_budget_and_masked_work(self):
        empty = make_problem(torch.zeros(1, dtype=torch.float64),
            torch.zeros(0, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.empty(0, dtype=torch.float64))
        scalar = make_problem(torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.tensor([-.1], dtype=torch.float64))
        problem = pack([scalar, empty])
        solved = torch.tensor([[.1], [0.]], dtype=torch.float64)
        fixed = pgs(problem, solved, max_sweeps=16, stop_at_tolerance=False)
        torch.testing.assert_close(fixed["final"], solved)
        self.assertEqual(fixed["sweeps"].tolist(), [16, 16])
        self.assertEqual(fixed["contact_evals"].tolist(), [16, 0])
        self.assertEqual(pgs(problem, solved)["sweeps"].tolist(), [0, 0])
        cost = work_summary(fixed, problem)
        self.assertEqual(cost["logical_totals"]["contact_evals"], 16)
        self.assertEqual(cost["padded_pgs_coordinate_visits"], 32)

    def test_reference_accuracy_uses_physical_positions_and_original_precision(self):
        redundant = pack([make_problem(torch.zeros(1, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.full((2,), -.1, dtype=torch.float64))])
        prediction = torch.tensor([[.1, 0.]], dtype=torch.float64)
        target = prediction.flip(-1)
        errors = reference_errors(redundant, prediction, target, .1)
        self.assertEqual(float(errors["position_rmse"]), 0.)
        self.assertAlmostEqual(float(errors["multiplier_rmse"]), .1)
        self.assertAlmostEqual(float(errors["objective_gap"]), 0.)

        scalar = scalar_problem()
        prediction32 = torch.tensor([[.1]], dtype=torch.float32)
        target64 = torch.tensor([[.1 + 1e-10]], dtype=torch.float64)
        errors = reference_errors(scalar, prediction32, target64, .1)
        expected = abs(float(prediction32) - float(target64))
        self.assertGreater(expected, 0.)
        self.assertAlmostEqual(float(errors["position_rmse"]), expected, places=17)
        self.assertEqual(errors["position_rmse"].dtype, torch.float64)

    def test_mass_weighting_and_objective_gap(self):
        problem = pack([make_problem(torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64), torch.tensor([1., .25], dtype=torch.float64),
            -torch.ones(2, dtype=torch.float64))])
        target = torch.tensor([[1., 4.]], dtype=torch.float64)
        errors = reference_errors(problem, torch.tensor([[0., 4.]]), target, .1)
        self.assertAlmostEqual(float(errors["position_rmse"]), math.sqrt(1 / 5))
        self.assertAlmostEqual(float(errors["position_normalized_rmse"]), math.sqrt(1 / 5) / .1)
        self.assertAlmostEqual(float(errors["objective_gap"]), .5)

    def test_timing_and_attention_work_do_not_conflate_batch_and_world_counts(self):
        timing = timing_summary([.01, .03, .02], 10)
        self.assertAlmostEqual(timing["median_seconds"], .02)
        self.assertAlmostEqual(timing["amortized_median_seconds_per_qp"], .002)
        self.assertAlmostEqual(timing["qps_per_second"], 500.)
        problem = scalar_problem()
        problem = {key: tensor.repeat(2, *([1] * (tensor.ndim - 1))) for key, tensor in problem.items()}
        model = ConditionalField(hidden_dim=8, message_steps=1, communication="global",
                                 attention_heads=2, attention_layers=2).double()
        result = run_cfm(model, problem, torch.zeros_like(problem["c"]), 16)
        cost = work_summary(result, problem, model)
        self.assertEqual(cost["logical_totals"]["nfe"], 32)
        self.assertEqual(cost["executed_batch_neural_calls"], 16)
        self.assertEqual(cost["padded_neural_token_evaluations"], 32)
        self.assertEqual(cost["dense_attention_score_entries"], 16 * 2 * 1 * 1 * 2 * 2)

    def test_reject_invalid_budgets_and_accuracy_settings(self):
        settings = tiny_config()["evaluation"]
        for budgets in ([], [0], [True], [1, 1], [1.5]):
            with self.assertRaises(ValueError):
                validate_budgets(dict(settings, calls=budgets))
        for key, number in (("position_tolerance", 0), ("tolerance", float("nan")),
                            ("timing_repeats", 0)):
            with self.assertRaises(ValueError):
                validate_budgets(dict(settings, **{key: number}))

    def test_end_to_end_report_compares_identical_budgets_and_keeps_reference_separate(self):
        config = tiny_config()
        config["evaluation"].update(calls=[1, 2, 16], start_modes=["zero"], guarded=False,
                                     position_tolerance=1e-12, timing_repeats=2)
        problem = scalar_problem()
        split = dict(problem=problem, optimum=torch.full_like(problem["c"], .1), names=["floor_0"])
        for head in ("analytic", "direct"):
            model = ConditionalField(hidden_dim=8, message_steps=1, head_type=head)
            before = copy.deepcopy(config)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "evaluation.json"
                report = evaluate(model, split, config, torch.device("cpu"), path)
                disk = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(disk["fixed_budget"]["budgets"], [1, 2, 16])
                self.assertIn("pgs_k16_fixed", path.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertEqual(config, before)
            rows = report["modes"]["zero"]
            self.assertEqual(rows["pgs"]["sweeps"], 1)
            for budget in (1, 2, 16):
                fixed, cfm = rows[f"pgs_k{budget}_fixed"], rows[f"cfm_k{budget}_raw"]
                self.assertEqual(fixed["sweeps"], budget)
                self.assertEqual(fixed["per_scene"][0]["contact_evals"], budget)
                self.assertEqual(cfm["nfe"], budget)
                self.assertEqual(len(cfm["timing"]["batch_seconds"]), 2)
                self.assertIn("accuracy", fixed["groups"]["floor"])
                self.assertEqual(fixed["accuracy"]["success_rate"], 1.)
                # FP32 .1 differs from the original FP64 reference by >1e-12.
                self.assertEqual(fixed["accuracy"]["position_within_tolerance_rate"], 0.)
                comparison = report["fixed_budget_comparison"]["zero"][str(budget)]
                self.assertEqual(comparison["pgs_method"], f"pgs_k{budget}_fixed")
                self.assertEqual(comparison["pgs_only_success"], int(head == "direct"))


if __name__ == "__main__":
    unittest.main()
