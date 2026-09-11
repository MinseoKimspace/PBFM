from __future__ import annotations

import json
import unittest

import torch

from src.contact_flow.paths import path_summary


class PathDiagnosticsTests(unittest.TestCase):
    def buffer(self):
        terminal = torch.tensor([[0., .02, .03], [.02, .03, .04],
                                 [0., 0., .005], [0., 0., 0.]], dtype=torch.float64)
        initial = torch.tensor([.1, .2, .01, 0.], dtype=torch.float64)
        success = terminal <= .001
        return {
            "format": "contact_paths_v2",
            "path_scores": torch.tensor([[3., 0., 1.], [1., 2., 3.],
                                         [0., .1, .2], [0., 0., 0.]]),
            "scene_type": ["vertical_stack", "vertical_stack", "free_flight", "free_flight"],
            "metadata": {"solver": {"tolerance": .001}},
            "diagnostics": {
                "terminal_violation": terminal,
                "terminal_energy": .5 * terminal.square(),
                "initial_violation": initial,
                "initial_energy": .5 * initial.square(),
                "proposal_violation": torch.tensor([.1, .2, 0., 0.]),
                "completed": torch.ones(4, 3, dtype=torch.bool),
                "failed": torch.zeros(4, 3, dtype=torch.bool),
                "nfe": torch.arange(1, 13).reshape(4, 3),
                "backtracks": torch.ones(4, 3, dtype=torch.long),
                "energy_evals": torch.full((4, 3), 4),
                "generation_seconds": 2.5,
                "field_calls": 4,
                "executed_field_rows": 48,
                "first_tolerance_time": torch.where(success, .7, -1.),
                "first_tolerance_nfe": torch.where(success, 2, -1),
            },
        }

    def test_scene_coverage_separates_solved_starts_and_failures(self):
        result = path_summary(self.buffer(), 1, "uniform")
        groups = result["groups"]
        self.assertEqual(groups["all"]["paths"], 12)
        self.assertEqual(groups["all"]["path_success_rate"], .5)
        self.assertEqual(groups["all"]["scene_success_rate"], .75)
        self.assertEqual(groups["all"]["weighted_success_mass"]["mean"], .5)
        self.assertEqual(groups["all"]["best_score_misses_success_count"], 1)
        self.assertEqual(groups["clean_solved"]["scenes"], 2)
        self.assertEqual(groups["start_solved"]["scenes"], 1)
        self.assertEqual(groups["scene:vertical_stack"]["scene_success_rate"], .5)
        self.assertEqual(groups["scene:free_flight"]["scene_success_rate"], 1.)
        json.dumps(result, allow_nan=False)

    def test_score_weighted_success_mass_is_not_scene_any_success(self):
        result = path_summary(self.buffer(), .00001, "score")
        stack = result["groups"]["scene:vertical_stack"]
        self.assertEqual(stack["scene_success_rate"], .5)
        self.assertEqual(stack["weighted_success_mass"]["mean"], 0.)
        self.assertAlmostEqual(stack["ess"]["mean"], 1.)

    def test_oracle_charges_every_candidate_not_just_selected_paths(self):
        result = path_summary(self.buffer(), 1)
        oracle = result["oracle"]
        self.assertEqual(oracle["candidate_index"], [0, 0, 0, 0])
        self.assertEqual(oracle["successful_scenes"], 3)
        self.assertEqual(oracle["charged_cost"]["logical_nfe"], 78)
        self.assertEqual(oracle["charged_cost"]["generation_seconds"], 2.5)
        self.assertEqual(oracle["charged_cost"]["executed_field_rows"], 48)
        self.assertFalse(oracle["comparable_to_clean_proposal_rollout"])
        self.assertIn("same noisy start", oracle["start"])

    def test_empty_groups_and_unavailable_time_stay_valid_json(self):
        buffer = self.buffer()
        buffer["diagnostics"]["proposal_violation"].fill_(.1)
        buffer["diagnostics"]["first_tolerance_time"].fill_(-1)
        result = path_summary(buffer, 1, scene_types=["external"] * 4)
        self.assertEqual(result["groups"]["clean_solved"]["scenes"], 0)
        self.assertIsNone(result["groups"]["clean_solved"]["path_success_rate"])
        self.assertIsNone(result["groups"]["all"]["first_tolerance_time"]["mean"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            path_summary(buffer, 1, scene_types=["bad"])

    def test_failed_candidates_cannot_be_oracle_successes(self):
        buffer = self.buffer()
        buffer["diagnostics"]["failed"][0, 0] = True
        buffer["diagnostics"]["completed"][0, 0] = False
        result = path_summary(buffer, 1)
        self.assertEqual(result["oracle"]["candidate_index"][0], 1)
        self.assertEqual(result["oracle"]["successful_scenes"], 2)
        self.assertEqual(result["groups"]["all"]["failed_paths"], 1)
        self.assertEqual(result["groups"]["all"]["incomplete_paths"], 1)


if __name__ == "__main__":
    unittest.main()
