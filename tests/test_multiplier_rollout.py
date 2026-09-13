import tempfile
import unittest
from pathlib import Path

import torch

from test_multiplier_flow import EndpointFunction, tiny_config
from src.contact_flow.dynamics import free_position, finite_difference_state
from src.contact_flow.physics import PhysicsConfig
from src.multiplier_flow.model import ConditionalField
from src.multiplier_flow.rollout import evaluate_motion, motion_scenes, simulate, swept_pairs


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


if __name__ == "__main__":
    unittest.main()
