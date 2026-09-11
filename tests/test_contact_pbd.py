import unittest

import torch

from src.contact_flow.pbd import integrate_pbd
from src.contact_flow.physics import PhysicsConfig, geometry


class NativePBDTests(unittest.TestCase):
    def setUp(self):
        self.physics = PhysicsConfig()

    def test_single_floor_full_correction_in_one_native_sweep(self):
        z = torch.tensor([[[0., .2]]], dtype=torch.float64)
        radius = torch.tensor([[.5]], dtype=torch.float64)
        result = integrate_pbd(z, radius, self.physics, max_sweeps=1, tolerance=1e-12, record=True)
        self.assertTrue(result["converged"].all())
        self.assertTrue(result["completed"].all())
        self.assertEqual(int(result["sweeps"][0]), 1)
        self.assertEqual(int(result["contact_evals"][0]), 3)
        self.assertEqual(int(result["contact_updates"][0]), 1)
        self.assertEqual(int(result["nfe"][0]), 0)
        self.assertEqual(float(result["time"][0]), 0)
        self.assertAlmostEqual(float(result["final"][0, 0, 1]), .495)
        torch.testing.assert_close(z + result["history"]["u"].sum(0), result["final"])

    def test_pair_mass_weighting_and_center_of_mass(self):
        z = torch.tensor([[[0., 4.], [.8, 4.]]], dtype=torch.float64)
        radius = torch.tensor([[.3, .7]], dtype=torch.float64)
        result = integrate_pbd(z, radius, self.physics, max_sweeps=1, tolerance=1e-12)
        self.assertTrue(result["converged"].all())
        correction = result["final"] - z
        masses = radius.square()  # Common pi*density cancels.
        torch.testing.assert_close((correction * masses[..., None]).sum(1), torch.zeros_like(z[:, 0]), atol=1e-12, rtol=0)
        ratio = abs(float(correction[0, 0, 0] / correction[0, 1, 0]))
        self.assertAlmostEqual(ratio, .7 ** 2 / .3 ** 2)
        changed_density = integrate_pbd(z, radius, PhysicsConfig(density=31), max_sweeps=1, tolerance=1e-12)
        torch.testing.assert_close(result["final"], changed_density["final"])

    def test_three_body_stack_converges_and_stops_worlds_independently(self):
        stack = torch.tensor([[[0., .4], [0., 1.3], [0., 2.2]],
                              [[-3., 3.], [0., 3.], [3., 3.]]], dtype=torch.float64)
        radius = torch.full((2, 3), .5, dtype=torch.float64)
        result = integrate_pbd(stack, radius, self.physics, max_sweeps=200, tolerance=1e-5, record=True)
        self.assertTrue(result["converged"].all(), result["failure_reason"])
        self.assertGreater(int(result["sweeps"][0]), 1)
        self.assertEqual(int(result["sweeps"][1]), 0)
        self.assertEqual(int(result["contact_evals"][1]), 0)
        self.assertEqual(int(result["first_tolerance_sweep"][1]), 0)
        self.assertLessEqual(float(geometry(result["final"], radius, self.physics)["max_violation"].max()), 1e-5)

    def test_finite_budget_exhaustion_is_completed_but_not_converged(self):
        z = torch.tensor([[[0., .4], [0., 1.3], [0., 2.2]]], dtype=torch.float64)
        radius = torch.full((1, 3), .5, dtype=torch.float64)
        result = integrate_pbd(z, radius, self.physics, max_sweeps=1, tolerance=1e-6)
        self.assertFalse(result["failed"].any())
        self.assertTrue(result["completed"].all())
        self.assertFalse(result["converged"].any())
        self.assertTrue(result["budget_exhausted"].all())
        self.assertEqual(result["failure_reason"], ["none"])
        self.assertEqual(result["termination_reason"], ["max_sweeps"])
        self.assertEqual(int(result["first_tolerance_sweep"][0]), -1)
        for kwargs in (dict(max_sweeps=0), dict(max_sweeps=1.5), dict(tolerance=float("nan"))):
            with self.assertRaises(ValueError):
                integrate_pbd(z, radius, self.physics, **kwargs)

    def test_degenerate_configuration_after_wall_sweep_is_explicit_failure(self):
        # Both bodies start well outside the wall; its sequential corrections
        # would put their centers at exactly the same point. Do not invent a
        # normal or return that invalid state as a successful finite budget.
        z = torch.tensor([[[-12., 3.], [-11.5, 3.]],
                          [[-3., 3.], [3., 3.]]], dtype=torch.float64)
        radius = torch.full((2, 2), .5, dtype=torch.float64)
        result = integrate_pbd(z, radius, self.physics, max_sweeps=4)
        self.assertEqual(result["failed"].tolist(), [True, False])
        self.assertEqual(result["completed"].tolist(), [False, True])
        self.assertEqual(result["failure_reason"][0], "invalid_contact_configuration")
        torch.testing.assert_close(result["final"], z)


if __name__ == "__main__":
    unittest.main()
