import math
import unittest

import torch

from src.contact_flow.physics import (FlowConfig, PhysicsConfig, apply_mobility,
                                      cap_mobility, contact_velocity, energy,
                                      geometry, make_baseline, mobility_diagonal_scale)
from src.contact_flow.solver import SolverConfig, integrate


class ContactPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PhysicsConfig()
        self.radius = torch.tensor([[0.5, 0.7]], dtype=torch.float64)
        self.z = torch.tensor([[[0.0, 0.45], [1.0, 0.65]]], dtype=torch.float64)

    def test_analytic_gradient_matches_autograd_and_slot_order(self):
        z = self.z.clone().requires_grad_(True)
        geo = geometry(z, self.radius, self.cfg)
        gradient, = torch.autograd.grad(geo["energy"].sum(), z)
        torch.testing.assert_close(geo["gradient"], gradient.flatten(1))
        self.assertEqual(geo["J"].shape, (1, 7, 4))
        self.assertEqual(geo["edge_i"].tolist(), [0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(geo["edge_j"].tolist(), [1, -1, -1, -1, -1, -1, -1])
        torch.testing.assert_close(geo["inverse_mass"], 1 / (math.pi * self.radius.square()))

    def test_psd_mobility_descends_and_remains_differentiable(self):
        torch.manual_seed(7)
        geo = geometry(self.z, self.radius, self.cfg)
        diagonal = torch.rand_like(geo["gap"], requires_grad=True)
        factor = torch.randn(*geo["gap"].shape, 3, dtype=self.z.dtype, requires_grad=True)
        u = apply_mobility(geo, diagonal, factor)
        self.assertLess(float((u.flatten(1) * geo["gradient"]).sum()), 0)
        u.square().sum().backward()
        self.assertTrue(torch.isfinite(diagonal.grad).all())
        self.assertTrue(torch.isfinite(factor.grad).all())

    def test_pair_only_field_preserves_mass_center(self):
        z = self.z + torch.tensor([0.0, 3.0], dtype=self.z.dtype)
        geo = geometry(z, self.radius, self.cfg)
        u = apply_mobility(geo, torch.ones_like(geo["gap"]), z.new_ones(1, 7, 2))
        total = (u / geo["inverse_mass"].unsqueeze(-1)).sum(dim=1)
        torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-12, rtol=0)

    def test_shared_ground_does_not_connect_independent_bodies(self):
        radius = torch.full((1, 2), .5, dtype=torch.float64)
        z = torch.tensor([[[-3., .45], [3., .45]]], dtype=torch.float64)
        geo = geometry(z, radius, self.cfg)
        self.assertNotEqual(int(geo["component"][0, 1]), int(geo["component"][0, 2]))
        factor = z.new_ones(1, 7, 1)
        u = apply_mobility(geo, z.new_zeros(1, 7), factor)
        changed = z.clone()
        changed[0, 0, 1] = .2
        other = apply_mobility(geometry(changed, radius, self.cfg), z.new_zeros(1, 7), factor)
        torch.testing.assert_close(u[:, 1], other[:, 1])

    def test_free_particles_zero_field_and_wall_slop(self):
        radius = torch.tensor([[.5]], dtype=torch.float64)
        z = torch.tensor([[[0., 3.]]], dtype=torch.float64)
        for method in ("gradient", "diagonal", "inverse", "jacobi"):
            torch.testing.assert_close(make_baseline(method, radius, self.cfg)(z, z.new_zeros(1)), torch.zeros_like(z))
        for sign in (-1., 1.):
            wall = torch.tensor([[[sign * 9.6, 3.]]], dtype=torch.float64)
            geo = geometry(wall, radius, self.cfg)
            self.assertAlmostEqual(float(geo["max_violation"]), .095)
            self.assertAlmostEqual(float(geo["penetration"].max()), .1)
            u = make_baseline("gradient", radius, self.cfg)(wall, wall.new_zeros(1))
            self.assertLess(float(sign * u[0, 0, 0]), 0)

    def test_singular_inverse_is_finite_and_bounded_variant_descends(self):
        # Both opposite side normals of one oversized circle make D singular.
        z = torch.tensor([[[0., 3.]]], dtype=torch.float64)
        radius = torch.tensor([[11.]], dtype=torch.float64)
        geo = geometry(z, radius, self.cfg)
        for bound in (None, 16.0):
            u = make_baseline("inverse", radius, self.cfg, mobility_bound=bound)(z, z.new_zeros(1))
            self.assertTrue(torch.isfinite(u).all())
            self.assertLessEqual(float((u.flatten(1) * geo["gradient"]).sum()), 1e-10)

    def test_diagonal_square_is_not_jacobi(self):
        a = make_baseline("diagonal", self.radius, self.cfg)(self.z, self.z.new_zeros(1))
        b = make_baseline("jacobi", self.radius, self.cfg)(self.z, self.z.new_zeros(1))
        self.assertFalse(torch.allclose(a, b))

    def test_small_inverse_spectral_cap_matches_isotropic_mobility(self):
        bound = 1e-8
        geo = geometry(self.z, self.radius, self.cfg)
        expected = apply_mobility(geo, torch.full_like(geo["gap"], bound), self.z.new_zeros(1, 7, 0))
        actual = make_baseline("inverse", self.radius, self.cfg, mobility_bound=bound)(self.z, self.z.new_zeros(1))
        torch.testing.assert_close(actual, expected, atol=1e-18, rtol=1e-8)

    def test_invalid_radii_and_coincident_centers_are_rejected(self):
        for radius in (-self.radius, self.radius * 0, self.radius * float("nan")):
            with self.assertRaises(ValueError):
                geometry(self.z, radius, self.cfg)
        with self.assertRaisesRegex(ValueError, "coincident"):
            geometry(self.z[:, :1].expand_as(self.z), self.radius, self.cfg)

    def test_normalized_fixed_controls_are_mass_invariant_and_gain_linear(self):
        geo = geometry(self.z, self.radius, self.cfg)
        heavy = geometry(self.z, self.radius, PhysicsConfig(density=17.0))
        diagonal = torch.full_like(geo["gap"], 2.0)
        factor = self.z.new_full((1, 7, 2), .3)
        base = contact_velocity(geo, diagonal, factor, FlowConfig())
        actual = contact_velocity(heavy, diagonal, factor, FlowConfig())
        torch.testing.assert_close(actual, base, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(contact_velocity(geo, diagonal, factor, FlowConfig(gain=6.0)), 6 * base)
        raw = contact_velocity(geo, diagonal, factor, FlowConfig(normalization="none"))
        raw_heavy = contact_velocity(heavy, diagonal, factor, FlowConfig(normalization="none"))
        torch.testing.assert_close(raw_heavy, raw / 17 ** 2)

    def test_normalized_psd_is_differentiable_and_stationary_when_feasible(self):
        geo = geometry(self.z, self.radius, self.cfg)
        diagonal = torch.ones_like(geo["gap"], requires_grad=True)
        factor = torch.full((1, 7, 2), .2, dtype=self.z.dtype, requires_grad=True)
        u = contact_velocity(geo, diagonal, factor, FlowConfig(gain=5))
        self.assertLess(float((u.flatten(1) * geo["gradient"]).sum()), 0)
        u.square().sum().backward()
        self.assertTrue(torch.isfinite(diagonal.grad).all())
        self.assertTrue(torch.isfinite(factor.grad).all())
        free = geometry(torch.tensor([[[0., 4.], [4., 4.]]], dtype=self.z.dtype), self.radius, self.cfg)
        torch.testing.assert_close(contact_velocity(free, diagonal.detach(), factor.detach(), FlowConfig()), torch.zeros_like(u))

    def test_cap_ignores_far_slots_and_independent_components(self):
        z = torch.tensor([[[-3., .4], [3., .4]]], dtype=self.z.dtype)
        radius = torch.full((1, 2), .5, dtype=self.z.dtype)
        geo = geometry(z, radius, self.cfg)
        diagonal = torch.ones_like(geo["gap"])
        factor = z.new_ones(1, 7, 1)
        a, b = cap_mobility(geo, diagonal, factor, 1.0)
        # Each isolated floor contact gets its own cap: (1+1)/2=1.
        torch.testing.assert_close(a[:, 1:3], z.new_full((1, 2), .5))
        changed_d, changed_l = diagonal.clone(), factor.clone()
        changed_d[:, 0] = 1e6  # Far pair must consume no budget.
        changed_l[:, 0] = 1e6
        changed_d[:, 1] = 1e4  # Other body component must not consume ours.
        changed_l[:, 1] = 1e4
        c, d = cap_mobility(geo, changed_d, changed_l, 1.0)
        torch.testing.assert_close(a[:, 2], c[:, 2])
        torch.testing.assert_close(b[:, 2], d[:, 2])
        self.assertEqual(float(c[:, 0]), 0.0)
        # Sum of row-square L plus max diagonal exactly bounds every component.
        self.assertLessEqual(float(c[:, 1] + d[:, 1].square().sum(-1)), 1.0000000001)

    def test_normalized_isotropic_single_contact_has_unit_residual_rate(self):
        z = self.z[:, :1]
        radius = self.radius[:, :1]
        flow = FlowConfig(normalization_eps=0, gain=7.0)
        u = make_baseline("isotropic", radius, self.cfg, flow=flow)(z, z.new_zeros(1))
        self.assertAlmostEqual(float(u[0, 0, 1]), 7 * (.5 - .45 - self.cfg.slop))
        inverse = make_baseline("inverse", radius, self.cfg, flow=flow)(z, z.new_zeros(1))
        torch.testing.assert_close(inverse, u, rtol=1e-7, atol=1e-12)

    def test_normalized_inverse_cap_and_mass_invariance(self):
        flow = FlowConfig(gain=2, mobility_bound=10)
        a = make_baseline("inverse", self.radius, self.cfg, flow=flow)(self.z, self.z.new_zeros(1))
        b = make_baseline("inverse", self.radius, PhysicsConfig(density=23), flow=flow)(self.z, self.z.new_zeros(1))
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        tiny = FlowConfig(mobility_bound=1e-8)
        geo = geometry(self.z, self.radius, self.cfg)
        expected = contact_velocity(geo, torch.ones_like(geo["gap"]), self.z.new_zeros(1, 7, 0), tiny)
        actual = make_baseline("inverse", self.radius, self.cfg, flow=tiny)(self.z, self.z.new_zeros(1))
        torch.testing.assert_close(actual, expected, atol=1e-17, rtol=1e-8)

    def test_flow_configuration_validation(self):
        for kwargs in (dict(gain=0), dict(gain=float("nan")), dict(mobility_bound=-1),
                       dict(normalization="wrong"), dict(normalization_eps=-1)):
            with self.assertRaises(ValueError):
                FlowConfig(**kwargs)


class ContactIntegratorTests(unittest.TestCase):
    def setUp(self):
        self.physics = PhysicsConfig()
        self.radius = torch.tensor([[.5]], dtype=torch.float64)
        self.z = torch.tensor([[[0., .3]]], dtype=torch.float64)

    def test_backtracking_advances_real_time_and_history_quadrature(self):
        radius = torch.tensor([[.5, .5]], dtype=torch.float64)
        initial = torch.tensor([[[0., .3], [0., 1.4]]], dtype=torch.float64)
        def stiff(z, tau):
            return make_baseline("gradient", radius, self.physics)(z, tau) * 5.0
        result = integrate(stiff, initial, radius, self.physics,
                           SolverConfig(steps=1, max_steps=200, max_displacement=.4, tolerance=1e-6),
                           record=True, stop_on_tolerance=False)
        self.assertTrue(result["completed"].all(), result["failure_reason"])
        self.assertGreater(int(result["backtracks"].sum()), 0)
        history = result["history"]
        torch.testing.assert_close(history["dt"].sum(dim=0), result["time"])
        expected = initial + (history["u"] * history["dt"][..., None, None]).sum(dim=0)
        torch.testing.assert_close(result["final"], expected)
        values = torch.cat([energy(x, radius, self.physics) for x in history["z"]] +
                           [energy(result["final"], radius, self.physics)])
        self.assertTrue((values[1:] <= values[:-1]).all())
        self.assertLess(float(history["dt"][0, 0]), 1.0)

    def test_true_stationary_reference_has_actual_zero_velocity(self):
        z = self.z.clone()
        z[..., 1] = 3.0
        result = integrate(make_baseline("gradient", self.radius, self.physics), z, self.radius,
                           self.physics, SolverConfig(steps=4), record=True, stop_on_tolerance=False)
        self.assertTrue(result["completed"].all())
        self.assertEqual(int(result["nfe"][0]), 4)
        self.assertEqual(float(result["time"][0]), 1.0)
        self.assertEqual(float(result["history"]["u"].abs().sum()), 0.0)

    def test_stop_on_tolerance_does_not_evaluate_field(self):
        z = self.z.clone()
        z[..., 1] = .4949
        result = integrate(lambda z, t: torch.full_like(z, float("nan")), z, self.radius,
                           self.physics, SolverConfig(), record=True)
        self.assertTrue(result["completed"].all())
        self.assertEqual(int(result["nfe"][0]), 0)
        self.assertEqual(result["history"]["dt"].shape[0], 0)

    def test_nonfinite_zero_and_uphill_fields_fail_without_fallback(self):
        fields = (lambda z, t: torch.full_like(z, float("nan")),
                  lambda z, t: torch.zeros_like(z),
                  lambda z, t: -make_baseline("gradient", self.radius, self.physics)(z, t))
        for field in fields:
            result = integrate(field, self.z, self.radius, self.physics, SolverConfig(), record=True)
            self.assertTrue(result["failed"].all())
            self.assertFalse(result["completed"].any())
            self.assertEqual(float(result["time"][0]), 0.0)
            self.assertEqual(float(result["history"]["dt"].sum()), 0.0)
            torch.testing.assert_close(result["final"], self.z)

    def test_unrepresentable_update_is_not_counted_as_progress(self):
        tiny = lambda z, t: make_baseline("gradient", self.radius, self.physics)(z, t) * 1e-30
        result = integrate(tiny, self.z, self.radius, self.physics, SolverConfig(), record=True)
        self.assertTrue(result["failed"].all())
        self.assertEqual(float(result["time"][0]), 0.0)

    def test_subresolution_reference_retains_nonzero_tangent_below_tolerance(self):
        for dtype in (torch.float32, torch.float64):
            z = self.z.to(dtype).clone()
            radius = self.radius.to(dtype)
            z[..., 1] = .4949
            baseline = make_baseline("gradient", radius, self.physics)
            result = integrate(lambda q, t: baseline(q, t) * 1e-30, z, radius,
                               self.physics, SolverConfig(steps=4), record=True, stop_on_tolerance=False)
            self.assertTrue(result["completed"].all(), result["failure_reason"])
            self.assertEqual(float(result["time"][0]), 1.0)
            self.assertEqual(int(result["accepted_roundoff"][0]), 4)
            self.assertGreater(float(result["history"]["u"].abs().sum()), 0.0)
            torch.testing.assert_close(result["final"], z, atol=0, rtol=0)

    def test_roundoff_exception_does_not_accept_uphill_or_large_zero_energy_change(self):
        z = self.z.clone()
        z[..., 1] = .4949
        baseline = make_baseline("gradient", self.radius, self.physics)
        uphill = integrate(lambda q, t: -baseline(q, t) * 1e-30, z, self.radius,
                            self.physics, SolverConfig(), stop_on_tolerance=False)
        self.assertTrue(uphill["failed"].all())
        self.assertEqual(int(uphill["accepted_roundoff"][0]), 0)
        # Nearly tangential but macroscopically large x motion cannot qualify
        # as roundoff merely because its tiny y contribution is unresolvable.
        def sideways(q, t):
            u = baseline(q, t) * 1e-30
            u[..., 0] = .1
            return u
        large = integrate(sideways, z, self.radius, self.physics,
                          SolverConfig(max_backtracks=2), stop_on_tolerance=False)
        self.assertTrue(large["failed"].all())
        self.assertEqual(int(large["accepted_roundoff"][0]), 0)

    def test_displacement_cap_and_iteration_budget_are_explicit(self):
        result = integrate(make_baseline("gradient", self.radius, self.physics), self.z, self.radius,
                           self.physics, SolverConfig(steps=1, max_steps=1, max_displacement=.01), record=True)
        self.assertTrue(result["failed"].all())
        self.assertEqual(result["failure_reason"], ["max_steps"])
        self.assertLessEqual(float((result["final"] - self.z).norm()), .0100000001)
        self.assertLess(float(result["time"][0]), 1.0)

    def test_failed_batch_row_does_not_stop_other_worlds(self):
        initial = self.z.expand(2, 1, 2).clone()
        radius = self.radius.expand(2, 1)
        baseline = make_baseline("gradient", radius, self.physics)
        def field(z, tau):
            u = baseline(z, tau)
            u[1] = float("nan")
            return u
        result = integrate(field, initial, radius, self.physics, SolverConfig(steps=4),
                           record=True, stop_on_tolerance=False)
        self.assertEqual(result["completed"].tolist(), [True, False])
        self.assertEqual(result["failed"].tolist(), [False, True])
        self.assertEqual(result["nfe"].tolist(), [4, 1])
        self.assertEqual(result["executed_field_rows"], 8)
        self.assertEqual(float(result["history"]["dt"][:, 1].sum()), 0.0)
        self.assertEqual(float(result["time"][0]), 1.0)

    def test_first_tolerance_cost_is_not_full_path_generation_cost(self):
        flow = FlowConfig(gain=8, normalization_eps=0)
        result = integrate(make_baseline("isotropic", self.radius, self.physics, flow=flow),
                           self.z, self.radius, self.physics,
                           SolverConfig(steps=8, max_steps=64), record=True, stop_on_tolerance=False)
        self.assertTrue(result["completed"].all(), result["failure_reason"])
        self.assertGreater(float(result["first_tolerance_time"][0]), 0)
        self.assertLess(float(result["first_tolerance_time"][0]), 1)
        self.assertLess(int(result["first_tolerance_nfe"][0]), int(result["nfe"][0]))
        self.assertLess(int(result["first_tolerance_energy_evals"][0]), int(result["energy_evals"][0]))
        self.assertEqual(result["history"]["max_violation"].shape, result["history"]["tau"].shape)
        failed = integrate(lambda z, t: torch.zeros_like(z), self.z, self.radius, self.physics, SolverConfig())
        self.assertEqual(float(failed["first_tolerance_time"][0]), -1)
        self.assertEqual(int(failed["first_tolerance_nfe"][0]), -1)
        free = self.z + torch.tensor([0., 3.], dtype=self.z.dtype)
        solved = integrate(lambda z, t: torch.zeros_like(z), free, self.radius, self.physics, SolverConfig())
        self.assertEqual(float(solved["first_tolerance_time"][0]), 0)
        self.assertEqual(int(solved["first_tolerance_nfe"][0]), 0)
        self.assertEqual(int(solved["first_tolerance_energy_evals"][0]), 1)


if __name__ == "__main__":
    unittest.main()
