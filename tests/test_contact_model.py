import unittest

import torch

from src.contact_flow.dynamics import free_position, finite_difference_state, make_projection_condition
from src.contact_flow.model import ContactFlowNet
from src.contact_flow.physics import PhysicsConfig, FlowConfig, geometry, cap_mobility, contact_velocity
from src.contact_flow.solver import SolverConfig, integrate


class ModelTests(unittest.TestCase):
    def test_neural_overflow_is_reported_by_integrator(self):
        z = torch.tensor([[[0., .4]]])
        radius = torch.tensor([[.5]])
        state = torch.cat([z, torch.zeros_like(z)], -1)
        model = ContactFlowNet(hidden_dim=8, time_dim=4, message_steps=1, rank=1)
        with torch.no_grad():
            model.head[-1].bias.fill_(float("inf"))
        condition = make_projection_condition(state, z)
        physics = PhysicsConfig()
        result = integrate(lambda q, t: model(q, t, radius, condition, physics),
                           z, radius, physics, SolverConfig())
        self.assertEqual(result["failure_reason"], ["nonfinite_field"])
        self.assertTrue(torch.equal(result["final"], z))

    def test_descent_backward_and_eval_no_grad(self):
        torch.manual_seed(4)
        model = ContactFlowNet(hidden_dim=16, time_dim=8, message_steps=2, rank=2)
        z = torch.tensor([[[0., .4], [.8, .5], [3., 3.]]])
        radius = torch.full((1, 3), .5)
        state = torch.cat([z, torch.zeros_like(z)], -1)
        condition = make_projection_condition(state, z)
        physics = PhysicsConfig()
        velocity = model(z, torch.tensor([[.4]]), radius, condition, physics)
        geo = geometry(z, radius, physics)
        self.assertLessEqual((velocity.flatten(1) * geo["gradient"]).sum().item(), 1e-7)
        velocity.square().sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(z, torch.zeros(1), radius, condition, physics)).all())

    def test_feasible_identity_and_fd(self):
        z = torch.tensor([[[0., 2.], [2., 2.]]])
        r = torch.full((1, 2), .5)
        state = torch.cat([z, torch.zeros_like(z)], -1)
        proposal = free_position(state, .01, -9.8)
        model = ContactFlowNet(hidden_dim=8, time_dim=4, message_steps=1, rank=1)
        u = model(proposal, torch.zeros(1), r, make_projection_condition(state, proposal), PhysicsConfig())
        self.assertTrue(torch.equal(u, torch.zeros_like(u)))
        result = finite_difference_state(state, proposal, .01)
        torch.testing.assert_close(result[..., 3], torch.full((1, 2), -.098), atol=1e-5, rtol=1e-4)

    def test_particle_permutation_does_not_change_contact_field(self):
        torch.manual_seed(7)
        model = ContactFlowNet(hidden_dim=16, time_dim=8, message_steps=2, rank=2,
                               flow=FlowConfig(mobility_bound=.1)).eval()
        # Unequal radii, coupled pairs and a ground contact exercise both pair
        # orientations and the non-paired external-constraint branch.
        z = torch.tensor([[[-.4, .45], [.4, .5], [1.42, .5], [-9.6, 2.0]]])
        radius = torch.tensor([[.5, .55, .5, .5]])
        condition = torch.cat([z, z, torch.zeros_like(z)], -1)
        permutation = torch.tensor([3, 2, 0, 1])
        tau = torch.tensor([.3])
        physics = PhysicsConfig()
        with torch.no_grad():
            velocity = model(z, tau, radius, condition, physics)
            permuted = model(z[:, permutation], tau, radius[:, permutation],
                             condition[:, permutation], physics)
        torch.testing.assert_close(permuted, velocity[:, permutation], atol=1e-7, rtol=1e-5)

    def test_disconnected_contact_does_not_change_another_component(self):
        torch.manual_seed(7)
        # The low bound deliberately activates the spectral cap.
        model = ContactFlowNet(hidden_dim=16, time_dim=8, message_steps=2, rank=2,
                               flow=FlowConfig(mobility_bound=.1)).eval()
        pair = torch.tensor([[[-.4, 2.0], [.4, 2.0]]])
        extra = torch.tensor([[[-.4, 5.0], [.4, 5.0]]])
        combined = torch.cat([pair, extra], 1)
        physics, tau = PhysicsConfig(), torch.tensor([.3])

        def run(z):
            radius = torch.full(z.shape[:2], .5)
            condition = torch.cat([z, z, torch.zeros_like(z)], -1)
            return model(z, tau, radius, condition, physics)

        with torch.no_grad():
            isolated = run(pair)
            together = run(combined)
        self.assertGreater(isolated.abs().max().item(), 1e-4)
        torch.testing.assert_close(together[:, :2], isolated, atol=1e-7, rtol=1e-5)

    def test_psd_bound_is_per_component(self):
        torch.manual_seed(7)
        model = ContactFlowNet(hidden_dim=16, time_dim=8, message_steps=2, rank=2,
                               flow=FlowConfig(mobility_bound=.1)).eval()
        z = torch.tensor([[[-.4, 2.0], [.4, 2.0], [-.4, 5.0], [.4, 5.0]]])
        radius = torch.full(z.shape[:2], .5)
        condition = torch.cat([z, z, torch.zeros_like(z)], -1)
        physics = PhysicsConfig()
        geo = geometry(z, radius, physics)
        diagonal, low_rank = model.mobility(z, torch.tensor([.3]), radius, condition, geo, physics)
        diagonal, low_rank = cap_mobility(geo, diagonal, low_rank, model.flow.mobility_bound)
        for component in (0, 2):
            mask = geo["component"][0] == component
            factor = low_rank[0, mask]
            matrix = torch.diag(diagonal[0, mask]) + factor @ factor.T
            eigenvalues = torch.linalg.eigvalsh(matrix)
            self.assertGreaterEqual(eigenvalues.min().item(), -1e-7)
            self.assertLessEqual(eigenvalues.max().item(), .100001)

    def test_rank_zero_and_shared_flow_gain(self):
        torch.manual_seed(9)
        flow = FlowConfig(gain=4, normalization="diagonal")
        model = ContactFlowNet(hidden_dim=8, time_dim=4, message_steps=1, rank=0, flow=flow)
        z = torch.tensor([[[0., .4], [.8, .5]]])
        radius = torch.full((1, 2), .5)
        condition = torch.cat([z, z, torch.zeros_like(z)], -1)
        physics, tau = PhysicsConfig(), torch.tensor([.2])
        geo = geometry(z, radius, physics)
        diagonal, low_rank = model.mobility(z, tau, radius, condition, geo, physics)
        self.assertEqual(low_rank.shape[-1], 0)
        output = model(z, tau, radius, condition, physics)
        torch.testing.assert_close(output, contact_velocity(geo, diagonal, low_rank, flow))
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
