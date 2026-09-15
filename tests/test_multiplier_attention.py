"""Information flow and physical-head invariants for the global CFM model."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.multiplier_flow.model import (CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT,
                                       ConditionalField, load_model)
from src.multiplier_flow.problem import contact_endpoint, make_problem, pack


def chain_problem(contacts, stiffness=1.0):
    """Floor contact followed by a one-dimensional chain of coupled contacts."""
    jacobian = torch.eye(contacts, dtype=torch.float64)
    for index in range(1, contacts):
        jacobian[index, index - 1] = -1
    return make_problem(torch.zeros(contacts, dtype=torch.float64), jacobian,
                        torch.full((contacts,), stiffness, dtype=torch.float64),
                        torch.full((contacts,), -.1, dtype=torch.float64))


def independent_union(first, second):
    return make_problem(torch.cat((first["p"], second["p"])),
                        torch.block_diag(first["J"], second["J"]),
                        torch.cat((first["w"], second["w"])),
                        torch.cat((first["c"], second["c"])))


def initialized_model(communication="global", feature_version="residual"):
    torch.manual_seed(14)
    model = ConditionalField(hidden_dim=16, message_steps=1,
                             communication=communication, attention_heads=4,
                             attention_layers=2, feature_version=feature_version).double()
    # A zero head is deliberately the local-projection initialization. Nonzero
    # head weights are needed to inspect information reaching the learned head.
    with torch.no_grad():
        model.head.weight.normal_(std=.05)
    return model


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_contact_permutation_equivariance(self):
        single = chain_problem(6)
        permutation = torch.tensor([5, 1, 3, 0, 4, 2])
        shuffled = dict(single, J=single["J"][permutation], c=single["c"][permutation])
        problem, permuted = pack([single]), pack([shuffled])
        state = torch.linspace(.01, .15, 6, dtype=torch.float64)[None]
        tau = torch.tensor([.3], dtype=torch.float64)
        model = initialized_model()
        torch.testing.assert_close(model(state, tau, problem)[:, permutation],
                                   model(state[:, permutation], tau, permuted))

    def test_padding_empty_rows_and_batch_independence(self):
        small, large = chain_problem(3), chain_problem(8)
        empty = make_problem(torch.zeros(2, dtype=torch.float64),
                             torch.zeros(0, 2, dtype=torch.float64),
                             torch.ones(2, dtype=torch.float64),
                             torch.empty(0, dtype=torch.float64))
        model = initialized_model()
        problem = pack([small, large, empty])
        state = .03 * problem["mask"].to(torch.float64)
        tau = torch.full((3,), .4, dtype=torch.float64)
        output = model(state, tau, problem)
        expected = model(state[:1, :3], tau[:1], pack([small]))
        torch.testing.assert_close(output[:1, :3], expected)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output[~problem["mask"]] == 0).all())
        output.square().sum().backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        all_empty = pack([empty, empty])
        model.zero_grad(set_to_none=True)
        output = model(torch.zeros_like(all_empty["c"]), tau[:2], all_empty)
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.sum().backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_independent_component_cannot_change_normalization_or_prediction(self):
        first, second = chain_problem(4), chain_problem(3, stiffness=1000.)
        second["c"].fill_(-100.)
        joined = independent_union(first, second)
        alone, together = pack([first]), pack([joined])
        # The global spectral eta changes, but feature normalization must not.
        self.assertLess(float(together["eta"]), float(alone["eta"]) / 100)
        state = torch.full_like(alone["c"], .05)
        joined_state = torch.cat((state, torch.full((1, 3), 50., dtype=state.dtype)), 1)
        tau = torch.tensor([.6], dtype=torch.float64)
        for communication in ("local", "global"):
            model = initialized_model(communication)
            torch.testing.assert_close(model(state, tau, alone),
                                       model(joined_state, tau, together)[:, :4])

    def test_distant_condition_reaches_first_call_only_with_global_attention(self):
        first = chain_problem(10)
        changed = dict(first, c=first["c"].clone())
        changed["c"][-1] = -.7
        problem, remote_change = pack([first]), pack([changed])
        self.assertEqual(float(problem["D"][0, 0, -1]), 0.)
        self.assertTrue(problem["reach"][0, 0, -1])
        state = torch.full_like(problem["c"], .1)
        tau = torch.tensor([.2], dtype=torch.float64)
        local = initialized_model("local")
        torch.testing.assert_close(local(state, tau, problem)[:, 0],
                                   local(state, tau, remote_change)[:, 0], atol=0, rtol=0)
        global_model = initialized_model()
        difference = (global_model(state, tau, problem)[:, 0]
                      - global_model(state, tau, remote_change)[:, 0]).abs()
        self.assertGreater(float(difference.detach()), 1e-8)

    def test_current_state_and_residual_are_read_on_every_call(self):
        model = initialized_model()
        problem = pack([chain_problem(4)])
        tau = torch.tensor([.3], dtype=torch.float64)
        state = torch.full_like(problem["c"], .03)
        changed = state.clone()
        changed[:, -1] += .1
        seen = []
        handle = model.encoder.register_forward_pre_hook(
            lambda module, arguments: seen.append(arguments[0].detach().clone()))
        model(state, tau, problem)
        model(changed, tau, problem)
        handle.remove()
        self.assertEqual(seen[0].shape[-1], 6)
        diagonal = problem["D"].diagonal(dim1=1, dim2=2)
        scale = model.length_scale / diagonal
        torch.testing.assert_close(seen[1][..., -1],
                                   (changed - contact_endpoint(problem, changed)) / scale)
        self.assertFalse(torch.equal(seen[0][..., -1], seen[1][..., -1]))

    def test_head_keeps_scalar_projection_and_allows_joint_correction(self):
        problem = pack([chain_problem(3)])
        state = torch.full_like(problem["c"], .1)
        tau = torch.tensor([.4], dtype=torch.float64)
        for communication in ("local", "global"):
            model = ConditionalField(hidden_dim=16, message_steps=1,
                                     communication=communication,
                                     feature_version="residual").double()
            torch.testing.assert_close(model.endpoint(state, tau, problem),
                                       contact_endpoint(problem, state))
            scalar = pack([chain_problem(1)])
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.uniform_(-2, 2)
            torch.testing.assert_close(model.endpoint(torch.full_like(scalar["c"], .3), tau, scalar),
                                       torch.full_like(scalar["c"], .1))

        model = initialized_model()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(1.)
        problem["c"][:] = torch.tensor([[0., -.2, 0.]], dtype=torch.float64)
        # Contacts with zero individual residual still receive a rate because
        # another contact in their component needs correction.
        rate = model.coupling_rate(torch.zeros_like(state), tau, problem)
        self.assertTrue((rate > 0).all())

    def test_attention_gradient_matches_finite_difference_through_analytic_head(self):
        problem = pack([chain_problem(5)])
        state = torch.full_like(problem["c"], .15)
        tau = torch.tensor([.4], dtype=torch.float64)
        model = initialized_model()

        def loss():
            return model.endpoint(state, tau, problem).square().sum()

        loss().backward()
        parameter = model.attention[0].qkv.weight
        flat_index = int(parameter.grad.abs().argmax())
        row, column = divmod(flat_index, parameter.shape[1])
        analytic = float(parameter.grad[row, column])
        self.assertGreater(abs(analytic), 1e-8)
        original = float(parameter[row, column].detach())
        values = []
        for shift in (-1e-5, 1e-5):
            with torch.no_grad():
                parameter[row, column] = original + shift
                values.append(float(loss()))
        with torch.no_grad():
            parameter[row, column] = original
        self.assertAlmostEqual(analytic, (values[1] - values[0]) / 2e-5, delta=1e-8)
        self.assertGreater(float(model.attention[0].relation_weight.grad.abs().sum()), 0.)

    def test_legacy_v2_loading_and_explicit_incompatibility(self):
        settings = dict(hidden_dim=10, message_steps=1, length_scale=.1)
        config = dict(model=settings, physics={}, dynamics={}, reference={}, data={}, seed=42)
        legacy = ConditionalField(**settings)
        self.assertEqual(legacy.encoder[0].in_features, 5)
        # No head divisibility restriction applies to the original local model.
        self.assertEqual(len(legacy.attention), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            checkpoint = dict(format=LEGACY_CHECKPOINT_FORMAT, objective="cfm", config=config,
                              model=legacy.state_dict())
            torch.save(checkpoint, path)
            loaded, metadata = load_model(path, config, "cpu")
            self.assertEqual(metadata["format"], LEGACY_CHECKPOINT_FORMAT)
            self.assertEqual(loaded.checkpoint_format, LEGACY_CHECKPOINT_FORMAT)
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
            changed = copy.deepcopy(config)
            changed["model"]["feature_version"] = "residual"
            checkpoint["config"] = changed
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "v2 weights require"):
                load_model(path, changed, "cpu")
            checkpoint["format"] = CHECKPOINT_FORMAT
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "weights do not match"):
                load_model(path, changed, "cpu")


if __name__ == "__main__":
    unittest.main()
