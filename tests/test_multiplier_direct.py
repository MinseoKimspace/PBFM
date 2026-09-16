"""Direct-field regression checks against explicit small QPs and Euler steps."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_multiplier_flow import release_problem, scalar_problem, tiny_config
from src.multiplier_flow.evaluation import evaluate
from src.multiplier_flow.model import CHECKPOINT_FORMAT, ConditionalField, cfm_loss, load_model
from src.multiplier_flow.problem import converged, make_problem, pack
from src.multiplier_flow.solvers import integrate_cfm, run_cfm
from src.multiplier_flow.training import physical_endpoint_loss, training_objective


def problem_with_gram(matrix, offsets):
    """Construct a valid contact QP with a prescribed positive-definite D."""
    matrix = torch.tensor(matrix, dtype=torch.float64)
    count = len(matrix)
    return pack([make_problem(torch.zeros(count, dtype=torch.float64),
        torch.linalg.cholesky(matrix), torch.ones(count, dtype=torch.float64),
        torch.tensor(offsets, dtype=torch.float64))])


class DirectFunction:
    """An independent field oracle; endpoint use would bypass the direct head."""

    head_type = "direct"
    neural_evaluations = 1

    def __init__(self, function):
        self.function = function

    def __call__(self, state, tau, problem):
        return self.function(state, tau, problem)

    def endpoint(self, *args):
        raise AssertionError("A direct solver must evaluate its field")


class DirectHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wrongly_inactive_contact_has_a_raw_cfm_gradient(self):
        problem = problem_with_gram([[2., -1.], [-1., 2.]], [-1., .2])
        source = torch.zeros_like(problem["c"])
        target = torch.tensor([[.6, .2]], dtype=torch.float64)
        tau = torch.zeros(1, dtype=torch.float64)
        self.assertTrue(converged(problem, target, 1e-12).all())

        analytic = ConditionalField(hidden_dim=8, message_steps=1).double()
        coupling = torch.zeros_like(source, requires_grad=True)
        analytic.coupling_rate = lambda state, time, context: coupling
        initial_field = analytic(source, tau, problem).detach()
        torch.testing.assert_close(initial_field, torch.tensor([[.5, 0.]], dtype=torch.float64))
        analytic_loss = cfm_loss(analytic, problem, source, target, tau)[0]
        analytic_gradient, = torch.autograd.grad(analytic_loss, coupling)
        # Contact 2's negative preactivation blocks its correction through r_1.
        self.assertEqual(float(analytic_gradient[0, 0]), 0.)
        self.assertGreater(abs(float(analytic_gradient[0, 1])), 0.)

        direct = ConditionalField(hidden_dim=8, message_steps=1, head_type="direct").double()
        velocity = initial_field.clone().requires_grad_()
        direct.coupling_rate = lambda state, time, context: velocity
        direct_loss = cfm_loss(direct, problem, source, target, tau)[0]
        direct_gradient, = torch.autograd.grad(direct_loss, velocity)
        scale = direct.length_scale / problem["D"].diagonal(dim1=1, dim2=2)
        torch.testing.assert_close(direct_gradient, (initial_field - target) / scale.square())
        self.assertTrue((direct_gradient < 0).all())
        torch.testing.assert_close(direct_loss, analytic_loss)

    def test_direct_forward_is_signed_and_does_not_apply_the_endpoint(self):
        problem = pack([release_problem()])
        model = ConditionalField(hidden_dim=8, message_steps=1, head_type="direct").double()
        with torch.no_grad():
            model.head.bias.fill_(-2.)
        state = torch.tensor([[.2, .03]], dtype=torch.float64)
        time = torch.tensor([.6], dtype=torch.float64)
        expected = -2 * model.length_scale / problem["D"].diagonal(dim1=1, dim2=2)
        torch.testing.assert_close(model(state, time, problem), expected)
        self.assertTrue((model(state, time, problem) < 0).all())
        with self.assertRaises(ValueError):
            model.endpoint(state, time, problem)
        with self.assertRaises(ValueError):
            model(state, torch.ones_like(time), problem)

    def test_direct_cfm_parameter_gradient_matches_finite_difference(self):
        torch.manual_seed(31)
        problem = pack([release_problem()])
        model = ConditionalField(hidden_dim=8, message_steps=1, head_type="direct").double()
        with torch.no_grad():
            model.head.weight.normal_(std=.05)
            model.head.bias.fill_(-.2)
        source = torch.tensor([[.12, .03]], dtype=torch.float64)
        target = torch.tensor([[0., .15]], dtype=torch.float64)
        tau = torch.tensor([.7], dtype=torch.float64)
        loss = cfm_loss(model, problem, source, target, tau)[0]
        loss.backward()
        gradient = float(model.head.bias.grad)
        original = float(model.head.bias.detach())
        values = []
        for delta in (-1e-6, 1e-6):
            with torch.no_grad():
                model.head.bias.fill_(original + delta)
            values.append(float(cfm_loss(model, problem, source, target, tau)[0].detach()))
        self.assertGreater(abs(gradient), 1e-6)
        self.assertAlmostEqual(gradient, (values[1] - values[0]) / 2e-6, places=7)

    def test_global_direct_field_responds_to_a_remote_current_multiplier(self):
        # One local message step, including gap features, cannot cross this chain.
        count = 5
        matrix = 2 * torch.eye(count, dtype=torch.float64)
        matrix += torch.diag(-torch.ones(count - 1, dtype=torch.float64), 1)
        matrix += torch.diag(-torch.ones(count - 1, dtype=torch.float64), -1)
        problem = problem_with_gram(matrix.tolist(), [-.1] * count)
        torch.manual_seed(11)
        model = ConditionalField(hidden_dim=8, message_steps=1,
            communication="global", attention_heads=2, attention_layers=1,
            feature_version="residual", head_type="direct").double()
        with torch.no_grad():
            model.head.weight.normal_(std=.3)
        state = torch.full_like(problem["c"], .1, requires_grad=True)
        time = torch.tensor([.4], dtype=torch.float64)
        gradient, = torch.autograd.grad(model(state, time, problem)[0, 0], state)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(abs(float(gradient[0, -1])), 1e-8)
        # Removing global attention removes that distant state dependency.
        local = ConditionalField(hidden_dim=8, message_steps=1,
            feature_version="residual", head_type="direct").double()
        local.load_state_dict({key: value for key, value in model.state_dict().items()
                               if not key.startswith("attention.")})
        local_gradient, = torch.autograd.grad(local(state, time, problem)[0, 0], state)
        self.assertEqual(float(local_gradient[0, -1]), 0.)

    def test_padding_empty_scenes_and_contact_permutation(self):
        single = release_problem()
        empty = make_problem(torch.zeros(2, dtype=torch.float64),
            torch.zeros(0, 2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
            torch.empty(0, dtype=torch.float64))
        isolated = make_problem(torch.zeros(1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.tensor([-.1], dtype=torch.float64))
        batch = pack([single, isolated, empty])
        torch.manual_seed(22)
        model = ConditionalField(hidden_dim=8, message_steps=1,
            communication="global", attention_heads=2, attention_layers=1,
            head_type="direct").double()
        with torch.no_grad():
            model.head.weight.normal_(std=.1)
        state = torch.tensor([[.12, .03], [.1, 0.], [0., 0.]], dtype=torch.float64)
        time = torch.full((3,), .4, dtype=torch.float64)
        velocity = model(state, time, batch)
        self.assertTrue(torch.isfinite(velocity).all())
        torch.testing.assert_close(velocity[~batch["mask"]], torch.zeros_like(velocity[~batch["mask"]]))
        torch.testing.assert_close(velocity[0:1], model(state[0:1], time[0:1], pack([single])))
        torch.testing.assert_close(velocity[1:2, :1], model(state[1:2, :1], time[1:2], pack([isolated])))
        permuted = pack([dict(single, J=single["J"].flip(0), c=single["c"].flip(0))])
        torch.testing.assert_close(velocity[0:1].flip(1), model(state[0:1].flip(1), time[:1], permuted))
        result = run_cfm(model, batch, state, 4)
        self.assertTrue(result["completed"].all())
        torch.testing.assert_close(result["final"][~batch["mask"]], torch.zeros_like(state[~batch["mask"]]))
        self.assertEqual(int(result["clipped_entries"][-1]), 0)
        self.assertEqual(float(result["projection_l1"][-1]), 0.)


class DirectIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projection_changes_state_not_the_signed_velocity(self):
        problem = scalar_problem()
        start = torch.full_like(problem["c"], .1)
        release = DirectFunction(lambda state, time, context: torch.full_like(state, -.08))
        result = run_cfm(release, problem, start, 4)
        torch.testing.assert_close(result["final"], torch.full_like(start, .02))
        self.assertEqual(int(result["clipped_entries"]), 0)
        # A negative field is allowed; only a step crossing lambda=0 is clipped.
        overshoot = DirectFunction(lambda state, time, context: -torch.ones_like(state))
        with patch("src.multiplier_flow.solvers.pgs", side_effect=AssertionError("Hidden PGS solve")):
            projected = run_cfm(overshoot, problem, start, 4)
            unprojected = run_cfm(overshoot, problem, start, 4, project_state=False)
            trained = integrate_cfm(overshoot, problem, start, 4)
        torch.testing.assert_close(projected["final"], torch.zeros_like(start))
        torch.testing.assert_close(trained, projected["final"])
        torch.testing.assert_close(unprojected["final"], torch.full_like(start, -.9))
        self.assertTrue(projected["completed"].all())
        self.assertTrue(unprojected["completed"].all())
        self.assertFalse(converged(problem, projected["final"], 1e-7).any())
        self.assertEqual(int(projected["clipped_entries"]), 4)
        self.assertEqual(int(projected["projection_steps"]), 4)
        self.assertAlmostEqual(float(projected["projection_l1"]), .9)
        self.assertAlmostEqual(float(projected["projection_max"]), .25)
        self.assertAlmostEqual(float(projected["min_unprojected_multiplier"]), -.25)
        self.assertEqual(int(unprojected["clipped_entries"]), 0)
        self.assertEqual(float(unprojected["projection_l1"]), 0.)
        self.assertEqual(int(projected["nfe"]), 4)

    def test_disabling_projection_diagnostics_preserves_solver_execution(self):
        problem = scalar_problem()
        start = torch.full_like(problem["c"], .2)
        model = ConditionalField(hidden_dim=8, message_steps=1, head_type="direct").double()
        with torch.no_grad():
            model.head.bias.fill_(-10.)
        diagnostics = ("clipped_entries", "projection_steps", "projection_l1",
                       "projection_max", "min_unprojected_multiplier")
        for options in (dict(), dict(project_state=False), dict(guarded=True)):
            with self.subTest(options=options):
                measured = run_cfm(model, problem, start, 4, **options)
                timed = run_cfm(model, problem, start, 4, collect_diagnostics=False, **options)
                self.assertTrue(measured["completed"].all())
                for key in ("final", "completed", "time", "nfe", "neural_evals",
                            "backtracks", "interventions", "accepted_steps",
                            "failure_code", "min_accepted_h"):
                    torch.testing.assert_close(timed[key], measured[key], atol=0, rtol=0)
                for key in diagnostics:
                    torch.testing.assert_close(timed[key], torch.zeros_like(timed[key]))
                self.assertLess(float(measured["min_unprojected_multiplier"]), 0.)
                if options.get("project_state", True):
                    self.assertGreater(int(measured["clipped_entries"]), 0)
                    self.assertGreater(float(measured["projection_l1"]), 0.)

    def test_training_evaluation_and_suffix_share_the_original_clock(self):
        torch.manual_seed(8)
        problem = pack([release_problem()])
        model = ConditionalField(hidden_dim=8, message_steps=1,
            communication="global", attention_heads=2, attention_layers=1,
            feature_version="residual", head_type="direct").double()
        with torch.no_grad():
            model.head.weight.normal_(std=.03)
        start = torch.tensor([[.12, .03]], dtype=torch.float64)
        for calls in (1, 2, 4, 7):
            trajectory = integrate_cfm(model, problem, start, calls, return_trajectory=True)
            evaluated = run_cfm(model, problem, start, calls)
            self.assertTrue(evaluated["completed"].all())
            self.assertEqual(int(evaluated["nfe"]), calls)
            torch.testing.assert_close(trajectory[-1], evaluated["final"], atol=1e-12, rtol=1e-12)
            split = calls // 2
            suffix = integrate_cfm(model, problem, trajectory[split], calls, start_step=split)
            torch.testing.assert_close(suffix, trajectory[-1], atol=1e-12, rtol=1e-12)
        seen = []
        def velocity(state, tau, context):
            seen.append(tau.clone())
            return tau[:, None].expand_as(state)
        result = run_cfm(DirectFunction(velocity), problem, torch.zeros_like(start), 4)
        self.assertEqual([float(time) for time in seen], [0., .25, .5, .75])
        torch.testing.assert_close(result["final"], torch.full_like(start, .375))

    def test_direct_prefix_receives_suffix_gradient(self):
        class TwoStage(torch.nn.Module):
            head_type = "direct"

            def __init__(self):
                super().__init__()
                self.early = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
                self.late = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))

            def forward(self, state, tau, problem):
                amount = torch.where(tau[:, None] < .5, self.early, self.late)
                return .5 * state + amount

        problem = pack([release_problem()])
        model = TwoStage()
        initial = torch.zeros_like(problem["c"])
        def operation():
            prefix = integrate_cfm(model, problem, initial, 4, end_step=2)
            return integrate_cfm(model, problem, prefix + .01, 4, start_step=2).square().sum()
        operation().backward()
        gradient = float(model.early.grad)
        self.assertGreater(abs(gradient), 1e-6)
        original = float(model.early.detach())
        values = []
        for offset in (-1e-6, 1e-6):
            with torch.no_grad():
                model.early.fill_(original + offset)
            values.append(float(operation().detach()))
        self.assertAlmostEqual(gradient, (values[1] - values[0]) / 2e-6, places=8)

    def test_inner_loss_uses_the_projected_rollout_and_backpropagates(self):
        problem = pack([release_problem()])
        model = ConditionalField(hidden_dim=8, message_steps=1, head_type="direct").double()
        with torch.no_grad():
            model.head.bias.fill_(1.)
        source = torch.tensor([[.15, .03]], dtype=torch.float64)
        target = torch.tensor([[0., .15]], dtype=torch.float64)
        tau = torch.tensor([.4], dtype=torch.float64)
        settings = dict(inner_rollout=dict(weight=1., calls=[2]), recovery=dict(weight=0.))
        loss, terms, costs = training_objective(model, problem, source, target, tau,
            settings, torch.Generator().manual_seed(42))
        prediction = integrate_cfm(model, problem, torch.zeros_like(source), 2)
        expected_inner = physical_endpoint_loss(problem, prediction, target, model.length_scale, {})[0]
        torch.testing.assert_close(terms["inner"], expected_inner.detach())
        torch.testing.assert_close(loss.detach(), terms["cfm"] + terms["inner"])
        self.assertEqual(costs, dict(inner_calls=2, recovery_calls=0))
        inner_gradient, = torch.autograd.grad(expected_inner, model.head.bias)
        self.assertGreater(abs(float(inner_gradient)), 1e-6)
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)

    def test_nonfinite_direct_field_fails_without_counting_a_step(self):
        problem = scalar_problem()
        start = torch.zeros_like(problem["c"])
        for invalid in (float("nan"), float("inf"), -float("inf")):
            model = DirectFunction(lambda state, time, context: torch.full_like(state, invalid))
            result = run_cfm(model, problem, start, 2)
            self.assertFalse(result["completed"].any())
            self.assertEqual(int(result["failure_code"]), 3)
            self.assertEqual(int(result["nfe"]), 1)
            self.assertEqual(int(result["projection_steps"]), 0)
            torch.testing.assert_close(result["final"], start)

    def test_guarded_batch_does_not_evaluate_a_completed_row_at_time_one(self):
        problem = {key: value.repeat(2, *([1] * (value.ndim - 1)))
                   for key, value in scalar_problem().items()}
        def velocity(state, tau, context):
            self.assertTrue((tau < 1).all())
            # Row 2 needs two backtracks before its first accepted step. Row 1
            # finishes while row 2 is still advancing along its shorter steps.
            return (torch.tensor([[1.], [10.]], dtype=state.dtype) * (.1 - state))
        result = run_cfm(DirectFunction(velocity), problem,
                         torch.zeros_like(problem["c"]), 2, guarded=True, max_backtracks=4)
        self.assertTrue(result["completed"].all())
        self.assertEqual(result["nfe"].tolist(), [2, 8])
        self.assertEqual(result["backtracks"].tolist(), [0, 2])
        torch.testing.assert_close(result["time"], torch.ones(2, dtype=torch.float64))

    def test_evaluation_keeps_unprojected_failure_separate_from_primary_results(self):
        config = tiny_config()
        config["evaluation"].update(calls=[1], start_modes=["zero"], guarded=False)
        problem = scalar_problem()
        split = dict(problem=problem, optimum=torch.full_like(problem["c"], .1),
                     names=["floor_0"])
        model = ConditionalField(hidden_dim=8, message_steps=1, head_type="direct")
        with torch.no_grad():
            model.head.bias.fill_(-10.)
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(model, split, config, torch.device("cpu"), Path(directory) / "eval.json")
        self.assertEqual(report["head_type"], "direct")
        self.assertEqual(report["primary_start_mode"], "zero")
        row = report["modes"]["zero"]["cfm_k1_raw"]
        self.assertEqual(row["success_count"], 0)
        self.assertEqual(row["negative_multiplier"], 0.)
        self.assertEqual(row["clipped_entries"], 1)
        self.assertAlmostEqual(row["projection_l1"], 1.)
        self.assertEqual(row["per_scene"][0]["clipped_entries"], 1)
        diagnostic = row["unclipped_diagnostic"]
        self.assertEqual(diagnostic["success_count"], 0)
        self.assertAlmostEqual(diagnostic["max_negative_multiplier"], 1.)
        self.assertEqual(diagnostic["clipped_entries"], 0)
        self.assertNotIn("seconds", diagnostic)


class HeadCheckpointTests(unittest.TestCase):
    def test_old_analytic_checkpoints_allow_an_explicit_default_head(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for version in ("multiplier_contact_cfm_v2", "multiplier_contact_cfm_v3"):
                config = tiny_config()
                if version.endswith("v3"):
                    config["model"].update(communication="global", attention_heads=2,
                        attention_layers=1, feature_version="residual")
                original = ConditionalField(**config["model"])
                torch.save(dict(format=version, objective="cfm", config=config,
                    model=original.state_dict()), path)
                selected = copy.deepcopy(config)
                selected["model"]["head_type"] = "analytic"
                loaded, _ = load_model(path, selected, "cpu")
                self.assertEqual(loaded.head_type, "analytic")
                for key, tensor in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], tensor)

    def test_legacy_format_cannot_declare_direct_weights(self):
        config = tiny_config()
        config["model"]["head_type"] = "direct"
        model = ConditionalField(**config["model"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for version in ("multiplier_contact_cfm_v2", "multiplier_contact_cfm_v3"):
                torch.save(dict(format=version, objective="cfm", config=config,
                    model=model.state_dict()), path)
                with self.assertRaises(ValueError):
                    load_model(path, config, "cpu")

    def test_current_direct_checkpoint_roundtrip_rejects_analytic_reinterpretation(self):
        config = tiny_config()
        config["model"]["head_type"] = "direct"
        model = ConditionalField(**config["model"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(dict(format=CHECKPOINT_FORMAT, objective="cfm", config=config,
                model=model.state_dict()), path)
            loaded, _ = load_model(path, config, "cpu")
            self.assertEqual(loaded.head_type, "direct")
            selected = copy.deepcopy(config)
            selected["model"]["head_type"] = "analytic"
            with self.assertRaises(ValueError):
                load_model(path, selected, "cpu")


if __name__ == "__main__":
    unittest.main()
