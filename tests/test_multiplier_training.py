"""Clock/gradient and end-to-end checks for controlled CFM rollout learning."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from test_multiplier_flow import release_problem, tiny_config
from src.multiplier_flow.experiment import VARIANTS, pair_cache_path, resolve_experiment
from src.multiplier_flow.data import prepare
from src.multiplier_flow.model import ConditionalField, cfm_loss, load_model
from src.multiplier_flow.problem import make_problem, pack
from src.multiplier_flow.solvers import integrate_cfm, pgs, run_cfm
from src.multiplier_flow.training import physical_endpoint_loss, training_objective, validate_objectives


class RolloutTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_integrator_matches_raw_evaluation_and_suffix(self):
        torch.manual_seed(8)
        problem = pack([release_problem()])
        model = ConditionalField(hidden_dim=8, message_steps=1,
            communication="global", attention_heads=2, attention_layers=1,
            feature_version="residual").double()
        with torch.no_grad():
            model.head.weight.normal_(std=.03)
        start = torch.tensor([[.12, .03]], dtype=torch.float64)
        for calls in (1, 2, 4, 7):
            trajectory = integrate_cfm(model, problem, start, calls, return_trajectory=True)
            evaluated = run_cfm(model, problem, start, calls)
            self.assertTrue(evaluated["completed"].all())
            torch.testing.assert_close(trajectory[-1], evaluated["final"], atol=1e-12, rtol=1e-12)
            split = calls // 2
            suffix = integrate_cfm(model, problem, trajectory[split], calls, start_step=split)
            torch.testing.assert_close(suffix, trajectory[-1], atol=1e-12, rtol=1e-12)

    def test_suffix_loss_reaches_parameters_used_only_in_prefix(self):
        class TwoStage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.early = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
                self.late = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))

            def endpoint(self, state, tau, problem):
                amount = torch.where(tau[:, None] < .5, self.early, self.late)
                return .5 * state + amount

        problem = pack([release_problem()])
        model = TwoStage()
        initial = torch.zeros_like(problem["c"])
        def operation():
            prefix = integrate_cfm(model, problem, initial, 4, end_step=2)
            perturbed = prefix + .01  # Same exogenous perturbation for finite differences.
            return integrate_cfm(model, problem, perturbed, 4, start_step=2).square().sum()
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

    def test_position_loss_respects_redundant_multiplier_solutions(self):
        single = make_problem(torch.zeros(1, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
            torch.full((2,), -.1, dtype=torch.float64))
        problem = pack([single])
        prediction = torch.tensor([[.1, 0.]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0., .1]], dtype=torch.float64)
        loss, residual, position = physical_endpoint_loss(problem, prediction, target, .1, {})
        self.assertLess(float(loss.detach()), 1e-20)
        self.assertLess(float(position.detach()), 1e-20)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_combined_objective_keeps_cfm_and_finite_recovery_gradients(self):
        problem = pack([release_problem()])
        source = torch.tensor([[.15, .03]], dtype=torch.float64)
        target = pgs(problem, torch.zeros_like(source), 1e-10)["final"]
        model = ConditionalField(hidden_dim=8, message_steps=1).double()
        settings = dict(inner_rollout=dict(weight=1., calls=[2]),
                        recovery=dict(weight=.5, calls=4, amplitude=.05))
        tau = torch.tensor([.4], dtype=torch.float64)
        expected_cfm = cfm_loss(model, problem, source, target, tau)[0]
        loss, terms, costs = training_objective(model, problem, source, target, tau,
            settings, torch.Generator().manual_seed(42))
        torch.testing.assert_close(terms["cfm"], expected_cfm.detach())
        torch.testing.assert_close(loss.detach(), terms["cfm"] + terms["inner"] + .5 * terms["recovery"])
        self.assertEqual(costs, dict(inner_calls=2, recovery_calls=4))
        loss.backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_variants_share_data_and_separate_outputs_without_mutating_config(self):
        config = tiny_config()
        original = copy.deepcopy(config)
        resolved = [resolve_experiment(config, variant, "runs/example") for variant in VARIANTS]
        self.assertEqual(config, original)
        self.assertEqual(len({str(pair_cache_path(c)) for c in resolved}), 1)
        self.assertEqual(len({c["outdir"] for c in resolved}), len(VARIANTS))
        self.assertTrue(all(c["data"] == config["data"] for c in resolved))
        self.assertEqual([c["model"]["communication"] for c in resolved],
                         ["local", "global", "local", "global", "global", "global", "global"])
        self.assertEqual([c["model"]["head_type"] for c in resolved],
                         ["analytic"] * 5 + ["direct", "direct"])
        self.assertEqual([c["train"]["inner_rollout"]["weight"] for c in resolved], [0, 0, 1, 1, 1, 0, 1])
        disabled = copy.deepcopy(config)
        disabled["train"]["inner_rollout"] = dict(weight=0)
        with self.assertRaises(ValueError):
            resolve_experiment(disabled, "D")
        for invalid in (dict(inner_rollout=dict(calls=[0])), dict(recovery=dict(calls=1)),
                        dict(inner_rollout=dict(weight=float("nan")))):
            with self.assertRaises(ValueError):
                validate_objectives(invalid)

    def test_all_variants_train_reload_and_select_across_budgets(self):
        from train_multiplier import train
        config = tiny_config()
        config["model"].update(attention_heads=2, attention_layers=1)
        config["train"].update(validation_calls=[1, 2],
            inner_rollout=dict(weight=1., calls=[1, 2]), recovery=dict(weight=0., calls=2, amplitude=.05))
        config["evaluation"]["start_modes"] = ["zero"]
        with tempfile.TemporaryDirectory() as directory:
            for variant in VARIANTS:
                selected = resolve_experiment(config, variant, directory)
                result = train(selected, torch.device("cpu"), max_updates=2)
                self.assertEqual(result["updates"], 2)
                model, checkpoint = load_model(result["checkpoint"], selected, "cpu")
                self.assertEqual(model.communication, selected["model"]["communication"])
                self.assertEqual(model.head_type, selected["model"]["head_type"])
                self.assertEqual(set(checkpoint["solver_validation"]["by_calls"]), {"1", "2"})
                history = json.loads((Path(selected["outdir"])/"cfm"/"history.json").read_text())
                self.assertGreaterEqual(history[-1]["train_total"], history[-1]["train_cfm"])
                self.assertIn("train_recovery", history[-1])
                self.assertGreater(history[-1]["training_endpoint_evaluations"], 0)
            self.assertTrue((Path(directory)/"shared_data"/"pairs.pt").exists())

    def test_shared_cache_rejects_a_second_writer_and_releases_its_own_lock(self):
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.pt"
            lock = path.with_suffix(".pt.lock")
            lock.write_text("pid=other_writer\n")
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                prepare(config, path)
            self.assertTrue(lock.exists())
            self.assertFalse(path.exists())
            lock.unlink()
            cache = prepare(config, path)
            self.assertFalse(lock.exists())
            self.assertTrue(path.exists())
            self.assertEqual(cache["spec"]["data"], config["data"])


if __name__ == "__main__":
    unittest.main()
