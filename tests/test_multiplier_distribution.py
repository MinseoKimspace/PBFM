"""Exact solution sets, stochastic sources, collapse detection and end-to-end IO."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from test_multiplier_flow import tiny_config
from src.multiplier_flow.distribution import (FORMAT, build_distribution_split,
    resolve_distribution, sample_batch, specification, target_samples)
from src.multiplier_flow.distribution_experiment import (distribution_loss, evaluate_distribution,
    load_distribution_model, score_predictions, train_distribution)
from src.multiplier_flow.model import ConditionalField, load_model
from src.multiplier_flow.problem import contact_endpoint, converged, decode, select
from src.multiplier_flow.solvers import integrate_cfm


def config():
    cfg = tiny_config()
    cfg["model"].update(communication="global", feature_version="residual", attention_heads=2,
                        attention_layers=1, head_type="analytic")
    cfg["distribution"] = dict(target="uniform_simplex", source_scale=1.,
        data=dict(base_contacts=[1, 2], duplicates=[2], train_per_setting=2,
                  val_per_setting=1, test_per_setting=1, inverse_mass_range=[.5, 2.],
                  total_multiplier_range=[.02, .15]),
        train=dict(max_updates=2, validate_every=1, validation_samples=8, validation_calls=4,
                   inner_rollout=dict(weight=1., calls=[2], residual_weight=1., position_weight=1.)),
        evaluation=dict(calls=[2, 4], samples_per_qp=8, batch_size=8, projections=16,
                        tolerance=.001, position_tolerance=.001, timing_repeats=2,
                        hybrid=True, render=False))
    return resolve_distribution(cfg)


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytic_targets_are_distinct_exact_kkt_with_identical_positions(self):
        cfg = config()
        cfg["distribution"]["data"]["duplicates"] = [2, 3]
        split = build_distribution_split(cfg, "test")
        indices = torch.arange(len(split["names"])).repeat_interleave(64)
        targets = target_samples(split, indices, torch.Generator().manual_seed(1))
        qp = select(split["problem"], indices)
        self.assertTrue(converged(qp, targets, 1e-12).all())
        torch.testing.assert_close(decode(qp, targets), decode(qp, split["canonical"][indices]), atol=1e-14, rtol=0)
        torch.testing.assert_close(contact_endpoint(qp, targets), targets, atol=1e-14, rtol=0)
        self.assertGreater(float(targets[:64].var(dim=0).sum()), 1e-5)
        for i, description in enumerate(split["descriptions"]):
            j = split["problem"]["J"][i, split["problem"]["mask"][i]]
            self.assertEqual(int(torch.linalg.matrix_rank(j)), description["base_contacts"])
            self.assertLess(int(torch.linalg.matrix_rank(j)), len(j))

    def test_uniform_simplex_marginal_moments_and_padding(self):
        split = build_distribution_split(config(), "test")
        indices = torch.zeros(12000, dtype=torch.long)
        targets = target_samples(split, indices, torch.Generator().manual_seed(2))
        fractions = targets[:, 0]/split["group_totals"][0, 0]
        self.assertAlmostEqual(float(fractions.mean()), .5, delta=.012)
        self.assertAlmostEqual(float(fractions.var()), 1/12, delta=.006)
        self.assertTrue((targets[:, 2:] == 0).all())
        torch.testing.assert_close(targets[:, :2].sum(-1), split["group_totals"][0, 0].expand(len(targets)))

    def test_prior_is_independent_of_labels_and_point_control_keeps_source_samples(self):
        cfg = config()
        split = build_distribution_split(cfg, "train")
        indices = torch.arange(len(split["names"]))
        def draw(data, options):
            return sample_batch(data, indices, options, torch.Generator().manual_seed(3), torch.Generator().manual_seed(4))
        qp, source, target = draw(split, cfg)
        changed = dict(split, group_totals=split["group_totals"]*2)
        _, changed_source, changed_target = draw(changed, cfg)
        torch.testing.assert_close(source, changed_source, atol=0, rtol=0)
        torch.testing.assert_close(changed_target, target*2, atol=0, rtol=0)
        point_cfg = copy.deepcopy(cfg)
        point_cfg["distribution"]["target"] = "point"
        _, point_source, point_target = draw(split, point_cfg)
        torch.testing.assert_close(point_source, source, atol=0, rtol=0)
        torch.testing.assert_close(point_target, split["canonical"], atol=0, rtol=0)
        self.assertNotIn("group_totals", qp)
        self.assertNotIn("canonical", qp)
        self.assertGreater(float(source.var()), 0)

    def test_conditional_metrics_detect_exact_but_collapsed_solutions(self):
        cfg = config()
        split = build_distribution_split(cfg, "test")
        n = 512
        indices = torch.arange(len(split["names"])).repeat_interleave(n)
        reference = target_samples(split, indices, torch.Generator().manual_seed(6))
        independent = target_samples(split, indices, torch.Generator().manual_seed(7))
        # A different exact constant per QP still counts as conditional collapse.
        collapsed = torch.stack([independent[i*n].repeat(n, 1) for i in range(len(split["names"]))]).reshape_as(reference)
        settings = dict(cfg["distribution"]["evaluation"], samples_per_qp=n)
        completed = torch.ones(len(indices), dtype=torch.bool)
        good = score_predictions(split, indices, independent, reference, completed, settings, 8)
        bad = score_predictions(split, indices, collapsed, reference, completed, settings, 8)
        self.assertEqual(bad["balanced"]["success_rate"], 1.)
        self.assertLess(bad["balanced"]["variance_ratio"], 1e-25)
        self.assertGreater(bad["balanced"]["sliced_w1"], 3*good["balanced"]["sliced_w1"])
        self.assertAlmostEqual(good["balanced"]["variance_ratio"], 1., delta=.15)
        inflated = score_predictions(split, indices, independent*2, reference, completed, settings, 8)
        self.assertAlmostEqual(inflated["balanced"]["group_total_relative_error"], 1.)
        self.assertLess(inflated["balanced"]["success_rate"], 1.)
        failed = score_predictions(split, indices, independent, reference, ~completed, settings, 8)
        self.assertEqual(failed["balanced"]["success_rate"], 0.)
        self.assertEqual(failed["failed_samples"], len(indices))

    def test_inner_loss_uses_random_source_and_has_finite_parameter_gradients(self):
        cfg = config()
        split = build_distribution_split(cfg, "train")
        qp, source, target = sample_batch(split, torch.arange(len(split["names"])), cfg,
            torch.Generator().manual_seed(9), torch.Generator().manual_seed(10))
        model = ConditionalField(**cfg["model"]).double()
        with patch("src.multiplier_flow.distribution_experiment.integrate_cfm", wraps=integrate_cfm) as integrate:
            loss, _, calls = distribution_loss(model, qp, source, target, source.new_full((len(source),), .5),
                                              cfg["distribution"]["train"], torch.Generator().manual_seed(0))
        torch.testing.assert_close(integrate.call_args.args[2], source, atol=0, rtol=0)
        self.assertEqual(calls, 2)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)

    def test_resolver_and_split_reproducibility_do_not_change_original_d(self):
        cfg = config()
        original = copy.deepcopy(cfg)
        resolved = resolve_distribution(cfg, "D_distribution_point", "test_root")
        self.assertEqual(cfg, original)
        self.assertEqual(Path(resolved["outdir"]), Path("test_root/D_distribution_point"))
        self.assertEqual(resolved["distribution"]["target"], "point")
        self.assertEqual(resolved["model"], cfg["model"])
        a, b = (build_distribution_split(cfg, "test") for _ in range(2))
        for key in a["problem"]:
            torch.testing.assert_close(a["problem"][key], b["problem"][key], atol=0, rtol=0)
        validation = build_distribution_split(cfg, "val")
        self.assertFalse(torch.equal(a["problem"]["c"], validation["problem"]["c"]))

    def test_training_checkpoint_reload_and_full_evaluation_report(self):
        for variant in ("D_distribution", "D_distribution_point"):
            cfg = config()
            with tempfile.TemporaryDirectory() as directory:
                cfg = resolve_distribution(cfg, variant, directory)
                result = train_distribution(cfg, torch.device("cpu"))
                path = Path(result["checkpoint"])
                model, checkpoint = load_distribution_model(path, cfg, torch.device("cpu"))
                self.assertEqual(checkpoint["format"], FORMAT)
                self.assertEqual(result["updates"], 2)
                history = json.loads((path.parent/"history.json").read_text())
                self.assertEqual(len(history), 2)
                self.assertIn("sliced_w1", history[-1]["validation"])
                with self.assertRaises(FileExistsError):
                    train_distribution(cfg, torch.device("cpu"))
                changed = copy.deepcopy(cfg)
                changed["distribution"]["source_scale"] = 2.
                with self.assertRaisesRegex(ValueError, "spec"):
                    load_distribution_model(path, changed, torch.device("cpu"))
                with self.assertRaises(ValueError):
                    load_model(path, cfg, torch.device("cpu"))
                split = build_distribution_split(cfg, "test")
                output = path.parent/"evaluation.json"
                cfg["distribution"]["evaluation"]["render"] = True
                with patch("src.multiplier_flow.distribution_experiment.timed", side_effect=lambda device, op: (op(), .02)):
                    report = evaluate_distribution(model, split, cfg, torch.device("cpu"), output=output)
                saved = json.loads(output.read_text())
                self.assertEqual(saved["spec"], specification(cfg))
                for name in ("cfm_k2_raw", "cfm_k4_raw", "cfm_k2_hybrid", "cfm_k4_hybrid", "pgs"):
                    row = report["methods"][name]
                    self.assertAlmostEqual(row["timing"]["total_chunk_median_seconds"], .04)
                    self.assertEqual(len(row["per_qp"]), 2)
                self.assertEqual(report["methods"]["collapsed_exact"]["balanced"]["success_rate"], 1.)
                self.assertLess(report["methods"]["collapsed_exact"]["balanced"]["variance_ratio"], 1e-25)
                with Image.open(output.with_suffix(".png")) as image:
                    self.assertGreater(image.width, 500)
                self.assertTrue(output.with_suffix(".pt").exists())
                self.assertTrue(output.with_suffix(".md").exists())

    def test_invalid_configuration_is_rejected(self):
        for field, value in (("duplicates", [1]), ("base_contacts", []), ("inverse_mass_range", [0, 2]),
                             ("total_multiplier_range", [.2, .1]), ("test_per_setting", 0)):
            cfg = config()
            cfg["distribution"]["data"][field] = value
            with self.assertRaises(ValueError):
                resolve_distribution(cfg)
        cfg = config()
        cfg["distribution"]["evaluation"]["samples_per_qp"] = 1
        with self.assertRaises(ValueError):
            resolve_distribution(cfg)


if __name__ == "__main__":
    unittest.main()
