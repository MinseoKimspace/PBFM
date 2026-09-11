import tempfile
import unittest
from pathlib import Path
import json

import torch

from train_projection import train, energy_objective, temperature_weights
from eval_projection import load_checkpoint, evaluate_projection
from src.contact_flow.physics import PhysicsConfig
from src.contact_flow.solver import SolverConfig
from src.contact_flow.model import ContactFlowNet
from src.contact_flow.io import load_states
from src.contact_flow.dynamics import free_position, make_projection_condition


class TrainingTests(unittest.TestCase):
    def test_training_cache_checkpoint_and_evaluation(self):
        cfg = {"data": {"train_count": 2, "val_count": 2, "num_objects": 2},
               "runtime": {"device": "cpu", "cpu_threads": 1},
               "paths": {"candidates": 2, "rank": 1},
               "flow": {"gain": 1.0, "normalization": "none", "mobility_bound": 1.0},
               "reference_solver": {"steps": 8, "max_steps": 128},
               "solver": {"steps": 4, "max_steps": 32},
               "model": {"hidden_dim": 8, "time_dim": 4, "message_steps": 1, "rank": 1},
               "train": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 4}}
        with tempfile.TemporaryDirectory() as directory:
            result = train(cfg, outdir=directory)
            model, checkpoint = load_checkpoint(result["checkpoint"], torch.device("cpu"))
            self.assertEqual(checkpoint["format"], "contact_flow_v2")
            self.assertEqual(checkpoint["flow"]["gain"], 1.0)
            self.assertEqual(model.flow.normalization, "none")
            self.assertTrue((Path(directory) / "reference_summary.json").exists())
            source, radius = load_states("", "val", 2, 123, 2)
            def factory(state, proposal, r):
                c = make_projection_condition(state, proposal)
                return lambda z, t: model(z, t, r, c, PhysicsConfig())
            metrics, _ = evaluate_projection(source, radius, factory, PhysicsConfig(),
                                              checkpoint["dynamics"], SolverConfig(steps=4, max_steps=32))
            self.assertEqual(metrics["samples"], 2)
            cached = torch.load(Path(directory) / "paths.pt", weights_only=True)
            buffer = cached["buffers"]["train"]
            weights, ess = temperature_weights(buffer, 1e-30)
            self.assertTrue(torch.isfinite(weights).all())
            self.assertGreaterEqual(ess, 1.0)
            uniform, _ = temperature_weights(buffer, 1e-30, "uniform")
            torch.testing.assert_close(uniform / buffer["dt"], torch.full_like(uniform, .5))
            with self.assertRaises(FileExistsError):
                train(cfg, outdir=directory)

    def test_prepare_cache_flow_mismatch_and_quality_gate(self):
        cfg = {"data": {"train_count": 2, "val_count": 2, "num_objects": 2},
               "runtime": {"device": "cpu", "cpu_threads": 1},
               "paths": {"candidates": 2, "rank": 0},
               "flow": {"gain": 1.0, "normalization": "none", "mobility_bound": 1.0},
               "reference_solver": {"steps": 8, "max_steps": 128},
               "train": {"min_violating_scene_coverage": 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            train(cfg, outdir=directory, prepare_only=True)
            with self.assertRaisesRegex(ValueError, "Reference coverage"):
                train(cfg, outdir=directory)
            self.assertFalse((Path(directory) / "best.pt").exists())
            cfg["flow"]["gain"] = 2.0
            with self.assertRaisesRegex(ValueError, "cache/settings differ"):
                train(cfg, outdir=directory, prepare_only=True)

    def test_rank_zero_uniform_training(self):
        cfg = {"data": {"train_count": 2, "val_count": 2, "num_objects": 2},
               "runtime": {"device": "cpu", "cpu_threads": 1},
               "paths": {"candidates": 2, "rank": 1},
               "flow": {"gain": 1.0, "mobility_bound": 1.0},
               "reference_solver": {"steps": 8, "max_steps": 128},
               "solver": {"steps": 4, "max_steps": 32},
               "model": {"hidden_dim": 8, "time_dim": 4, "message_steps": 1},
               "train": {"epochs": 1, "steps_per_epoch": 1, "batch_size": 2}}
        with tempfile.TemporaryDirectory() as directory:
            result = train(cfg, outdir=directory, rank=0, weighting="uniform")
            model, checkpoint = load_checkpoint(result["checkpoint"], torch.device("cpu"))
            self.assertEqual(model.rank, 0)
            self.assertEqual(checkpoint["config"]["paths"]["weighting"], "uniform")
            rows = json.loads((Path(directory) / "history.json").read_text())
            self.assertAlmostEqual(rows[0]["path_ess"], 2.0)

    def test_energy_ablation_has_parameter_gradients(self):
        source, radius = load_states("", "train", 2, 6, 2)
        z = free_position(source, 1/60, -9.8)
        model = ContactFlowNet(hidden_dim=8, time_dim=4, message_steps=1, rank=1)
        value = energy_objective(model, z, torch.zeros(2), radius,
                                 make_projection_condition(source, z), PhysicsConfig(), steps=2)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_energy_ablation_cap_advances_actual_solver_time(self):
        times = []
        def upward(z, tau, radius, condition, physics):
            times.append(tau.clone())
            return torch.stack([torch.zeros_like(z[..., 0]), torch.ones_like(z[..., 0])], -1)
        z = torch.tensor([[[0., .4]]])
        state = torch.cat([z, torch.zeros_like(z)], -1)
        energy_objective(upward, z, torch.tensor([.2]), torch.tensor([[.5]]),
                         make_projection_condition(state, z), PhysicsConfig(), steps=2,
                         max_displacement=.01)
        torch.testing.assert_close(times[1], torch.tensor([.21]))


if __name__ == "__main__":
    unittest.main()
