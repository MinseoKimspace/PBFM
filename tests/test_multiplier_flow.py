import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.multiplier_flow.data import CACHE_FORMAT, prepare, release_problem, sample_pairs
from src.multiplier_flow.evaluation import evaluate, preflight, relinearization_check, scene_indices, solver_validation
from src.multiplier_flow.model import CHECKPOINT_FORMAT, ConditionalField, cfm_loss, load_model
from src.multiplier_flow.problem import (converged, decode, field, gap, make_problem,
                                         pack, position_error, select, value)
from src.multiplier_flow.solvers import active_set_solution, pgs, run_cfm


def scalar_problem():
    return pack([make_problem(torch.zeros(1, dtype=torch.float64),
                              torch.ones(1, 1, dtype=torch.float64),
                              torch.ones(1, dtype=torch.float64),
                              torch.tensor([-.1], dtype=torch.float64), eta_fraction=1.)])


def tiny_config():
    return dict(seed=42, device="cpu", cpu_threads=1, outdir="unused",
                physics=dict(y_ground=0., xy_limit=10., slop=.005, contact_margin=.25, density=1.),
                dynamics=dict(time_step=1/60, gravity_y=-9.8, linear_damping=.1),
                reference=dict(eta_fraction=.9, qp_tolerance=1e-7, max_sweeps=10000),
                data=dict(train_sizes=[3], val_sizes=[3], test_sizes=[5],
                          train_per_size=2, val_per_size=1, test_per_size=1, release_count=1,
                          floor_count=2, sources_per_scene=12, reference_batch_size=8),
                model=dict(hidden_dim=8, message_steps=1, length_scale=.1),
                train=dict(epochs=1, batch_size=8, lr=.001, weight_decay=.0001,
                           grad_clip=1., tau_zero_fraction=.1, solver_validation_every=2, validation_calls=2),
                preflight=dict(stack_sizes=[3]),
                evaluation=dict(calls=[1, 2], tolerance=.001, timing_repeats=1,
                                max_scenes=0, start_modes=["zero", "over"], guarded=True,
                                max_backtracks=2, render=False, image_size=160))


class ProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projected_stationarity(self):
        p = pack([release_problem()])
        x = torch.tensor([[.02, .3]], dtype=torch.float64)
        b = field(p, x)
        self.assertLessEqual(float((gap(p, x) * b).sum()), float(-b.square().sum() / p["eta"]) + 1e-12)
        optimum = torch.tensor([[0., .15]], dtype=torch.float64)
        self.assertTrue(converged(p, optimum, 1e-10).all())
        self.assertGreater(float(gap(p, optimum).norm()), 0)

    def test_release_matches_independent_oracle(self):
        p = pack([release_problem()])
        start = torch.tensor([[.1, .1]], dtype=torch.float64)
        solution = pgs(p, start, tolerance=1e-10)
        exact = active_set_solution(p)
        self.assertTrue(solution["converged"].all())
        torch.testing.assert_close(decode(p, solution["final"]), torch.tensor([[.15, .15]], dtype=torch.float64), atol=1e-9, rtol=0)
        self.assertLess(float(position_error(p, solution["final"], exact)), 1e-16)
        self.assertLess(float(solution["final"][0, 0]), 1e-9)

    def test_singular_redundant_contacts(self):
        one = make_problem(torch.zeros(1, dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64),
                           torch.ones(1, dtype=torch.float64), torch.full((2,), -.1, dtype=torch.float64))
        p = pack([one])
        result = pgs(p, torch.zeros_like(p["c"]), 1e-10)
        self.assertLess(float(position_error(p, result["final"], active_set_solution(p))), 1e-16)
        self.assertEqual(float(position_error(p, torch.tensor([[.1, 0.]], dtype=torch.float64),
                                                torch.tensor([[0., .1]], dtype=torch.float64))), 0.)

    def test_relinearization_does_not_teleport(self):
        result = relinearization_check()
        self.assertTrue(result["passed"])
        self.assertGreater(result["unresolved_stationarity_norm"], 0)

    def test_empty_contacts_and_feasible_components_are_identity(self):
        empty = make_problem(torch.zeros(2, dtype=torch.float64), torch.zeros(0, 2, dtype=torch.float64),
                             torch.ones(2, dtype=torch.float64), torch.empty(0, dtype=torch.float64))
        feasible = make_problem(torch.zeros(1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64),
                                torch.ones(1, dtype=torch.float64), torch.tensor([.01], dtype=torch.float64))
        for p in (pack([empty]), pack([feasible])):
            x = torch.zeros_like(p["c"])
            net = ConditionalField(hidden_dim=8, message_steps=1).double()
            with torch.no_grad():
                net.head.bias.fill_(10.)
            torch.testing.assert_close(net(x, torch.ones(1), p), x)
            torch.testing.assert_close(run_cfm(net, p, x, 4)["final"], x)


class FieldTests(unittest.TestCase):
    def test_loss_matches_conditional_path_not_reference_ode(self):
        class ExactField:
            length_scale = .1
            def __call__(self, state, tau, problem):
                return (.1-state)/(1-tau[:, None])
        p = scalar_problem()
        for initial in (0., .1, .2):
            source = torch.full_like(p["c"], initial)
            target = torch.full_like(source, .1)
            tau = torch.tensor([.6], dtype=torch.float64)
            loss, mse = cfm_loss(ExactField(), p, source, target, tau)
            self.assertLess(float(loss), 1e-25)
            self.assertLess(float(mse), 1e-25)
            for calls in (1, 2, 4, 8):
                result = run_cfm(ExactField(), p, source, calls)
                torch.testing.assert_close(result["final"], target, atol=1e-12, rtol=0)
                self.assertEqual(int(result["nfe"]), calls)
                self.assertEqual(float(result["time"]), 1.)

    def test_raw_euler_has_no_hidden_clipping_or_finish(self):
        class NegativeField:
            def __call__(self, state, tau, problem):
                return -torch.ones_like(state)
        p = scalar_problem()
        result = run_cfm(NegativeField(), p, torch.zeros_like(p["c"]), 4)
        self.assertTrue(result["completed"].all())
        torch.testing.assert_close(result["final"], -torch.ones_like(p["c"]))
        self.assertFalse(converged(p, result["final"], .001).any())

    def test_tau_increases_and_field_is_recomputed(self):
        seen = []
        def field_at_time(state, tau, problem):
            seen.append(tau.clone())
            return tau[:, None].expand_as(state)
        p = scalar_problem()
        result = run_cfm(field_at_time, p, torch.zeros_like(p["c"]), 4)
        self.assertEqual([float(t) for t in seen], [0., .25, .5, .75])
        self.assertAlmostEqual(float(result["final"]), .375)

    def test_parameter_gradient_and_contact_permutation(self):
        torch.manual_seed(2)
        single = release_problem()
        p = pack([single])
        net = ConditionalField(hidden_dim=8, message_steps=2).double()
        with torch.no_grad():
            net.head.weight.normal_(std=.01)
        source = torch.tensor([[.12, .03]], dtype=torch.float64)
        target = torch.tensor([[0., .15]], dtype=torch.float64)
        tau = torch.tensor([.7], dtype=torch.float64)
        loss, _ = cfm_loss(net, p, source, target, tau)
        loss.backward()
        analytic = float(net.head.bias.grad)
        original, values = float(net.head.bias.detach()), []
        for shift in (-1e-5, 1e-5):
            with torch.no_grad():
                net.head.bias.fill_(original + shift)
            values.append(float(cfm_loss(net, p, source, target, tau)[0].detach()))
        with torch.no_grad():
            net.head.bias.fill_(original)
        self.assertAlmostEqual(analytic, (values[1]-values[0])/2e-5, delta=1e-6)
        permuted = pack([dict(single, J=single["J"].flip(0), c=single["c"].flip(0))])
        torch.testing.assert_close(net(source, tau, p).flip(1), net(source.flip(1), tau, permuted))

    def test_guard_and_nonfinite_failure_are_explicit(self):
        p = scalar_problem()
        start = torch.zeros_like(p["c"])
        def good(state, tau, problem):
            return torch.full_like(state, .1)
        result = run_cfm(good, p, start, 4, guarded=True)
        self.assertTrue(result["completed"].all())
        self.assertLessEqual(float(value(p, result["final"])), float(value(p, start)))
        def bad(state, tau, problem):
            return torch.full_like(state, 1000.)
        result = run_cfm(bad, p, start, 1, guarded=True, max_backtracks=2)
        self.assertFalse(result["completed"].any())
        self.assertEqual(int(result["nfe"]), 1)  # Backtracking reuses the slope.
        self.assertEqual(int(result["backtracks"]), 2)
        torch.testing.assert_close(result["final"], start)
        def nonfinite(state, tau, problem):
            return torch.full_like(state, float("nan"))
        result = run_cfm(nonfinite, p, start, 1)
        self.assertEqual(int(result["failure_code"]), 3)
        torch.testing.assert_close(result["final"], start)


class PipelineTests(unittest.TestCase):
    def test_incomplete_flow_never_counts_as_solver_success(self):
        p = scalar_problem()
        def fails_after_reaching_solution(state, tau, problem):
            return torch.where(tau[:, None] == 0, 2*(.1-state), torch.full_like(state, float("nan")))
        cfg = tiny_config()
        cfg["evaluation"]["start_modes"] = ["zero"]
        split = dict(problem=p, optimum=torch.full_like(p["c"], .1), names=["floor_0"])
        report = solver_validation(fails_after_reaching_solution, split, cfg, torch.device("cpu"))
        self.assertEqual(report["balanced"]["success_rate"], 0.)

    def test_balanced_sources_target_optimum_and_fixed_anchor(self):
        p = pack([release_problem(), release_problem(.08, .25)])
        optimum = pgs(p, torch.zeros_like(p["c"]), 1e-10)["final"]
        anchor = p["p"].clone()
        pairs = sample_pairs(p, optimum, dict(sources_per_scene=36), torch.Generator().manual_seed(2))
        for modes in pairs["start_mode"].reshape(2, 36):
            self.assertEqual(torch.bincount(modes, minlength=6).tolist(), [6]*6)
        torch.testing.assert_close(pairs["target"], optimum[pairs["context"]])
        self.assertNotIn("duration", pairs)
        self.assertTrue((pairs["source"] >= 0).all())
        solved = pairs["start_mode"] == 5
        torch.testing.assert_close(pairs["source"][solved], pairs["target"][solved])
        torch.testing.assert_close(p["p"], anchor)

    def test_scene_selection_is_stratified(self):
        names = [f"stack_{i}" for i in range(8)] + [f"release_{i}" for i in range(4)]
        selected = scene_indices(names, 4)
        self.assertTrue(any(names[int(i)].startswith("release") for i in selected))
        self.assertEqual(len(scene_indices(names, 0)), 12)

    def test_training_checkpoint_eval_and_cache_compatibility(self):
        from train_multiplier import train
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            cfg["outdir"] = directory
            result = train(cfg, torch.device("cpu"), max_updates=3)
            self.assertEqual(result["updates"], 3)
            path = Path(directory)/"cfm"/"last.pt"
            net, checkpoint = load_model(path, cfg, torch.device("cpu"))
            self.assertEqual(checkpoint["format"], CHECKPOINT_FORMAT)
            self.assertEqual(checkpoint["updates"], 3)
            self.assertTrue((Path(directory)/"cfm"/"best_solver.pt").exists())
            self.assertFalse((Path(directory)/"endpoint").exists())
            self.assertFalse((Path(directory)/"map").exists())
            with self.assertRaises(FileExistsError):
                train(cfg, torch.device("cpu"), max_updates=1)
            cache = prepare(cfg, Path(directory)/"pairs.pt")
            self.assertEqual(cache["format"], CACHE_FORMAT)
            self.assertTrue(converged(cache["splits"]["train"]["problem"], cache["splits"]["train"]["optimum"], 1e-6).all())
            report = evaluate(net, cache["splits"]["test"], cfg, torch.device("cpu"), Path(directory)/"eval.json")
            self.assertEqual(report["tau_interval"], [0., 1.])
            self.assertIn("cfm_k1_raw", report["modes"]["zero"])
            self.assertNotIn("finite_time_map_position_mse", report["modes"]["zero"]["cfm_k1_raw"])
            self.assertIn("negative_multiplier", report["modes"]["over"]["cfm_k2_guarded"])
            self.assertIn("balanced", checkpoint.get("solver_validation", {}) or
                          torch.load(Path(directory)/"cfm"/"best_solver.pt", weights_only=True)["solver_validation"])
            changed = copy.deepcopy(cfg)
            changed["reference"]["eta_fraction"] = .8
            with self.assertRaises(ValueError):
                prepare(changed, Path(directory)/"pairs.pt")
            old = Path(directory)/"old.pt"
            torch.save(dict(format="multiplier_map_v1", objective="map"), old)
            with self.assertRaisesRegex(ValueError, "old B/C"):
                load_model(old, cfg, torch.device("cpu"))
            legacy_cache = Path(directory)/"old_cache.pt"
            torch.save(dict(format="multiplier_segments_v2", spec=cache["spec"]), legacy_cache)
            with self.assertRaises(ValueError):
                prepare(cfg, legacy_cache)

    def test_preflight_and_renderer(self):
        cfg = tiny_config()
        cfg["evaluation"]["render"] = True
        with tempfile.TemporaryDirectory() as directory:
            result = preflight(cfg, Path(directory)/"preflight.json")
            self.assertTrue(result["passed"])
            self.assertTrue((Path(directory)/"renders"/"floor_under.png").exists())


if __name__ == "__main__":
    unittest.main()
