import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.func import jvp

from src.multiplier_flow.data import prepare, release_problem, sample_segments
from src.multiplier_flow.evaluation import evaluate, preflight, relinearization_check, scene_indices
from src.multiplier_flow.model import MultiplierMap, losses
from src.multiplier_flow.problem import (converged, decode, field, gap, make_problem,
                                         pack, position_error, residuals, select, value)
from src.multiplier_flow.solvers import active_set_solution, pgs, reference, run_map


def scalar_problem():
    return pack([make_problem(torch.zeros(1, dtype=torch.float64),
                              torch.ones(1, 1, dtype=torch.float64),
                              torch.ones(1, dtype=torch.float64),
                              torch.tensor([-0.1], dtype=torch.float64), eta_fraction=1.0)])


def tiny_config():
    return dict(seed=42, device="cpu", cpu_threads=1, outdir="unused",
                physics=dict(y_ground=0., xy_limit=10., slop=.005, contact_margin=.25, density=1.),
                dynamics=dict(time_step=1/60, gravity_y=-9.8, linear_damping=.1),
                reference=dict(eta_fraction=.9, rtol=1e-5, atol=1e-8, max_steps=10000,
                               qp_tolerance=1e-7, max_sweeps=10000),
                data=dict(train_sizes=[3], val_sizes=[3], test_sizes=[5],
                          train_per_size=2, val_per_size=1, test_per_size=1, release_count=1,
                          segments_per_scene=5, h_range=[.25, 4.]),
                model=dict(hidden_dim=8, message_steps=1, length_scale=.1),
                train=dict(epochs=1, batch_size=8, lr=.001, weight_decay=.0001,
                           grad_clip=1., matching_weight=.1),
                preflight=dict(stack_sizes=[3], times=[1., 256.]),
                evaluation=dict(total_time=4., calls=[1, 2], tolerance=.001, timing_repeats=1,
                                max_scenes=2, start_modes=["zero", "over"], max_backtracks=4,
                                render=False, image_size=480))


class ProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_continuous_descent_and_projected_stationarity(self):
        p = pack([release_problem()])
        x = torch.tensor([[.02, .3]], dtype=torch.float64)
        b = field(p, x)
        self.assertLessEqual(float((gap(p, x) * b).sum()), float(-b.square().sum() / p["eta"]) + 1e-12)
        optimum = torch.tensor([[0., .15]], dtype=torch.float64)
        self.assertTrue(converged(p, optimum, 1e-10).all())
        self.assertGreater(float(gap(p, optimum).norm()), 0)  # grad Q is NOT the stopping test.

    def test_release_matches_independent_oracle(self):
        p = pack([release_problem()])
        start = torch.tensor([[.1, .1]], dtype=torch.float64)
        torch.testing.assert_close(decode(p, start), torch.tensor([[.2, .1]], dtype=torch.float64))
        solution = pgs(p, start, tolerance=1e-10)
        exact = active_set_solution(p)
        self.assertTrue(solution["converged"].all())
        torch.testing.assert_close(decode(p, solution["final"]), torch.tensor([[.15, .15]], dtype=torch.float64), atol=1e-9, rtol=0)
        self.assertLess(float(position_error(p, solution["final"], exact)), 1e-16)
        self.assertLess(float(solution["final"][0, 0]), 1e-9)

    def test_dual_descent_is_not_penetration_energy_descent(self):
        p = pack([make_problem(torch.zeros(2, dtype=torch.float64),
            torch.tensor([[1., 0.], [.5, .75**.5]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64), torch.tensor([-.031, -.02], dtype=torch.float64))])
        lam = torch.tensor([[.02, .02]], dtype=torch.float64)
        g, b = gap(p, lam), field(p, lam)
        g_dot = (p["D"] @ b[..., None]).squeeze(-1)
        self.assertLess(float((g * b).sum()), 0)
        self.assertGreater(float((g.clamp_max(0) * g_dot).sum()), 0)

    def test_reference_is_finite_time_not_equilibrium(self):
        p = scalar_problem()
        for initial in (0., .1, .2):
            x = torch.tensor([[initial]], dtype=torch.float64)
            result = reference(p, x, 1., rtol=1e-7, atol=1e-10)
            expected = .1 + (initial - .1) * torch.exp(torch.tensor(-1., dtype=torch.float64))
            torch.testing.assert_close(result["final"], expected.reshape(1, 1), atol=2e-7, rtol=0)
            self.assertGreaterEqual(float(result["final"].min()), 0)
            self.assertLessEqual(float(value(p, result["final"])), float(value(p, x)) + 1e-12)
        zero = torch.zeros(1, 1, dtype=torch.float64)
        torch.testing.assert_close(reference(p, zero, 0.)["final"], zero)

    def test_singular_redundant_contacts(self):
        one = make_problem(torch.zeros(1, dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64),
                           torch.ones(1, dtype=torch.float64), torch.full((2,), -.1, dtype=torch.float64))
        p = pack([one])
        result = pgs(p, torch.zeros_like(p["c"]), 1e-10)
        exact = active_set_solution(p)
        self.assertLess(float(position_error(p, result["final"], exact)), 1e-16)
        left = torch.tensor([[.1, 0.]], dtype=torch.float64)
        right = torch.tensor([[0., .1]], dtype=torch.float64)
        self.assertEqual(float(position_error(p, left, right)), 0.)

    def test_relinearization_does_not_teleport(self):
        result = relinearization_check()
        self.assertTrue(result["passed"])
        self.assertGreater(result["unresolved_stationarity_norm"], 0)

    def test_no_contacts_padding_and_inactive_fixed_point(self):
        empty = make_problem(torch.zeros(2, dtype=torch.float64), torch.zeros(0, 2, dtype=torch.float64),
                             torch.ones(2, dtype=torch.float64), torch.empty(0, dtype=torch.float64))
        p = pack([empty])
        x = torch.zeros_like(p["c"])
        torch.testing.assert_close(reference(p, x, 2.)["final"], x)
        self.assertTrue(pgs(p, x)["converged"].all())
        net = MultiplierMap(hidden_dim=8, message_steps=1).double()
        torch.testing.assert_close(net(x, torch.ones(1), p), x)


class ModelTests(unittest.TestCase):
    def test_exact_map_matching_and_time_composition(self):
        class ExactMap:
            length_scale = .1
            def __call__(self, x, h, problem):
                return .1 + (x - .1) * torch.exp(-h[:, None])
        p = scalar_problem()
        start = torch.zeros_like(p["c"])
        h = torch.tensor([2.], dtype=torch.float64)
        target = ExactMap()(start, h, p)
        endpoint, matching = losses(ExactMap(), p, start, h, target, matching=True)
        self.assertLess(float(endpoint + matching), 1e-25)
        for calls in (1, 2, 4, 8):
            result = run_map(ExactMap(), p, start, 2., calls)
            torch.testing.assert_close(result["final"], target, atol=1e-12, rtol=0)

    def test_identity_nonnegativity_and_negative_increment(self):
        p = scalar_problem()
        net = MultiplierMap(hidden_dim=8, message_steps=1).double()
        x = torch.tensor([[.2]], dtype=torch.float64)
        torch.testing.assert_close(net(x, torch.zeros(1), p), x, atol=0, rtol=0)
        output = net(x, torch.ones(1), p)
        self.assertGreaterEqual(float(output), 0)
        self.assertLess(float(output), float(x))
        solved = torch.tensor([[.1]], dtype=torch.float64)
        torch.testing.assert_close(net(solved, torch.ones(1), p), solved, atol=0, rtol=0)

    def test_jvp_and_mixed_parameter_derivative(self):
        torch.manual_seed(2)
        p = pack([release_problem()])
        net = MultiplierMap(hidden_dim=8, message_steps=2).double()
        with torch.no_grad():
            net.head.weight.normal_(std=.01)
        x = torch.tensor([[.12, .03]], dtype=torch.float64)
        h = torch.tensor([.7], dtype=torch.float64)
        _, derivative = jvp(lambda time: net(x, time, p), (h,), (torch.ones_like(h),))
        finite_difference = (net(x, h + 1e-5, p) - net(x, h - 1e-5, p)) / 2e-5
        torch.testing.assert_close(derivative, finite_difference, atol=1e-8, rtol=1e-6)
        target = reference(p, x, h)["final"]
        endpoint, matching = losses(net, p, x, h, target, matching=True)
        total = endpoint + .1 * matching
        total.backward()
        analytic = float(net.head.bias.grad)
        parameter = net.head.bias
        original = float(parameter)
        values = []
        for shift in (-1e-5, 1e-5):
            with torch.no_grad():
                parameter.fill_(original + shift)
            e, m = losses(net, p, x, h, target, matching=True)
            values.append(float((e + .1 * m).detach()))
        with torch.no_grad():
            parameter.fill_(original)
        numerical = (values[1] - values[0]) / 2e-5
        self.assertAlmostEqual(analytic, numerical, delta=1e-6)
        self.assertTrue(all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None))

    def test_contact_permutation(self):
        single = release_problem()
        permuted = dict(single, J=single["J"].flip(0), c=single["c"].flip(0))
        net = MultiplierMap(hidden_dim=8, message_steps=2).double()
        with torch.no_grad():
            net.head.weight.normal_(std=.01)
        x = torch.tensor([[.1, .04]], dtype=torch.float64)
        h = torch.tensor([1.2], dtype=torch.float64)
        torch.testing.assert_close(net(x, h, pack([single])).flip(1), net(x.flip(1), h, pack([permuted])))

    def test_guard_preserves_time_and_never_hides_failure(self):
        p = scalar_problem()
        net = MultiplierMap(hidden_dim=8, message_steps=1).double()
        start = torch.zeros_like(p["c"])
        result = run_map(net, p, start, 4., 4, guarded=True)
        self.assertTrue(result["completed"].all())
        self.assertEqual(float(result["time"]), 4.)
        self.assertLessEqual(float(value(p, result["final"])), float(value(p, start)))
        class BadMap:
            def __call__(self, x, h, problem):
                return x + 1000
        rejected = run_map(BadMap(), p, start, 1., 1, guarded=True, max_backtracks=2)
        self.assertFalse(rejected["completed"].any())
        torch.testing.assert_close(rejected["final"], start)
        self.assertEqual(int(rejected["nfe"]), 3)


class PipelineTests(unittest.TestCase):
    def test_balanced_32_segments_and_fixed_anchor(self):
        p = pack([release_problem(), release_problem(.08, .25)])
        optimum = pgs(p, torch.zeros_like(p["c"]), 1e-10)["final"]
        original = p["p"].clone()
        settings = dict(segments_per_scene=32, h_range=[.25, 256.],
                        time_anchors=[32., 64., 128., 256.], anchor_fraction=1.)
        result = sample_segments(p, optimum, settings, torch.Generator().manual_seed(2))
        for modes in result["start_mode"].reshape(2, 32):
            counts = torch.bincount(modes, minlength=5)
            self.assertLessEqual(int(counts.max()-counts.min()), 1)
        self.assertEqual(int(result["horizon_check"].sum()), 2)
        self.assertTrue((result["duration"][result["horizon_check"]] == 256).all())
        self.assertTrue((result["start"] >= 0).all())
        self.assertEqual(set(result["duration"].tolist()), {32., 64., 128., 256.})
        torch.testing.assert_close(p["p"], original)

    def test_scene_limits_do_not_take_prefix_and_zero_means_all(self):
        names = [f"stack_{i}" for i in range(8)] + [f"release_{i}" for i in range(4)]
        selected = scene_indices(names, 4)
        self.assertTrue(any(names[int(i)].startswith("release") for i in selected))
        self.assertEqual(len(scene_indices(names, 0)), 12)

    def test_exact_update_budget_and_solver_checkpoint(self):
        from train_multiplier import train
        config = tiny_config()
        config["data"]["h_range"][1] = 256.
        config["train"].update(max_updates=3, solver_validation_every=2, validation_calls=1)
        with tempfile.TemporaryDirectory() as directory:
            config["outdir"] = directory
            result = train(config, "map", torch.device("cpu"))
            self.assertEqual(result["updates"], 3)
            checkpoint = torch.load(Path(directory)/"map"/"last.pt", weights_only=True)
            self.assertEqual(checkpoint["updates"], 3)
            self.assertTrue((Path(directory)/"map"/"best_solver.pt").exists())

    def test_both_training_arms_and_evaluation(self):
        from train_multiplier import train
        config = tiny_config()
        config["data"]["h_range"][1] = 256.
        with tempfile.TemporaryDirectory() as directory:
            config["outdir"] = directory
            train(config, "endpoint", torch.device("cpu"), epochs=1)
            train(config, "map", torch.device("cpu"), epochs=1)
            with self.assertRaises(FileExistsError):
                train(config, "endpoint", torch.device("cpu"), epochs=1)
            checkpoint = torch.load(Path(directory) / "map" / "best.pt", weights_only=True)
            cache = prepare(config, Path(directory) / "segments.pt")
            self.assertEqual(checkpoint["cache_spec"], cache["spec"])
            model = MultiplierMap(**config["model"])
            model.load_state_dict(checkpoint["model"])
            report = evaluate(model.eval(), cache["splits"]["test"], config, torch.device("cpu"),
                              Path(directory) / "eval.json")
            self.assertIn("finite_time_map_position_mse", report["modes"]["zero"]["map_k1_raw"])
            self.assertIn("interventions", report["modes"]["over"]["map_k2_guarded"])

    def test_cache_shared_but_mismatch_rejected(self):
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.pt"
            first = prepare(config, path)
            second = prepare(config, path)
            torch.testing.assert_close(first["splits"]["train"]["target"], second["splits"]["train"]["target"])
            self.assertFalse(torch.equal(first["splits"]["train"]["problem"]["p"][:1], first["splits"]["val"]["problem"]["p"][:1]))
            changed = copy.deepcopy(config)
            changed["reference"]["eta_fraction"] = .8
            with self.assertRaises(ValueError):
                prepare(changed, path)

    def test_preflight_independent_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            result = preflight(tiny_config(), Path(directory) / "preflight.json")
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
