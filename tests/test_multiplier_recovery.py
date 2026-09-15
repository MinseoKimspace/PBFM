import json
import math
import unittest

import torch

from src.multiplier_flow.model import LocalProjection
from src.multiplier_flow.problem import make_problem, pack, position_error
from src.multiplier_flow.recovery import perturb_state, recovery_evaluation, recovery_gain
from src.multiplier_flow.solvers import integrate_cfm


def chain_problem():
    jacobian = torch.tensor([[1., -1., 0., 0.],
                             [0., 1., -1., 0.],
                             [0., 0., 1., -1.]], dtype=torch.float64)
    optimum = torch.full((1, 3), 0.2, dtype=torch.float64)
    single = make_problem(torch.zeros(4, dtype=torch.float64), jacobian,
                          torch.ones(4, dtype=torch.float64), -(jacobian @ jacobian.T @ optimum[0]))
    return pack([single]), optimum


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items() if not key.endswith("seconds")}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_perturbations_are_repeatable_bounded_and_return_clipped_delta(self):
        problem, optimum = chain_problem()
        state = torch.zeros_like(optimum)
        for kind in ("independent", "correlated"):
            settings = dict(kind=kind, amplitude=0.1, length_scale=0.1)
            actual, delta = perturb_state(problem, state, **settings,
                                          generator=torch.Generator().manual_seed(13))
            repeated, repeated_delta = perturb_state(problem, state, **settings,
                                                     generator=torch.Generator().manual_seed(13))
            torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
            torch.testing.assert_close(delta, repeated_delta, rtol=0, atol=0)
            torch.testing.assert_close(actual - state, delta, rtol=0, atol=0)
            self.assertTrue((actual >= 0).all())
            self.assertLessEqual(float(position_error(problem, actual, state).sqrt()[0]), 0.010000000001)
            diagonal = problem["D"].diagonal(dim1=1, dim2=2)
            self.assertTrue((delta.abs() <= 8 * 0.01 / diagonal + 1e-14).all())
        # A negative sampled displacement from a zero multiplier is actually zero.
        self.assertTrue((delta == 0).any())

    def test_padding_and_null_problems_do_not_receive_physical_noise(self):
        empty = make_problem(torch.zeros(2, dtype=torch.float64), torch.zeros(0, 2, dtype=torch.float64),
                             torch.ones(2, dtype=torch.float64), torch.empty(0, dtype=torch.float64))
        problem = pack([empty])
        state = torch.zeros_like(problem["c"], requires_grad=True)
        final, delta = perturb_state(problem, state, kind="correlated", amplitude=0.1, length_scale=0.1,
                                     generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(final, state)
        torch.testing.assert_close(delta, torch.zeros_like(state))
        final.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        gain = recovery_gain(problem, state, final, state, final)
        self.assertEqual(gain["actual_perturbation_rms"], 0.)
        self.assertIsNone(gain["gain"])
        json.dumps(gain, allow_nan=False)

    def test_redundant_contact_null_direction_has_no_gain_ratio(self):
        single = make_problem(torch.zeros(1, dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64),
                              torch.ones(1, dtype=torch.float64), torch.full((2,), -0.2, dtype=torch.float64))
        problem = pack([single])
        clean = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
        perturbed = torch.tensor([[0.15, 0.05]], dtype=torch.float64)
        report = recovery_gain(problem, clean, perturbed, clean, perturbed)
        self.assertIsNone(report["gain"])
        self.assertLess(report["actual_perturbation_rms"], 1e-10)
        json.dumps(report, allow_nan=False)

    def test_suffix_uses_original_clock_and_matches_unsplit_solve(self):
        seen = []

        class TimeDependentEndpoint:
            neural_evaluations = 1

            def endpoint(self, state, tau, problem):
                seen.append(float(tau[0]))
                return state + (1 - tau[:, None]) * (1 + tau[:, None])

        problem, optimum = chain_problem()
        model, start = TimeDependentEndpoint(), torch.zeros_like(optimum)
        prefix = integrate_cfm(model, problem, start, 4, end_step=2)
        self.assertEqual(seen, [0., 0.25])
        seen.clear()
        suffix = integrate_cfm(model, problem, prefix, 4, start_step=2, return_trajectory=True)
        self.assertEqual(seen, [0.5, 0.75])
        self.assertEqual(suffix.shape, (3, 1, 3))
        torch.testing.assert_close(suffix[0], prefix)
        whole = integrate_cfm(model, problem, start, 4)
        torch.testing.assert_close(suffix[-1], whole)

    def test_local_head_already_contracts_a_chain_smooth_mode(self):
        problem, optimum = chain_problem()
        smooth = torch.tensor([[math.sqrt(0.5), 1., math.sqrt(0.5)]], dtype=torch.float64)
        perturbed = optimum + 0.01 * smooth
        clean_final = integrate_cfm(LocalProjection(), problem, optimum, 1)
        perturbed_final = integrate_cfm(LocalProjection(), problem, perturbed, 1)
        report = recovery_gain(problem, optimum, perturbed, clean_final, perturbed_final)
        self.assertAlmostEqual(report["gain"], math.sqrt(0.5), places=12)
        self.assertGreater(float(position_error(problem, perturbed_final, optimum)), 0.)

    def test_diagnostics_use_strict_json_matching_noise_and_stratified_scenes(self):
        problem, optimum = chain_problem()
        split = dict(problem={key: value.repeat(4, *([1] * (value.ndim - 1))) for key, value in problem.items()},
                     optimum=optimum.repeat(4, 1), names=["stack_0", "stack_1", "stack_2", "release_0"])
        config = dict(seed=3, model=dict(length_scale=0.1), reference=dict(max_sweeps=100),
                      evaluation=dict(tolerance=1e-5, recovery=dict(
                          max_scenes=2, calls=[2, 4], start_fractions=[0.5],
                          kinds=["independent", "correlated"], amplitudes=[0.1])))
        report = recovery_evaluation(LocalProjection(), split, config, "cpu")
        repeated = recovery_evaluation(LocalProjection(), split, config, "cpu")
        json.dumps(report, allow_nan=False)
        self.assertEqual(without_timings(report), without_timings(repeated))
        self.assertEqual(report["scenes"], ["stack_0", "release_0"])
        self.assertEqual(len(report["rows"]), 16)
        summary = report["summary"]["cfm_k2_independent"]
        self.assertEqual(summary["trials"], 2)
        self.assertEqual(summary["clean"]["completed_count"], 2)
        self.assertEqual(summary["valid_gain_count"] + summary["null_gain_count"], 2)
        local_first = next(row for row in report["cooperative_case"]["rows"]
                           if row["method"] == "local" and row["calls"] == 1)
        torch.testing.assert_close(torch.tensor(local_first["final_gap"]), torch.tensor([-0.1, 0., -0.1]))
        self.assertLess(max(abs(value) for value in report["cooperative_case"]["exact_coupling_head_gap"]), 1e-7)
        paired = {}
        for row in report["rows"]:
            self.assertEqual(row["clean"]["trajectory"][0]["tau"], 0.5)
            self.assertEqual(row["perturbed"]["trajectory"][-1]["tau"], 1.)
            self.assertEqual(row["clean"]["nfe"], row["remaining_calls"])
            self.assertIn("contact_evals", row["pgs_perturbed"])
            self.assertIn("projection_position_mse", row["perturbed"])
            key = (row["name"], row["calls"], row["kind"])
            comparable = {name: value for name, value in without_timings(row).items() if name != "method"}
            if key in paired:
                self.assertEqual(comparable, paired[key])
            paired[key] = comparable

    def test_nonfinite_suffix_is_a_json_safe_failure(self):
        class FailingEndpoint:
            neural_evaluations = 1

            def endpoint(self, state, tau, problem):
                return torch.where(tau[:, None] < 0.5, torch.ones_like(state),
                                   torch.full_like(state, float("nan")))

        problem, optimum = chain_problem()
        split = dict(problem=problem, optimum=optimum, names=["chain_0"])
        config = dict(model=dict(length_scale=0.1), reference=dict(max_sweeps=100),
                      evaluation=dict(tolerance=1e-5, recovery=dict(
                          max_scenes=1, calls=[2], kinds=["independent"], amplitudes=[0.1])))
        report = recovery_evaluation(FailingEndpoint(), split, config, "cpu")
        json.dumps(report, allow_nan=False)
        learned = next(row for row in report["rows"] if row["method"] == "cfm")
        self.assertFalse(learned["perturbed"]["completed"])
        self.assertFalse(learned["perturbed"]["success"])
        self.assertIsNone(learned["perturbed"]["projected_gradient"])
        self.assertIsNone(learned["gain"])
        summary = report["summary"]["cfm_k2_independent"]
        self.assertEqual(summary["perturbed"]["completed_count"], 0)
        self.assertEqual(summary["perturbed"]["success_count"], 0)
        self.assertEqual(summary["null_gain_count"], 1)


if __name__ == "__main__":
    unittest.main()
