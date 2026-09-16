# Direct velocity head ablation

The research target is lower runtime at the same contact-solver tolerance.
Compare two output heads while keeping the dataset, global communication,
features, optimizer, and training budgets unchanged. This experiment does not
add super tokens, cross-attention, or physical-frame rollout training.

| Variant | Head | Training losses |
| --- | --- | --- |
| `B` | Analytic contact endpoint | CFM |
| `B_direct` | Signed direct velocity | CFM |
| `D` | Analytic contact endpoint | CFM + inner endpoint loss |
| `D_direct` | Signed direct velocity | CFM + inner endpoint loss |

`A`, `C`, and `D_recovery` remain available with the analytic head. Recovery
training is disabled in all four variants above; recovery evaluation remains a
separate diagnostic.

Recovery reports include both each model's own prefix and a `common_start`
comparison. The latter uses the same untrained local-analytic prefix and
identical perturbed state for every checkpoint, then preserves the original
FM time and remaining call budget. Reference endpoints are used only to
measure errors in this common-state comparison.

## Field and integration

Both heads receive the same current multiplier, contact-problem features, and
FM time. The analytic head retains its contact endpoint and stable
convex-combination update. The direct head predicts a signed velocity without
an endpoint ReLU or division by remaining FM time.

CFM supervises the **raw, unprojected velocity** along the same straight paths
between source multipliers and reference solutions. During direct-head
integration, each Euler step uses

```text
lambda_next = max(0, lambda + dtau * velocity_theta(lambda, tau, problem))
```

Padding is masked. The inner loss differentiates through these deployed
updates and evaluates the final KKT residual and reference-position error.
Projection enforces nonnegative multipliers; convergence still requires the
same residual and penetration checks used for the analytic head and PGS.

PGS supplies converged **training endpoints**, not intermediate training
trajectories. Zero-start CFM inference has no PGS initialization or finishing
step. The existing `over` and `mixed` starts are reference-derived diagnostics,
so report their results separately from zero-start inference.

## Run the controlled comparison

Use the single configuration `configs/multiplier_cfm_ablation.yaml`. Its batch
size remains 512 and its update budget remains 30,000. The explicit output root
below keeps this comparison separate from previous results; use a fresh root
for another repeat. All variants share one solution-pair cache.

PowerShell:

```powershell
$configPath = "configs/multiplier_cfm_ablation.yaml"
$experimentRoot = "runs/multiplier_direct_head"
python train_multiplier.py --config $configPath --outdir $experimentRoot --variant B --prepare-only
if ($LASTEXITCODE -ne 0) { throw "Pair preparation failed" }
foreach ($variantName in @("B", "B_direct", "D", "D_direct")) {
    python train_multiplier.py --config $configPath --outdir $experimentRoot --variant $variantName
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $variantName" }
    python eval_multiplier.py --config $configPath --outdir $experimentRoot --variant $variantName --checkpoint best_solver
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $variantName" }
    python eval_multiplier_rollout.py --config $configPath --outdir $experimentRoot --variant $variantName --checkpoint best_solver --no-render
    if ($LASTEXITCODE -ne 0) { throw "Physical rollout failed: $variantName" }
}
```

For a resource estimate, run `train_multiplier.py` with the same arguments and
`--profile-only`. This measures the configured training objective, including
inner integration for the D variants. It does not train a checkpoint.

## Read the results

Each variant writes to `<output root>/<variant>/`. Profiles and checkpoints
record the actual head and solver description. `best_solver.pt` is selected
with the existing group-balanced solver validation, independently of the
checkpoint selected by validation CFM loss.

Compare success rate, KKT residual, penetration, projection diagnostics, and
runtime at the same call budgets and tolerance. Report single-scene physical
rollout latency separately from batched frozen-QP timing. Existing fixed-budget
timings describe an accuracy/runtime tradeoff; a faster run with more failures
does not establish a speedup at matched tolerance. Do not drop failed scenes
from speed comparisons or treat the current Python PGS implementation as a
production-engine benchmark.

Clipping counters and unclipped counterfactuals are collected in separate,
untimed passes. Timed inference still includes the actual projection update.
These diagnostics do not change the trained model or its primary trajectory.

New checkpoints use `multiplier_contact_cfm_v4`. Existing v2/v3 analytic
checkpoints remain loadable with their matching architecture and data settings;
an omitted `head_type` means `analytic`. Analytic checkpoints cannot be loaded
as direct-head checkpoints even when tensor shapes match.
