# Contact-energy path FM v2

이 파이프라인은 Box2D의 다음 위치를 직접 회귀하지 않습니다.
**접촉 에너지를 줄이는 후보 경로를 만들고, 그 경로의 실제 tangent를 FM으로 학습**합니다.
v2는 보정 속도·질량 정규화·후보 품질 진단을 명시적으로 분리한 버전입니다.
좋은 참조 경로가 생겼다는 것과, 학습된 FM이 PBD보다 좋은 solver라는 것은 별개의 주장입니다.

## 1. 유지한 부분과 호환성

Box2D 데이터 생성기와 `data/box2d_render.py`, 자유 운동·FD 속도 계산,
별도 full-dynamics 실험(`train.py`, `train_delta.py`, `eval.py`)은 유지했습니다.
구 projection 모델/loss는 삭제된 상태이며 새 코드는 `src/contact_flow/`에 있습니다.

체크포인트는 `contact_flow_v2`, 경로 cache는 `contact_path_cache_v2`,
개별 buffer는 `contact_paths_v2`입니다.
**구 projection 및 contact_flow_v1 체크포인트/cache를 재사용하지 않습니다.**
기존 결과를 삭제하거나 덮어쓰지 않고 새 기본 폴더 `runs/contact_fm_v2`를 씁니다.
gain·정규화·cap이 달라지면 경로도 다시 생성해야 합니다.
평가에서 학습된 모델의 gain만 바꾸는 것은 허용하지 않습니다.

## 2. 물리 시간 k와 solver 시간 tau

```text
현재 위치 x_k, 속도 v_k
  -> v_free = (v_k + gravity*dt) / (1 + damping*dt)
  -> proposal = x_k + v_free*dt
  -> proposal에서 tau=0 시작, 현재 배치 z를 반복 보정
  -> 보정 위치 x_next
  -> v_next = (x_next - x_k) / dt
```

`proposal`은 충돌을 무시한 자유 운동 위치입니다.
중력은 물리 스텝에 한 번 적용하고 solver 적분 중에는 다시 적용하지 않습니다.
condition `(proposal, x_k, v_k)`도 solver 내내 고정됩니다.
다음 field 호출에는 최신 z가 들어가므로 이전 보정 결과에 반응합니다.
별도 self-conditioning head나 learned scalar-energy head는 없습니다.

**물리 rollout의 시작 noise는 항상 0입니다.**
학습 경로만 `a = proposal + Gaussian noise`에서 시작합니다.
이 noise는 주변 보정 상태를 학습하기 위한 것이지 추론 시 random kick을 주려는 것이 아닙니다.
학습 중에도 noise 없는 proposal에서 물리 validation을 수행합니다.

접촉 에너지가 0인 noisy 배치는 현 field로 원래 proposal로 복귀하지 않습니다.
그래서 추론에 noise를 넣으면 noise/dt의 인위적 속도가 생길 수 있습니다.
proximity score만 켜도 이 구조적 성질은 바뀌지 않습니다.

## 3. 공통 field: gain, 정규화, cap

접촉 gap은 원 중심 거리에서 반지름 합을 뺀 값입니다.
바닥과 좌우 벽도 같은 부호 규칙을 씁니다. 음수 gap은 관통입니다.

```text
e_c(z) = min(gap_c(z) + slop, 0)
E(z)   = 0.5 * sum_c e_c(z)^2
g(z)   = grad E(z) = J(z)^T e(z)
mass_i = pi * density * radius_i^2
W      = diag(1/mass_1, 1/mass_1, ..., 1/mass_N, 1/mass_N)
M      = inverse(W)
A      = J W J^T
G      = W J^T                    # near-contact columns only

C      = diag(d) + L L^T          # d >= 0, component-blocked PSD
P_cc   = 1 / ((1 + relative_eps) * A_cc)
B      = P C P
u      = -gain * G P C P G^T g
```

랜덤 후보와 모델 모두 **동일한 `contact_velocity` 함수**를 사용합니다.
`flow.normalization: none`이면 P를 생략하는 비교가 됩니다.
`diagonal`이면 모든 질량에 공통 배율을 곱해도, 고정 C에서 그 배율이 상쇄됩니다.
네트워크는 질량 정보를 입력으로도 받으므로 학습 모델 전체의 질량 불변성을 자동 보장한다는 뜻은 아닙니다.

cap은 **P를 적용하기 전 C**에 적용합니다.
`max(d) + ||L||_F^2`로 spectral upper bound를 계산하며, near-contact 연결 성분별로 제한합니다.
멀리 떨어진 비접촉 슬롯이나 별개의 접촉 집합이 다른 집합의 cap을 소비하지 않습니다.
후보 생성과 모델의 서로 달랐던 cap 적용 방식을 통일했습니다.
정규화한 최종 B를 다시 raw cap으로 자르지 않습니다.

`gain`은 흐름의 속도, `solver.steps=K`는 초기 적분 간격 1/K입니다.
단일 선형 접촉에서 잔차가 `dr/dtau=-alpha*r`이면 최종 잔차는 `r0*exp(-alpha)`입니다.
K를 늘리는 것만으로 alpha가 커지지는 않습니다.
gain을 크게 하면 더 짧은 적분 간격이 필요해 계산 비용이 늘 수 있습니다.

활성 접촉과 Jacobian이 고정되고 그 활성 접촉으로 식을 제한하면
`de/dtau = -gain * A B A e`입니다.
실제 코드는 비위반 near contact도 포함하므로 일반적인 clipped residual 미분에는 활성 마스크가 붙습니다.
대각 정규화가 접촉 결합의 느린 모드까지 해결해 주지는 않습니다.

연속시간에는 다음이 성립합니다.

```text
dE/dtau = -gain * (P G^T g)^T C (P G^T g) <= 0
```

허용 slop 안에서 g=0이면 field도 0입니다.
이 식은 유한 Euler step, 모든 배치에서의 수렴, 물리 rollout 전체의 안정성 보장은 아닙니다.
주 FM 학습은 알려진 analytic gradient를 사용하며,
구 grad_fm처럼 학습된 scalar energy에 대한 2차 미분을 요구하지 않습니다.

## 4. 모델과 참조 경로

모델 노드는 현재 z, 고정 condition, 반지름, inverse mass, gradient, tau embedding을 받습니다.
pair 입력은 법선, gap, 활성 잔차, 접촉 질량 대각성분입니다.
양방향 메시지와 대칭 pair head를 사용해 입자 번호 순서가 결과를 바꾸지 않게 했습니다.
low-rank 항은 같은 near-pair 연결 성분 안에서만 결합합니다.
공통 바닥은 떨어진 원들을 하나의 연결 성분으로 합치지 않습니다.

경로 생성:

1. 한 물리 장면에서 공유 시작점 a를 뽑습니다.
2. 후보마다 diagonal/low-rank 계수를 뽑습니다.
3. 같은 에너지 field 형식으로 tau=1까지 실제 적분합니다.
4. 경로 score로 후보별 확률을 정합니다.
5. 실제 tangent와 수용된 시간 간격을 저장합니다.

랜덤 후보의 raw 계수는 경로 안에서 고정됩니다.
그러나 J, G, g, P, 연결 성분과 cap은 현재 z에서 다시 계산합니다.
따라서 effective field는 상태에 따라 바뀝니다.

기본 `candidate_pool: informed`는 **역접촉결합 기반 에너지 field 후보 1개 + 랜덤 후보 7개**입니다.
informed 후보는 normalized 좌표에서 regularized `C=(P A^2 P)^-1`을 사용하고 C의 고유값을 cap으로 제한합니다.
이 후보는 정확한 spectral clipping을 쓰고, 랜덤/모델은 보수적인 spectral upper bound를 씁니다.
같은 C cap을 만족한다는 뜻이지 표현 가능한 모든 행렬이 동일하다는 뜻은 아닙니다.
PBD/Box2D의 최종 위치 label을 가져오는 것은 아니지만, 고전 solver 지식을 사용한 알고리즘 경로 supervision입니다.
이를 순수한 random-only 또는 simulation-free 학습이라고 부르면 안 됩니다.
`--candidate-pool random`으로 고전 후보를 제외한 분리 실험을 할 수 있습니다.

## 5. 경로 score와 FM objective

```text
score_eta = E(z_eta(1))
          + integral_weight * integral_0^1 E(z_eta(tau)) dtau
          + proximity_weight * ||z_eta(1) - proposal||_M^2
          + terminal_violation_weight * max_c(-e_c(z_eta(1)))

w_eta = softmax(-score_eta / T)       # same starting scene only
L_FM  = sum_eta w_eta * integral ||u_theta - u_eta||^2 dtau
```

기본 proximity_weight는 0입니다. 운동에서의 이동 거리를 주 목표로 추가하지 않았습니다.
terminal_violation_weight는 최종 최대 관통을 평가하는 score 항이며, 특정 teacher 위치에 대한 MSE가 아닙니다.
에너지와 최대 잔차의 단위가 다르므로 이 계수와 T는 장면 스케일을 바꾸면 다시 점검해야 합니다.

한 경로의 모든 시각은 같은 경로 가중치를 사용합니다.
각 학습 record에는 실제 accepted dtau도 곱해, 작은 시간 간격이 많은 경로가 과대대표되지 않게 합니다.
정답은 실제 reference tangent이며 endpoint-source나 secant로 대체하지 않습니다.
`--weighting uniform`은 온도를 크게 근사하는 대신 정확히 1/R의 후보 확률을 사용합니다.

FM의 이상적인 제곱오차 해는 현재 (z,tau,condition)에서의 가중 조건부 평균 tangent입니다.
시작 분포는 proposal 주변의 noise 분포이고,
중간 분포와 도착 분포는 가중 후보 경로가 만드는 분포입니다.
이 분포가 자동으로 feasible 상태의 분포가 되는 것은 아닙니다.
참조 후보가 문제를 못 풀면 FM loss 감소만으로 그 문제가 해결된다고 할 수 없습니다.
제한된 신경망, hard contact gate, 수치 적분과 유한 표본 때문에 일반 CFM의 이상적인 보장을 그대로 주장하지 않습니다.
noise 없는 단일 proposal에서의 결정론적 추론도 시작 분포 전체의 샘플링과 구분해야 합니다.
[Flow Matching 원논문](https://arxiv.org/abs/2210.02747)의 조건부 field/주변 확률경로 관점과 연결되는 학습 구성입니다.

`val_fm = weighted tangent MSE / RMS_train^2`입니다.
단일 RMS 스케일로 loss만 정규화하고 실제 field를 차원별로 rescale하거나 평균을 더하지 않습니다.
epoch 간 비교를 위해 val_fm과 RMS는 고정된 최종 온도 `temperature_end`를 사용합니다.
훈련 샘플링은 현재 epoch의 온도를 사용하며, 평가의 참조 경로 통계는 checkpoint에 저장된 그 온도를 사용합니다.

## 6. 적분, 수렴, native PBD 비교

FM/reference 적분은 [0,1]에서 adaptive Euler를 사용합니다.

- 실제 gradient에 대한 하강 방향 확인.
- 최대 입자 이동량 제한.
- Armijo 검사 실패 시 보폭을 절반으로 감소.
- 수용한 실제 보폭만큼 tau 전진.

추론은 tolerance 충족 또는 tau=1에서 끝납니다.
참조 경로는 시간 적분 가중치를 맞추기 위해 tau=1까지 기록합니다.
이미 tolerance를 만족하고 제안 이동이 좌표 한 ULP 이하이며 계산된 E가 비증가이면 roundoff step을 허용합니다.
그때도 실제 nonzero tangent를 보존하고 `accepted_roundoff`를 기록합니다.
수치적으로 불완전한 참조 경로를 조용히 버리지 않습니다.

`first_tolerance_*`는 처음 허용오차에 들어간 시점과 실제 그때까지의 비용입니다.
이후에도 계속 허용오차 안에 머물렀다는 의미는 아닙니다.
최종 성공률과 함께 확인해야 합니다.
`slop=.005, tolerance=.001`이면 실제 관통 약 .006까지 허용될 수 있습니다.

새 `pbd.py`는 mass-weighted Gauss-Seidel **본래 위치 보정 반복**입니다.
각 접촉의 현재 기하를 다시 계산해 unit correction을 적용하며, tau ODE로 바꾸지 않습니다.
PBD에는 FM gain을 곱하지 않습니다.
[PBD 원논문](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf)의 접촉 projection 비교군입니다.
XPBD의 compliance·마찰·회전까지 포함한 엔진 구현은 아닙니다.

유한 PBD sweep 예산 소진도 FM의 tau=1 미수렴과 같이
`completed=True, converged=False`로 기록하고 rollout을 계속합니다.
PBD는 별도 `budget_exhausted`를 기록합니다.
수치 오류나 FM 적분 시간 미완료는 실패로 분리하며 해당 world를 중단합니다.
숨겨진 PBD fallback, 상태 reset, tau 재시작은 없습니다.

## 7. 진단과 실험 설정

기본 입력은 5개 원, train128/val32 장면, 장면당 8개 후보입니다.
수직 stack·기울어진 stack·자유 낙하를 분리해 기록합니다.
외부 데이터는 source/radius와 선택적 scene_type만 읽고 target을 학습에 사용하지 않습니다.
procedural 생성도 설정된 ground/domain을 반영합니다.

`reference_summary.json`에는 다음을 기록합니다.

- 전체 후보 성공률, 성공 후보가 하나 이상인 장면 비율.
- 성공 후보에 실리는 실제 가중 확률과 ESS.
- clean proposal의 초기 위반/해결 그룹.
- 실제 noisy 시작점의 초기 위반/해결 그룹.
- 장면 유형별 지표, 에너지·잔차 감소량, 최저 score가 실패 후보를 고른 횟수.
- 최초 tolerance 도달 시간/NFE, 전체 후보 NFE/backtracking/생성 시간.
- 장면별 최선의 참조 후보: 동일한 noisy 시작점 기준이며 **모든 후보 생성 비용을 청구**.

이 oracle은 clean-proposal rollout의 무료 속도 비교군이 아닙니다.
자유 낙하 성공률로 stack 실패를 가리지 마세요.
학습 전 `min_violating_scene_coverage`로 clean 위반 장면의 후보 coverage를 검사합니다.
기본 기준은 train/val 각각 50%이며, 미달 시 cache/진단은 남기고 본훈련을 시작하지 않습니다.
`--prepare-only`는 진단용이므로 coverage가 낮아도 완료된 cache를 저장합니다.

체크포인트는 FM/energy 양쪽 모두 같은 물리 기준으로 고릅니다:
적분 실패 최소 -> tolerance 성공 장면 최대 -> 최종 에너지 최소.
FM loss만으로 best checkpoint를 고르지 않습니다.

평가는 초기 위반 여부·장면 유형별 성공률, 잔차, NFE/backtracking, 실제 실행 시간을 기록합니다.
PBD는 sweep/contact evaluation 비용을 별도로 보고하며 NFE 1회와 sweep 1회를 같은 비용으로 취급하지 않습니다.
기존 렌더러로 실제 solver 경로 PNG를 만들고, 기본 rollout은 300 물리 스텝입니다.
완료 스텝 수, 미수렴 횟수, 관통, 과분리 gap, 보정이 추가한 운동 에너지도 기록합니다.

## 8. 실행 순서와 분리 실험

필수: PyTorch, PyYAML, 렌더용 Pillow. 동일 구조의 JSON 설정도 지원합니다.

```powershell
# 체크포인트/cache 없이 실행. 단일·3/5-stack·접촉 활성화 사례.
python eval_projection.py --config configs/eval_projection.yaml --preflight

# gain/normalization/cap 및 경로 품질을 확인한 후 cache 생성.
python train_projection.py --config configs/train_projection.yaml --prepare-only

# FM 본훈련과 평가.
python train_projection.py --config configs/train_projection.yaml
python eval_projection.py --config configs/eval_projection.yaml
```

preflight는 gain [1,4,16,32,64], normalization [none,diagonal],
cap [16,128]을 비교합니다. 무관한 baseline 축은 중복 실행하지 않습니다.
고전 gradient/diagonal/Jacobi는 본래 수식을 유지하고 gain만 적용합니다.
isotropic/inverse는 공통 normalized C 설정을 사용합니다.
PBD는 별도의 sweep [4,16,64]로 비교합니다.

```powershell
# 같은 field 구조의 energy-only objective 비교군.
python train_projection.py --config configs/train_projection.yaml --objective energy

# 학습된 대각선만으로 충분한지.
python train_projection.py --config configs/train_projection.yaml --rank 0

# 경로 score 가중치의 효과.
python train_projection.py --config configs/train_projection.yaml --weighting uniform

# 고전 후보의 효과: 먼저 경로 품질 검사.
python train_projection.py --config configs/train_projection.yaml --candidate-pool random --prepare-only
```

옵션별 기본 출력 폴더에 suffix가 붙습니다. 명시적인 `--outdir`도 가능합니다.
기존 best/last checkpoint는 덮어쓰지 않습니다.
다른 run을 평가할 때 `--checkpoint`와 `--path-cache`가 같은 실험을 가리키게 하세요.
cache의 생성 설정을 체크포인트와 비교하며, 참조 통계는 해당 checkpoint epoch의 T/weighting으로 계산합니다.

```powershell
python eval_projection.py --config configs/eval_projection.yaml --checkpoint runs/contact_fm_v2_energy/best.pt --path-cache runs/contact_fm_v2_energy/paths.pt
```

gain/cap/normalization을 바꾼 checkpoint에는 같은 flow 설정의 eval YAML이 필요합니다.
모델 rank 변경은 참조 후보 rank를 바꾸지 않으므로 같은 경로에서 모델 결합 구조만 비교합니다.
energy 비교군도 같은 생성 경로 상태 분포를 쓰며, 경로 생성 비용까지 없앤 baseline은 아닙니다.
energy_steps만큼 추가 field 호출이 필요하므로 같은 epoch/batch 수가 동일 FLOP 비교를 뜻하지는 않습니다.

## 9. 현재 기본값을 선택한 근거와 한계

v1의 약 28% 후보 성공은 모두 자유 낙하 유형에서 나왔고 stack에서는 성공 후보가 없었습니다.
따라서 v2에서는 다음을 분리 검증했습니다.

- random-only 후보는 gain을 매우 크게 해도 5-stack 수렴 비용/적분 완료 문제가 남았습니다.
- informed + diagonal normalization + gain16 + cap128에서는
  seed42/43의 train128/val32 모든 장면에 성공 후보가 생겼습니다.
- 이 검사에서 전체 1,280개 경로가 완료됐고 최대 NFE는 train372/val404였습니다.
  참조 max_steps 기본값은 여유를 둔 1024입니다.
- 기존 score는 성공 후보가 있어도 미수렴 후보를 선택하는 장면이 train20/val5개였습니다.
  terminal_violation_weight=.1에서 둘 다 0개로 줄었습니다.
- T=1e-4 -> 3e-5로 낮추면 성공 경로 가중치가 커졌습니다.

중요: 이 설정의 ESS는 어려운 장면에서 대체로 1에 가까워,
**informed 후보를 중심으로 학습하는 성격이 강합니다.**
이것은 참조 경로 품질 개선이지, FM 학습의 우위나 독립적인 novelty를 증명한 결과가 아닙니다.
random/uniform/energy/diagonal/native-PBD 비교에서 이 사실을 분리해야 합니다.
GPU 본훈련과 여러 seed에 걸친 모델 성능 검증은 별도로 필요합니다.

남은 한계:

- 단위 solver 시간에 모든 배치가 정확히 도착하는 유한시간 수렴 보장은 없습니다.
- 접촉 에너지 감소는 전체 물리 rollout 안정성 보장이 아닙니다.
- 위치·속도 전체의 Box2D 동역학, 마찰·회전·반발계수를 재현하는 목표가 아닙니다.
- free proposal과 solver 선분 중간의 tunneling을 완전히 막는 swept CCD는 없습니다.
- dense 후보 pair/Jacobian 기반 toy 구현입니다. 큰 3D용 sparse broadphase는 후속 작업입니다.
- 동적 graph gate, 특정 배치의 stationary point, 잘못 정의된 원 중심 접촉 등의 한계가 있습니다.
- 학습된 모델의 on-policy 데이터 수집, 광범위한 충돌 분포와 실제 solver 효율 검증이 남아 있습니다.

## 10. 코드 위치와 테스트

| 파일 | 역할 |
|---|---|
| src/contact_flow/dynamics.py | 자유 운동, condition, FD 속도 |
| src/contact_flow/physics.py | 접촉 E/J/G, 공유 FlowConfig, 정규화·cap, analytic field |
| src/contact_flow/solver.py | adaptive tau 적분, 감소 검사, 최초 수렴 비용 |
| src/contact_flow/pbd.py | 본래 mass-weighted PBD sweep |
| src/contact_flow/paths.py | 후보 생성, tangent/시간 가중치, v2 cache |
| src/contact_flow/diagnostics.py | 장면별 성공 확률·ESS·유료 oracle 진단 |
| src/contact_flow/model.py | 메시지 전달, rank0/low-rank C head |
| train_projection.py | 학습, 품질 gate, 비교군, 체크포인트 |
| eval_projection.py | gain/cap/normalization preflight, PBD·FM 평가, 렌더 |

```powershell
python -m unittest discover -s tests -v
```

단위 테스트 통과와 짧은 CPU smoke는 구현 검증입니다. 수렴 성능이나 논문 기여의 증명은 아닙니다.

v2 연결 검증에서는 테스트 87개, 기본 참조 설정의 128/32 장면 생성,
소형 네트워크의 2-epoch 학습, checkpoint/cache 평가, PNG 렌더링을 확인했습니다.
한 수직 stack에서 FM/inverse/native PBD 모두 300 물리 스텝을 수치 오류 없이 실행했습니다.
그러나 짧게 학습한 FM은 그중 14번의 projection에서 tolerance를 놓쳤고,
세 장면 one-step 평가에서도 자유 낙하만 성공했습니다.
이는 충분히 학습된 solver 성능이나 장기 안정성을 검증한 결과가 아닙니다.
