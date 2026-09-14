# 초기값 스케일링 + 기하 μ: 비학습 preflight

## 지금 실행할 명령

프로젝트 루트에서 다음 **한 명령**만 실행한다. CPU float64 검증이며 GPU,
Box2D 데이터, `pairs.pt`, 학습 checkpoint가 필요 없다.

```powershell
python eval_multiplier.py --config configs/multiplier_homotopy_preflight.yaml --homotopy-preflight --device cpu
```

결과: `runs/multiplier_homotopy_preflight/homotopy_preflight.json`.
같은 명령을 다시 실행하면 이 진단 JSON만 갱신한다. 이전 결과를 보존하려면:

```powershell
python eval_multiplier.py --config configs/multiplier_homotopy_preflight.yaml --homotopy-preflight --device cpu --outdir runs/homotopy_preflight_trial2
```

**이번 단계에는 prepare/train 명령을 실행하지 않는다.** 기존 CFM의 모델,
loss, config, 데이터 cache, checkpoint 형식과 저장 경로는 변경하지 않았다.
기존 `--preflight`는 CFM v2의 QP/head 검증이고, 새 `--homotopy-preflight`와 다르다.

## 무엇을 비교하나

동일한 접촉 문제에서 초기값 3종 × μ 스케줄 2종을 비교한다.

| 초기값 | 정의 | 역할 |
|---|---|---|
| `epsilon` | `epsilon_ratio * gap_scale / D_ii` | 작은 초기값이 만드는 불리한 경로의 비교 기준 |
| `diagonal_scaled` | `sqrt(mu0 / D_ii)` | 자기 접촉의 대각/barrier 스케일을 맞춘 후보 |
| `isolated_root` | `D_ii*lambda_i² + c_i*lambda_i - mu0 = 0`의 양수 해 | 자기 접촉의 c까지 반영한 후보 |

두 후보 모두 다른 접촉을 무시한 초기화다. **전역 centering이나 실제 비관통을
보장하지 않는다.** `initial_actual_gap`, `initial_metrics`, `r0`를 확인한다.
초기화가 보정량을 만들더라도 원래 proposal `p`는 절대 다시 정의하지 않는다.

스케줄은 같은 시작/끝 μ를 사용하는 `linear`와 `geometric`이다.

```text
mu0 = mu0_factor * gap_scale² / mean(D_ii)
mu_min = mu0 * mu_min_ratio > 0
linear:    mu(t) = mu0 + t*(mu_min - mu0)
geometric: mu(t) = mu0 * (mu_min / mu0)^t
```

μ는 마찰계수가 아니다. 모든 질량을 같은 비율로 바꿨을 때 보정 위치의 스케일이
불필요하게 바뀌지 않도록 `mu0`와 초기 lambda의 단위를 함께 맞춘다.
`mu0_factors: [0.1, 1.0, 10.0]`으로 바꾸면 규모별 추가 비교도 가능하지만,
먼저 기본값 `[1.0]`으로 실행한다.

## 참조 경로와 복구 field

고정 법선 QP의 `D, c, p`와 최초 `r0`를 끝까지 유지한다.

```text
r0 = D*lambda0 + c - mu0/lambda0
H(lambda,t) = D*lambda + c - mu(t)/lambda - (1-t)*r0
A = D + diag(mu(t)/lambda²)
rhs = mu_dot(t)/lambda - r0 - kappa*H
u_ref = solve(A, rhs)
```

정확한 연속 계산에서는 `dH/dt = -kappa*H`다. 감소하는 것은 **homotopy 잔차**이며,
물리 에너지나 관통량의 단조 감소를 뜻하지 않는다. `t`는 solver 진행도다.
물리 시간 dt나 FM 재시작 횟수가 아니다.

이번에는 **호길이 재매개변수화를 하지 않는다.** μ 스케줄을 바꾸면 `(1-t)r0`와의
관계도 달라지므로 단순한 속도 변경뿐 아니라 다른 경로를 만드는 비교다.

참조 경로는 균등한 t 표본에서 Newton root solve로 `H=0`을 풀어 생성한다.
이 계산을 Euler 성공으로 세지 않는다. 각 Newton 선형계 풀이, line search,
잔차 평가와 시간을 별도로 기록한다. μ_min은 양수이므로 최종 해도 근사 해다.
원래 QP에 대한 projected-gradient/관통량 기준으로 실제 성공을 판정한다.

## 보고서 읽는 순서

1. `cases`: 단일 관통/분리/경계 접촉, 접촉 보정 취소, 중복 접촉,
   3·5·8·10 stack을 각각 확인한다. 중복 접촉은 multiplier보다 위치/KKT를 비교한다.
2. `reference`: 고정밀 경로가 끝까지 계산됐는지, 마지막 실제 KKT 잔차가 작은지 확인한다.
3. `reference.movement`: 원래 시간축 8구간별 정규화 lambda 이동량 비율.
   마지막 구간 집중은 불리한 스케줄의 징후다. 균등 이동이 적은-step 수렴을 보장하지는 않는다.
4. `recovery`: 중간 경로 상태에 양수성을 유지하는 곱셈 교란을 넣는다.
   r0를 다시 계산하지 않고 RK4 512 step으로 적분한다(검증용 고정밀 계산,
   후보의 4·8·16·32 step 예산과 구분). `H_norm`과 `expected_H_norm`,
   `relative_decay_error`, `differential_identity_error`를 비교한다.
5. `euler.k*_raw`: **같은 [0,1] 구간을 K=4·8·16·32 균등 Euler step**으로 적분한다.
   매번 정확한 참조 field의 선형계를 풀므로 아직 FM 모델의 결과가 아니다.
   음수/0 lambda 또는 비유한 값이 나오면 명시적으로 실패한다. clipping하지 않는다.
6. `euler.k*_positive_guard`: 양수성 때문에 보폭을 줄이는 진단이다.
   `actual_nfe`가 K를 초과할 수 있다. 줄인 보폭을 같은 K-step 성공으로 계산하지 않는다.
   `completed`, `elapsed_solver_time`, `interventions`, `reason`을 함께 본다.
   이 guard는 에너지 하강이나 정밀 수렴을 보장하지 않는다.
7. `pgs_zero`, `pgs_same_initialization`: 같은 KKT tolerance까지 원래 PGS를 실행한다.
   초기화 포함 시간, sweep, 접촉 갱신 횟수를 비교한다.

`summary`는 처음부터 풀려 있던 장면을 제외해 성능을 집계하고 실패 장면 이름도 남긴다.
분리/경계 접촉의 실패도 `initially_solved_failures`로 별도 기록하고 준비 여부 판정에 포함한다.
해당 장면의 자세한 결과는 `cases`에서 확인한다. 참조·복구 준비 여부와
균등 K 예산 성공 여부를 구분한다. `ready_for_learning`은 이 작은 검증에서의
참고 표시일 뿐 학습을 자동 시작하거나 FM의 성공을 보장하지 않는다.

PGS sweep, Newton 선형계 풀이, 미래의 NN 호출은 비용이 서로 다르다.
현재 timing은 동일 CPU의 단일 실행이며 Python overhead를 포함한다. 가속 주장에는
추가 반복 측정과 같은 품질의 solver 비교가 필요하다. 독립 active-set oracle 시간은
진단 비용으로 별도 표기하며 실제 solver 시간에서 숨긴 마무리로 사용하지 않는다.

## 판정과 다음 단계

- 참조 경로 자체가 미수렴하면 μ/초기값/참조 수치 설정을 먼저 점검한다.
- 복구 검사가 실패하면 원래 t에서의 RK4 해상도·양수성·잔차 스케일을 점검한다.
- 참조는 성공하지만 raw K가 실패하면 적분 예산 문제를 분리해서 본다.
- guarded만 성공하면 추가 호출/개입 비용까지 포함한다.
- K=8은 목표 예산이지 절대 조건이 아니다. 전체 K별 정확도–비용 곡선을 본다.
- 아직 새로운 FM head, homotopy 학습 데이터, checkpoint, 물리 rollout은 만들지 않았다.
  결과를 검토한 뒤 학습 설계를 결정한다.

## 회귀 테스트

```powershell
python -m unittest discover -s tests -p "test_multiplier_homotopy.py" -v
python -m unittest discover -s tests -p "test_multiplier*.py" -v
```
