# Multiplier flow-map pilot: A / B / C

기존 `contact_flow_v2`를 **보존**하고 별도로 추가한 실험이다.
현재 범위는 **고정 법선·고정 접촉 집합의 최소 이동 projection**이다.
일반 PBD의 특정 순서가 만든 끝점을 정답으로 강제하지 않는다.
전체 nonlinear PBD/XPBD 대체, restitution, 마찰, CCD, 물리 rollout은 아직 이 모듈의 범위가 아니다.

## 1. 실험 구성

| 실험 | 하는 일 | 학습 |
|---|---|---|
| A | 참조 흐름, dual PGS, 작은 독립 active-set QP 검증 | 없음 |
| B (`endpoint`) | 여러 시작 상태·시간 간격의 구간 끝점 위치 회귀 | 있음 |
| C (`map`) | B와 동일한 데이터·모델에 Lagrangian map matching 추가 | 있음 |

B와 C는 같은 cache, 초기 파라미터 seed, epoch별 샘플 순서, 모델 구조를 사용한다.
동일 epoch는 동일 데이터 노출이며 **동일 학습 FLOP/시간은 아니다**.
기존 `train_delta.py`는 full-state next-step 예측이며 B 또는 projection Delta+PBD가 아니다.

## 2. 물리 문제와 참조 흐름

```text
p = 중력/감쇠까지 적용한 원래 자유 운동 proposal (solver 동안 고정)
C0 = proposal에서의 접촉 gap + slop
J0 = proposal에서 고정한 접촉 Jacobian
W  = 좌표별 inverse mass
D  = J0 W J0^T

z(lambda) = p + W J0^T lambda
Q(lambda) = 0.5 lambda^T D lambda + C0^T lambda
lambda >= 0

eta = eta_fraction / lambda_max(D)
T(lambda) = max(lambda - eta * (D lambda + C0), 0)
b(lambda) = T(lambda) - lambda
d lambda / d h = b(lambda)
```

Q는 겹침 에너지가 아니라 최소 질량가중 이동 문제의 dual 목적이다.
Q는 음수가 될 수 있고, Q=0은 종료 기준이 아니다.
정확한 연속 흐름에서는 `grad(Q)^T b <= -||b||²/eta`이고 lambda는 비음수다.
이것은 매 순간의 관통 에너지 감소 또는 물리 rollout 안정성 보장이 아니다.

접촉 슬롯은 near-contact 전체를 한 번 고정한다. 이미 분리되었더라도
누적 lambda를 줄여야 할 수 있으므로 위반 여부만으로 슬롯을 삭제하지 않는다.
PGS도 누적 lambda의 **음수 증분**을 허용한다. 기존 `contact_flow/pbd.py`의
위반 접촉만 밀어내는 native PBD와 다르다.

참조 구간은 float64의 adaptive step-doubled SSP RK3로 적분한다.
서브스텝은 1 이하라 SSP 구조의 비음수 성질을 유지한다.
수렴 tolerance에 들어가도 지정된 h까지 적분한다. **유한 시간 label을 QP 해로 바꾸지 않는다.**
참조 적분기, PGS, tiny active-set oracle은 서로 다른 알고리즘으로 검사한다.
active-set 열거는 접촉 12개 이하의 CPU 검증용이지 빠른 대규모 solver가 아니다.

## 3. 모델과 정확히 어떤 loss를 비교하는가

모델은 접촉을 노드로 보고 정규화한 D의 결합으로 메시지를 주고받는 작은 네트워크다.
현재 multiplier, gap, C0, eta와 질량 대각성분, 시간 h를 받는다.
접촉 개수와 입자 개수가 다른 문제를 padding/mask로 함께 처리한다.
아직 dense D를 쓰는 toy 구현이며 sparse 3D 성능을 주장하지 않는다.

```text
F_theta(lambda,h) = exp(-h)*lambda + (1-exp(-h))*candidate_theta
candidate_theta  = max(T(lambda) + learned_residual, 0)
```

따라서 `F(lambda,0)=lambda`, `F>=0`이지만 `F-lambda`는 음수도 가능하다.
이미 정확히 고정점인 연결 성분은 그대로 유지한다.
0으로 초기화한 residual head는 frozen-T exponential step에서 출발한다.
**이 analytic bias는 B/C 양쪽에 동일하게 들어간다.**

```text
L_segment = mean mass-weighted position error(z(F), z(lambda_ref(h))) / length_scale²
L_map     = mean ||(partial_h F - b(F)) / multiplier_scale||²
multiplier_scale_c = length_scale / D_cc

L_B = L_segment
L_C = L_segment + matching_weight * L_map
```

L_map 평균은 유효 접촉만 포함한다. D/scale은 각 문제에서 고정되어 있다.
구간 끝점 loss는 lambda MSE가 아니다. 중복 접촉으로 lambda가 비유일해도
같은 실제 위치를 만드는 해를 불필요하게 벌하지 않는다.

`torch.func.jvp`로 h 방향 미분을 계산한다. C의 역전파에는 파라미터-시간
혼합 2차 미분이 있으며, `b(F)`도 F에 대해 미분한다. 임의 detach는 하지 않는다.
전체 위치 Hessian은 만들지 않는다. 추론에는 보통 forward만 필요하다.
이는 [FMM §3.2–3.3](https://arxiv.org/html/2406.07507v2)의 알려진 field에 대한
Lagrangian map-matching 구성이며, 여러 stochastic CFM 경로의 marginalization
우위를 입증하는 실험은 아니다.

## 4. 데이터와 결과 해석

기본: 3/5개 원 학습·검증, 8/10개 원 test, 별도의 비스듬한 두 제약 release 사례.
train/val/test seed는 서로 다르다. 외력 proposal 계산은 기존 모듈을 재사용한다.
시작 multiplier는 zero/under/over/mixed/near-equilibrium을 섞는다.
교란된 시작점에서도 **같은 p**를 유지하고 참조 흐름을 새로 계산한다.
기본 h는 0.25~256의 log-uniform이고 장면마다 zero-start h=256 label도 있다.
짧은 구간은 의도적으로 미수렴일 수 있다.

`segments_summary.json`은 장면 유형별 최장시간 참조 성공률과 생성 NFE를 기록한다.
train/val의 zero-start 최장시간 label이 projection tolerance를 만족하지 않으면
본훈련은 중단한다. `--prepare-only`와 `--profile-only`는 진단을 위해 허용한다.
실패한 장면을 제거하거나 충분히 수렴한 해로 label을 덮어쓰지 않는다.
기본 slop=.005, tolerance=.001이므로 포함된 접촉의 실제 관통 허용은 약 .006이다.

체크포인트는 B/C 공통인 validation segment-position loss로 선택한다.
이 선택은 map 재현 오차 기준이지 solver 속도/수렴이 최적이라는 뜻은 아니다.

평가는 구분해서 기록한다:

- 유한 시간 참조 map 위치 오차 vs 충분히 수렴한 QP 위치 오차.
- 관통, projected-gradient residual, complementarity, 성공률.
- raw vs Q-guarded 결과와 시간 완료 여부, 개입/재시도/NFE.
- 같은 장치의 reference/PGS/model 실제 시간. 각 방법 warm-up 뒤 반복 평균.

1/2/4/8회 평가는 **같은 총 시간 T의 분할 비교**다. 반복 증가가 반드시 개선을
뜻하지 않는다. 정확한 map은 모든 분할에서 같은 결과를 낸다.
guard는 Q 증가 시 더 작은 h로 모델을 다시 호출하고 수용한 시간만 전진한다.
숨겨진 GD/PBD fallback이나 finishing pass는 없다. 예산 소진은 실패로 남긴다.
PGS sweep과 신경망 NFE는 같은 비용 단위가 아니다.
평가 시간은 cache된 기하·D/eta setup과 학습 데이터 생성 비용을 제외하므로
현재 수치는 **local solve 비용**이다. 전체 시뮬레이터 speedup으로 해석하지 않는다.

## 5. 실행

기존 requirements의 PyTorch>=2.2, NumPy, PyYAML, Pillow를 사용한다.
config는 공통 파일 하나이며 `--objective`로 B/C만 선택한다.

```powershell
# A: 체크포인트 없이 참조 흐름/PGS/독립 QP/상태 갱신 검증
python eval_multiplier.py --preflight

# 같은 segment cache를 한 번 생성
python train_multiplier.py --prepare-only

# 실제 장치에서 C의 mixed-derivative 비용부터 확인
python train_multiplier.py --profile-only --device cuda

# B / C: 같은 cache로 별도 학습
python train_multiplier.py --objective endpoint
python train_multiplier.py --objective map

# 미학습 8/10개 장면; --split val로 분포 내 평가도 가능
python eval_multiplier.py --objective endpoint --split test
python eval_multiplier.py --objective map --split test
```

출력은 `runs/multiplier_flow/`이며 기존 v2 결과와 섞이지 않는다.
`--outdir`, `--config`, `--device`, 학습용 `--epochs`를 지원한다.
기존 checkpoint는 덮어쓰지 않는다. 다른 seed/참조/물리 설정에는 새 outdir를 쓴다.
cache 설정과 checkpoint를 검증하므로 추론에서 eta나 학습 문제를 몰래 바꿀 수 없다.

```text
segments.pt / segments_summary.json   공통 데이터와 참조 품질
preflight.json / renders/             A 검증 및 원 배치 비교
profile_endpoint.json                B/C forward-backward 비용 (해당 실행 장치)
endpoint/{best,last}.pt, history.json B 학습
map/{best,last}.pt, history.json      C 학습
endpoint/eval_test.json               B raw/guarded/고전 비교
map/eval_test.json                    C raw/guarded/고전 비교
```

## 6. 법선 변경은 무엇까지 구현했는가

`relinearize`는 원래 p와 실제 primal 위치를 유지하고 contact ID별 lambda guess를
넘긴다. 새 contact의 guess는 0이고 사라진 contact는 제거한다. 대신 위치와
새 `p+W J^T lambda_guess` 사이의 **stationarity mismatch를 명시적으로 반환**한다.
새 문제의 offset은 `C(z_bar)+J_new(p-z_bar)`다.
이것은 위치가 몰래 바뀌지 않는 bookkeeping 검증이지, mismatch를 해결하는
nonlinear solver 구현이 아니다. 이 결과를 즉시 decode해서 rollout에 넣으면 안 된다.

## 7. 테스트

```powershell
python -m unittest discover -s tests -p test_multiplier_flow.py -v
python -m unittest discover -s tests
```

해석 가능한 유한 시간 해, 과보정 취소, 중복 접촉, 독립 QP, contact permutation,
고정점/항등/비음수, JVP와 혼합 parameter derivative의 유한차분 검증,
guard 실패와 시간 누적, cache 호환성을 검사한다.
