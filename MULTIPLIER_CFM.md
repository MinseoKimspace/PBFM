# Contact projection: Conditional Flow Matching

현재 multiplier 실험은 **CFM 하나**다. 구 B·C, finite-time map 모델,
JVP/Lagrangian matching, 시간 h=256의 참조 ODE label은 제거했다.
기존 contact_flow 실험과 저장된 B·C 체크포인트/결과 파일은 수정하거나 삭제하지 않는다.

## 1. 무엇을 학습하나?

외력과 damping으로 계산한 다음 위치 proposal을 p라고 한다.
한 물리 프레임에서는 접촉 Jacobian J와 inverse mass W를 고정한다.

```text
D = J W J^T
Q(lambda) = 0.5 lambda^T D lambda + c^T lambda
lambda >= 0
x = p + W J^T lambda
```

c는 proposal에서 측정한 gap에 slop을 더한 값이다.
PGS를 qp_tolerance까지 실행해 수렴해 lambda_star를 만든다.
이제 정답은 **그 수렴해**이고, 참조 흐름을 h만큼 진행한 미수렴 상태가 아니다.
양의 질량인 고정 법선 볼록 문제의 최적 위치는 유일하지만, 중복 접촉의
multiplier는 비유일할 수 있다. 이번 label은 정해진 PGS 순서로 얻은 표현이다.

CFM은 다음 경로의 순간 변화율을 학습한다.

```python
source = sampled_nonnegative_multiplier
target = converged_qp_multiplier

tau = sampled_time                  # 0 <= tau < 1
lambda_tau = (1 - tau) * source + tau * target
target_rate = target - source

predicted_rate = model(lambda_tau, tau, original_problem)
loss = mean_valid_contacts(((predicted_rate - target_rate) / scale)**2)
scale = length_scale / D_diagonal
```

- 조건은 원래 proposal에서 구성한 접촉 문제다. 학습/적분 중 p를 바꾸지 않는다.
- 모델에 target이나 별도 clean source를 넣지 않는다.
- QP label은 학습용 감독이다. teacher-free라고 주장하지 않는다.
- 같은 condition에서 도착점이 하나일 수 있다. 임의 noise로 물리 해의 다양성을 만들지 않는다.
- endpoint estimate MSE는 로그용이다. endpoint loss/energy loss/JVP loss를 더하지 않는다.
- parameter에 대한 1차 역전파만 사용한다. time JVP나 dense Hessian이 없다.

## 2. 네트워크와 추론

각 접촉의 lambda, 원래 gap c, 현재 gap c+D lambda, D 대각성분, tau를 입력한다.
정규화한 D를 통해 접촉 간 message passing을 하고, 접촉마다 signed scalar rate를 출력한다.
출력에 projected-gradient 보정이나 알려진 solver step을 더하지 않는다.
서로 연결되지 않은 성분 중 c>=0이고 lambda=0인 성분은 정확히 움직이지 않도록 mask한다.
이것은 free-flight identity이지, 일반적인 최소점 정지/수렴 보장이 아니다.

```python
lam = zeros_like(contact_gap)        # 실제 rollout에는 시작 noise 없음
for s in range(K):
    tau = s / K
    rate = model(lam, tau, problem)
    lam = lam + rate / K            # Euler: tau=0 -> 1
x_next = p + W @ J.T @ lam
v_next = (x_next - x_previous) / dt
```

물리 시간 dt와 계산 시간 tau는 다르다. 중력은 물리 프레임마다 한 번 적용한다.
K는 같은 [0,1] 구간의 Euler 적분 횟수다. K를 늘린다고 수렴/품질이 반드시 좋아지지는 않는다.
적분 중에는 새로운 접촉을 생성하거나 법선을 재계산하지 않는다.

**Raw**는 multiplier clipping, energy projection, PBD finish가 없는 CFM 적분이다.
학습 경로는 lambda>=0이지만, 부정확한 예측/Euler 적분은 음수 multiplier를 만들 수 있다.
이를 숨기지 않고 negative_multiplier와 실패/수렴 지표로 보고한다.

**Guarded**는 별도 진단이다. Euler 후보의 lambda>=0 및 Q 비증가를 확인하고,
실패하면 같은 slope에서 보폭만 절반으로 줄인다. 수용된 시간만 tau에 더한다.
Q 검사는 해 수렴이나 실제 비선형 접촉 에너지 감소를 보장하지 않는다.
backtracking은 slope를 재사용하므로 NFE와 횟수가 다르다.
예산 안에 tau=1에 도달하지 못하면 실패이며, 그 중간 결과로 다음 물리 프레임을 만들지 않는다.
GD/PGS fallback이나 마무리 solver는 없다.

## 3. 데이터와 예산

| 항목 | Pilot | Large |
|---|---:|---:|
| Train 3개 / 5개 장면 수 | 각 24 | 각 512 |
| Val/Test 크기별 장면 수 | Val 8 / Test 6 | 각 48 |
| 각 split의 release / floor | 4 / 8 | 128 / 64 |
| 장면당 source 수 | 12 | 36 |
| Train source-target 쌍 | 720 | 43,776 |
| batch size | 64 | 128 |
| 학습 예산 | 100 epochs | 30,000 updates |
| hidden / message steps | 64 / 3 | 64 / 3 |

기존 수직 stack/비스듬한 stack/자유낙하와 두 halfspace의 보정 취소 문제를 유지한다.
추가 floor 장면은 작은 resting 오차부터 큰 관통까지 log scale로 생성한다.
일부는 이미 유효해서 0 보정이 정답이다. 동일한 평가 rollout을 복사한 데이터는 아니다.

source 종류는 zero / under / over / mixed / near / solved이며 균형 있게 포함한다.
모든 source는 **같은 원래 p의 동일한 수렴해**를 목표로 한다.
solved source는 target과 같아서 0 field를 감독한다.
tau는 매 batch 새로 추출한다. 기본 10%는 tau=0, 나머지는 [0,1) uniform이다.
validation은 고정 seed의 동일한 tau 샘플로 평가한다.

train/val/test seed는 분리한다. Train/Val의 stack 크기는 3·5, Test는 8·10이다.
이것은 여전히 제한된 장면 분포이며, 일반적인 물리 장면이나 3D 성능을 보장하지 않는다.
새 CFM은 목표와 floor 데이터도 바뀌었으므로 과거 B·C 수치와의 차이를
오직 matching loss의 효과라고 해석하면 안 된다.

## 4. 실행 순서

pbfm 환경을 활성화하고 프로젝트 루트에서 실행한다.
아래는 모두 Large 설정이며, **학습은 한 번만** 한다.
`--objective endpoint/map`과 `--objectives` 옵션은 삭제했다.

```powershell
# 1. 신경망 없이 수렴 QP label / 독립 oracle / 보정 취소 / 렌더링 검증
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --preflight

# 2. 전체 source-target pair cache 생성 (한 프로세스에서 한 번)
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --prepare-only

# 3. CFM 학습 한 번
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --device cuda

# 4. 정적 projection: validation 및 원 개수 일반화
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --split val --device cuda
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --split test --device cuda

# 5. PGS와 CFM의 실제 300프레임 움직임 / GIF
python eval_multiplier_rollout.py --config configs/multiplier_cfm_large.yaml --device cuda
```

전체 생성/학습 전에 비용만 확인하려면 다음 두 명령을 선택적으로 사용한다.

```powershell
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --prepare-pilot 8
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --profile-only --device cuda
```

pilot은 outdir/prepare_pilot에 별도 저장하며 학습하지 않는다.
profile은 전체 pairs cache가 필요하며 생성하지 않았다면 먼저 생성한다. optimizer update는 하지 않는다.
작은 전체 실험은 모든 명령의 config를 `configs/multiplier_cfm.yaml`로 바꿔 실행한다.

```powershell
# 학습 없이 PGS 움직임만
python eval_multiplier_rollout.py --config configs/multiplier_cfm_large.yaml --pgs-only --device cpu

# 안전 검사는 별도 결과에 기록
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --split test --guarded --device cuda
python eval_multiplier_rollout.py --config configs/multiplier_cfm_large.yaml --guarded --device cuda

# 학습된 다른 checkpoint / 짧은 rollout
python eval_multiplier_rollout.py --config configs/multiplier_cfm_large.yaml --checkpoint best --steps 60 --output runs/cfm_motion_short
```

## 5. 저장 위치와 해석

```text
runs/multiplier_cfm_large/
  pairs.pt, pairs_summary.json       새 CFM cache / 수렴 label 품질
  pairs_parts/                      완료 split을 중단 후 재사용
  preflight.json, renders/           비학습 검증
  profile_cfm.json                   CFM forward/backward 비용
  cfm/
    best.pt                         고정 validation CFM loss 최저
    best_solver.pt                  장면 그룹 균등 raw 수렴률 최선, 동률이면 PG residual
    last.pt, history.json
    eval_val_best_solver.json
    eval_test_best_solver.json
  motion/
    best_solver_cfm/                 rollout JSON, trajectory PT, GIF, PNG
    best_solver_cfm_guarded/         안전 검사 별도 비교
```

평가 기본 checkpoint는 best_solver다. 이는 rollout 선택이 아니라 정적 validation 선택이다.
release 개수가 많아도 checkpoint 선택을 독점하지 않도록
장면 종류·원 개수·시작 모드별 성공률을 균등 평균한다. micro 평균도 별도로 남긴다.
부분 적분 실패는 성공으로 세지 않는다. `converged_count`와 `completed_count`도 따로 기록한다.

- CFM tangent loss, 예측 endpoint MSE, 실제 적분 뒤 projection MSE는 서로 다르다.
- projection 성공은 관통, projected-gradient residual, 음수 multiplier 기준을 함께 만족해야 한다.
- slop=.005를 적용한 뒤 residual tolerance=.001이다.
- rollout 완료와 접촉 수렴은 다르다. 정지 속도 0만으로 올바른 접촉이라고 판정하지 않는다.
- 원래 위치와 최종 위치 사이의 swept 지표는 FM 내부 경로/CCD 검사가 아니다.
- 실제 움직임에서는 매 프레임 새 문제를 만들지만 내부 법선은 고정이다.
  restitution/마찰/CCD 해결, nonlinear 재선형화, warm start, PBD 마무리는 없다.
- timing은 동일 장치에서 비교한다. NFE와 PGS sweep은 동등 비용이 아니다.
  작은 진단 harness의 wall-clock을 최적화된 대규모 solver 속도로 일반화하지 않는다.

기존 `segments.pt`, `multiplier_map_v1` checkpoint는 새 코드에서 거부한다.
이전 run 디렉터리는 지우거나 덮어쓰지 않았다. 새 데이터 생성과 재학습이 필요하다.
학습 checkpoint가 있는 outdir에서는 새 훈련을 거부한다. resume는 지원하지 않는다.
다른 실험은 `--outdir`, rollout 보고서는 `--output`으로 새 경로를 지정한다.

## 6. 구현과 테스트

- `src/multiplier_flow/data.py`: PGS 수렴 label, 여러 source, 새 cache format.
- `model.py`: conditional velocity field와 CFM tangent loss.
- `solvers.py`: PGS/oracle, raw 또는 명시적 guarded Euler.
- `evaluation.py`: QP preflight, K별 정적 평가와 그룹별 수렴 지표.
- `rollout.py`: 외력 -> CFM -> 위치 복원 -> FD, 접촉 변화와 렌더링.
- `problem.py`: 고정 법선 QP, decode, KKT residual. 여기의 analytic field는
  residual/비학습 검증용일 뿐 CFM의 matching target으로 사용하지 않는다.

```powershell
python -m unittest discover -s tests -p "test_multiplier*.py" -v
python -m unittest discover -s tests
```

테스트는 수렴해 label, CFM 식, tau 진행, signed correction, 1차 gradient,
padding/순열, 자유낙하 identity, guard 실패, checkpoint/cache 격리, JSON/GIF를 검증한다.
통과는 구현 검증이며 새 모델의 접촉 정확도나 FM 우월성의 증거가 아니다.
