# D 기반 조건부 해 분포 학습

하나의 고정 QP에서 **서로 다른 정확한 multiplier 해**를 생성하는 실험이다.
기존 D의 전역 attention, residual 입력, analytic head, Euler 적분을 그대로 사용한다.
중복 접촉이 있는 합성 QP를 사용하며, 생성된 해들은 모두 같은 물리적 위치를 표현한다.
여러 물리적 미래나 시뮬레이터의 불확실성을 생성하는 실험은 아니다.

## 실행

설정은 기존 `configs/multiplier_cfm_ablation.yaml`의 `distribution` 절에 있다.
배치 크기·학습률·gradient clipping은 상위 `train` 설정을 사용한다.
기본값은 배치 512, 학습률 0.0003, 10,000 optimizer updates다.
별도 데이터 준비 명령 없이 학습 시 QP를 만들고 target을 계속 다시 샘플링한다.

```powershell
# 조건부 분포 학습과 평가
python train_multiplier_distribution.py --variant D_distribution --device cuda
python eval_multiplier_distribution.py --variant D_distribution --device cuda

# 동일 구조·초기 분포·QP에서 하나의 정확한 해로 학습하는 비교군
python train_multiplier_distribution.py --variant D_distribution_point --device cuda
python eval_multiplier_distribution.py --variant D_distribution_point --device cuda

# FM 이후 PGS가 분포를 어떻게 바꾸는지도 평가
python eval_multiplier_distribution.py --variant D_distribution --device cuda --hybrid --output runs/multiplier_cfm_ablation/D_distribution/cfm/eval_test_hybrid.json
```

기본 평가: 새로운 test QP 48개, QP마다 256개 독립 source, K=4/8/16/32.
학습 QP는 768개, validation QP는 24개이며 split별 seed가 다르다.
QP마다 질량·접촉별 총 multiplier·접촉 순서를 샘플링한다.

```powershell
# 새 seed 실험: 학습·평가에 같은 seed와 출력 root를 지정
python train_multiplier_distribution.py --variant D_distribution --seed 43 --outdir runs/distribution_seed43 --device cuda
python eval_multiplier_distribution.py --variant D_distribution --seed 43 --outdir runs/distribution_seed43 --device cuda

# 평가 표본 수와 적분 예산 확대; 학습 조건은 유지
python eval_multiplier_distribution.py --variant D_distribution --calls 4 8 16 32 64 --samples-per-qp 512 --output runs/multiplier_cfm_ablation/D_distribution/cfm/eval_test_extended.json --device cuda
```

`--outdir`은 variant가 붙기 전 root다. 기존 결과가 있으면 덮어쓰지 않고 종료한다.
새 학습에는 새 root, 추가 평가에는 새 `--output`을 지정한다. 자동 resume는 지원하지 않는다.
`--max-updates`와 `--batch-size`로 학습 실행을 줄일 수 있지만 validation 크기는 YAML에서 조절한다.

## 정확한 정답 분포의 구성

기본 Jacobian `J0`는 첫 물체의 바닥 접촉과 이웃 물체 간 상대 접촉으로 이루어진
가역적인 1차원 chain이다. `W`는 양수 inverse mass, `mu`는 양수 벡터다.

```text
A0 = J0 W J0^T
c0 = -A0 mu
```

각 접촉 행과 gap을 `r`개씩 복제하고 접촉 순서를 섞는다.
그룹 g 안의 multiplier 합이 `mu[g]`인 모든 비음수 벡터가 정확한 해다.

```text
sum(lambda[g, :]) = mu[g]
lambda[g, :] >= 0
c + J W J^T lambda = 0
```

따라서 KKT를 만족하고 `x = p + W J^T lambda`도 모두 같다.
기본 접촉 수는 2/4/8개, 복제 수는 2/3개다. 모든 기본 접촉이 최적점에서 활성화된다.

`D_distribution`의 target은 그룹마다 독립적인 균등 simplex 분포다.

```text
e_i ~ Exponential(1)
lambda_i = mu[g] * e_i / sum_{j in g}(e_j)
```

이 분포는 해 집합 위에 실험자가 정한 분포이며, 자연적으로 유일하게 정해지는 물리 분포가 아니다.
정답 생성에 PGS나 다른 수치 solver의 출력·궤적을 사용하지 않는다.
`D_distribution_point`는 각 그룹에 균등 배분한 하나의 정확한 해 `mu[g]/r`로 학습한다.

## Source와 loss

두 variant 모두 source는 QP 입력에서 생성한다.

```text
lambda0_i ~ Uniform(0, source_scale * length_scale / (J W J^T)_ii)
lambda1 ~ target_distribution(. | QP)
t ~ 기존 train 시간 샘플링 규칙
lambda_t = (1-t) lambda0 + t lambda1
L = L_CFM + inner_weight * L_physical(Phi_K(lambda0), lambda1)
```

- 매 학습 batch에서 source와 target을 독립적으로 새로 샘플링한다.
- 원래 CFM field loss와 동일한 접촉별 단위 정규화를 사용한다.
- Inner rollout도 같은 random source에서 시작한다. 기본 K는 1/2/4 중 무작위다.
- Physical loss는 KKT residual과 복원 위치 오차다. 모든 정확한 multiplier가 같은 위치를
  만들므로 위치 loss가 특정 multiplier 배분으로 모이도록 요구하지 않는다.
- 별도의 sample 간 repulsion이나 multiplier endpoint MSE는 추가하지 않는다.
- 모델 입력에는 `mu`, 그룹 ID, target을 넣지 않는다. 그룹 정보는 정답 생성과 평가에만 사용한다.
- 한 번의 모델 입력은 하나의 multiplier 상태다. Batch 안의 다른 샘플과 attention하지 않는다.

동일 seed의 두 variant는 모델 초기값, QP, source, 시간·inner 예산 샘플링을 공유한다.
Target RNG를 따로 두어 target 방식이 달라도 다른 샘플링 순서가 바뀌지 않는다.
분포 재현은 학습으로 검증할 가설이며, noise 입력만으로 보장되지 않는다.

## 평가 읽는 법

모든 분포 지표는 **각 QP 내부**에서 먼저 계산하고 QP별 평균을 보고한다.
여러 QP의 샘플을 섞어서 생기는 다양성을 조건부 다양성으로 계산하지 않는다.

| 지표 | 의미 |
|---|---|
| `success_rate` | 전체 샘플 중 적분을 완료하고 기존 projected KKT 판정 기준을 만족하는 비율 |
| `position_rmse` | 해석적 정답의 위치 대비 질량 가중 RMSE |
| `sliced_w1` | 여러 무작위 방향으로 투영한 생성·정답 경험 분포의 Wasserstein-1 거리. 작을수록 좋음 |
| `marginal_w1` | multiplier 좌표별 경험 분포 거리 |
| `variance_ratio` | 생성 분산 / 정확한 균등 simplex 분산. 한 점으로 모이면 0, 목표는 대략 1 |
| `group_total_relative_error` | 각 중복 그룹의 총 multiplier 오차 |
| `mean_coordinate_bias` | 정확한 분포 평균 대비 좌표별 편향 |

분포 거리 계산 시 multiplier를 **정확한** 그룹 총량으로 정규화한다.
예측한 총량으로 다시 나눠 오차를 숨기지 않는다. 실패한 적분의 마지막 유한 상태도 분포 지표에
포함하며 실패 개수를 따로 기록한다. 다양성이 커도 KKT가 나쁘면 해 분포 학습에 성공한 것이 아니다.
정확도는 저장된 원본 FP64 QP로 다시 계산하며, solver는 FP32로 실행한다.

비교 결과:

- `reference_mc`: 독립적인 두 정답 표본 사이의 유한 표본 오차 수준. 거리가 정확히 0일 필요는 없다.
- `collapsed_exact`: KKT는 정확하지만 분산이 0인 해석적 비교군.
- `source`: 적분 전 초기 분포.
- `local_k*`: 학습 없는 analytic head와 같은 적분 시계. D의 학습이 추가한 효과를 확인한다.
- `pgs`: 같은 random source에서 허용 오차까지 PGS 실행.
- `cfm_k*_raw`: K회 FM만 실행. K는 NFE와 같다.
- `cfm_k*_hybrid`: 선택 사항. 같은 QP에서 FM 후 PGS로 마무리한다. PGS가 배분 분포를 바꿀 수 있다.

Point 비교군도 평가 target은 균등 simplex 분포다. 단일 정답 학습이 얼마나 collapse하는지 비교하려는 의도다.
Timing은 chunk마다 warmup 1회 후 반복 실행한 중앙값의 합이다. GPU는 동기화한다.
QP 생성·전송·지표 계산·렌더링 시간은 제외한다. 샘플당 시간은 batch 처리량을 나눈 값으로,
실시간 엔진의 단일 프레임 지연 시간과 다르다. NFE 합계와 PGS sweep 합계도 JSON에 저장한다.

## 체크포인트와 출력

기본 경로: `runs/multiplier_cfm_ablation/<variant>/cfm/`.

- `experiment.json`: 전체 설정, 정답 구성, checkpoint 선택 규칙.
- `history.json`: 학습 loss, validation 정확도·분포 지표, 누적 시간·field 평가 수.
- `best_distribution.pt`: validation KKT 성공률 우선, 동률이면 sliced W1이 작은 checkpoint.
- `best.pt`: 고정 validation CFM loss가 가장 작은 checkpoint. 이 보조 loss는 t=0.5에서 측정한다.
- `last.pt`: 마지막 checkpoint.
- `eval_test_best_distribution.json`: QP별 지표, 집계, solver 계산량·시간, checkpoint 정보.
- 같은 이름의 `.md`: 요약 표. `.png`: 정확도·분포 거리·한 QP의 분포 그림.
- 같은 이름의 `.pt`: 원본 QP, 그룹 정보, source·정답·각 방법의 생성 샘플.

새 checkpoint format은 기존 D의 단일 정답 실험과 구분한다. 이 실험은 새로 학습해야 한다.
기존 D의 pair cache, 학습 스크립트, rollout 경로는 사용하지 않는다.
Matplotlib 렌더링은 별도 프로세스에서 실행해 solver와 plotting runtime을 분리한다.

## 검증과 해석 범위

```powershell
python -m unittest discover -s tests -p "test_multiplier_distribution.py" -v
```

테스트는 정확한 여러 KKT 해·동일 복원 위치, simplex 통계, label과 독립적인 prior,
조건부 collapse 검출, random-source inner gradient, split·비교군 재현성,
checkpoint 격리, 학습·평가·그래프 저장을 검증한다.

이 합성 실험은 D가 비유일한 해 집합 위의 지정된 분포를 재현할 수 있는지 확인하는 첫 단계다.
실제 접촉 데이터의 비유일성, 여러 물리적 궤적, PGS 대비 실시간 이점은 별도로 검증해야 한다.
