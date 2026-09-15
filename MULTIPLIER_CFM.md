# Contact-structured CFM: 전역 통신과 solver rollout 학습

현재 실험은 고정 접촉 QP를 적은 신경망 호출로 푸는 조건부 Flow Matching(CFM) solver다.
기존 `multiplier_cfm_large`의 CFM과 analytic contact head를 유지하고,
**전역 정보 전달**과 **실제 적분 결과를 통한 학습**의 효과를 분리해서 확인한다.
Homotopy 코드·설정·별도 preflight는 제거했다. 아래 `--preflight`는 접촉 QP와 CFM head 검증이다.

## 1. 비교 실험

새 공통 설정은 `configs/multiplier_cfm_ablation.yaml`이다.
`--variant`가 통신 방식과 loss 사용 여부를 선택한다.

| Variant | 접촉 간 통신 | Inner rollout loss | 교란 rollout loss |
|---|---|---|---|
| A | 기존 국소 메시지 전달 | 없음 | 없음 |
| B | 국소 메시지 전달 + 성분 내부 full attention | 없음 | 없음 |
| C | 기존 국소 메시지 전달 | 있음 | 없음 |
| D | 국소 메시지 전달 + 성분 내부 full attention | 있음 | 없음 |
| D_recovery | D와 동일 | 있음 | 있음 |

먼저 A/B/C/D를 비교한다. `D_recovery`는 교란 학습의 추가 효과를 보는 후속 비교다.
CFM·analytic head·접촉 후보·학습 데이터·기본 입력 특징은 네 모델에 공통이다.
새 비교 실험은 모두 `feature_version: residual`을 사용한다. 기존 설정은 기존 특징을 유지한다.

학습·검증·테스트 모두 3·5·8·10개 물체를 포함하되, 서로 다른 seed로 장면을 생성한다.
현재 검사는 **학습에 포함된 크기에서 새로운 장면을 푸는 능력**을 측정한다.
더 큰 물체 수로의 일반화는 별도 실험이 필요하다.

기본 규모는 train 288개 QP/3,456개 source-target 쌍,
val/test 각각 96개 QP/1,152개 쌍이다. 각 split에 floor 16개, release 16개 문제가 포함된다.
학습 예산은 3,000 optimizer updates다. 충분히 학습된 최종 성능을 보장하는 설정은 아니다.
Rollout 학습은 update당 계산량이 더 크므로 업데이트 수와 학습 시간을 함께 비교한다.

## 2. 고정 접촉 문제와 analytic head

한 물리 프레임의 proposal `p`, 접촉 Jacobian `J`, inverse mass `W`를 고정한다.

```text
D = J W J^T
Q(lambda) = 0.5 lambda^T D lambda + c^T lambda
g(lambda) = c + D lambda
lambda >= 0
x = p + W J^T lambda
```

`c`는 proposal의 signed gap에 slop을 더한 값이다.
KKT 조건은 `lambda >= 0`, `g >= 0`, `lambda_i * g_i = 0`이다.
정답 multiplier는 충분히 수렴한 PGS로 만든다. 따라서 정답을 사용하는 감독 학습이다.
중복 접촉에서는 multiplier가 유일하지 않을 수 있으므로 위치 오차와 KKT 잔차도 평가한다.

신경망은 다른 접촉의 multiplier 변화율 `r`을 예측한다.
각 접촉의 endpoint는 다른 접촉의 예측을 사용한 해석식으로 계산한다.

```python
predicted = lam + (1 - tau) * r
endpoint_i = relu(-(c_i + sum(D_ij * predicted_j for j != i)) / D_ii)
u_theta = (endpoint - lam) / (1 - tau)
```

CFM이 맞추는 것은 `r` 자체가 아니라 analytic head를 포함한 전체 field `u_theta`다.
`r=0`은 다른 접촉의 현재 multiplier를 사용하는 비학습 `local` 비교군이다.
단일 접촉은 이 head만으로 해결된다. 그 결과는 FM 학습의 성과와 구분한다.
누적 multiplier는 비음수지만, 과보정을 해제하기 위한 증분은 음수일 수 있다.

현재 만족된 접촉도 이웃을 보정하기 위해 함께 움직일 수 있다.
자기 잔차가 0이라는 이유만으로 접촉별 출력을 막지 않는다.
Head는 endpoint의 비음수성을 보장하지만 전체 KKT 수렴이나 에너지 감소를 보장하지 않는다.

## 3. 전역 통신

기존 `D` 기반 국소 메시지 전달을 유지한 뒤,
같은 연결 성분에 속한 모든 접촉끼리 attention을 수행한다.
직접 연결되지 않은 접촉도 같은 성분 안에서는 정보를 교환할 수 있다.
정규화된 `D_ij`의 부호와 크기를 attention의 관계 정보로 사용한다.

서로 독립인 성분 사이와 padding에는 attention을 허용하지 않는다.
고정 벽을 공유한다는 사실만으로 서로 다른 물체의 접촉을 연결하지 않는다.
새 residual 특징은 고정된 물리 단위로 정규화하며,
국소 모델에 동적인 전역 요약을 추가하지 않는다.

전역 통신은 보정에 필요한 정보를 전달한다. 그 자체가 오류를 줄이는 규칙은 아니다.
지금은 global token으로 압축하지 않고 작은 문제에 적합한 full attention을 사용한다.
이 때문에 접촉 수가 커지면 attention의 메모리·시간 비용도 따로 확인해야 한다.

## 4. CFM + 미분 가능한 inner rollout

기존 CFM 경로와 target은 유지한다.

```text
lambda_tau = (1 - tau) * source + tau * target
u_target = target - source
L_CFM = mean_valid_contacts(((u_theta - u_target) / scale)^2)
scale_i = length_scale / D_ii
```

Source는 zero/under/over/mixed/near/solved를 포함한다.
신경망 입력에 정답을 넣지 않는다. 학습 시 정답은 경로와 loss 구성에만 사용한다.
기본 시간 샘플의 10%는 `tau=0`, 나머지는 `[0, 0.999)`다.
`tau_min_remaining`은 학습 target의 수치 오차가 작은 남은 시간으로 나뉘는 것을 제한한다.
추론은 여전히 `tau=1`까지 진행한다.

추론과 inner rollout 학습은 같은 미분 가능한 Euler step을 사용한다.

```python
lam = zeros_like(c)
for step in range(K):
    tau = step / K
    endpoint = model.endpoint(lam, tau, problem)
    alpha = (1 / K) / (1 - tau)
    lam = (1 - alpha) * lam + alpha * endpoint
```

이 업데이트는 Euler와 같으며, 비음수인 현재 상태와 endpoint의 convex combination이다.
마지막 step에서 `alpha=1`이고 `tau=1`에서는 field를 호출하지 않는다.
학습에서는 중간 상태를 detach하지 않으므로 앞선 호출의 출력까지 gradient가 전달된다.

```text
L = L_CFM + inner_weight * L_inner + recovery_weight * L_recovery
```

`L_inner`는 실제 zero-start 적분의 최종 projected KKT 잔차와
정답 위치에 대한 질량 가중 오차를 정규화한 loss다.
KKT 항은 `R_eta = lambda - relu(lambda - eta * g)`를
`eta * length_scale`로 나눈 제곱 오차이며, 위치 항은 질량 가중 MSE를
`length_scale^2`로 나눈 값이다. 비음수성은 공통 적분 step이 유지한다.
`train.inner_rollout.calls: [1, 2, 4]`는 매 update 하나의 K를 균등하게 선택한다.
CFM 샘플링과 별도 RNG를 사용해 비교 모델의 학습 샘플 순서를 유지한다.
`--profile-only`는 선택한 variant의 전체 loss에 대한 forward/backward 비용을 측정한다.

## 5. 교란 후 남은 적분으로 복구

평가는 중간 상태까지 적분하고 교란을 넣은 뒤 **같은 QP, 같은 시간 격자**에서 남은 호출을 수행한다.
예를 들어 K=4의 `tau=0.5`에서 교란하면 이후 호출 시간은 0.5와 0.75다.
시간을 0으로 재시작하지 않는다.

독립적인 교란과 접촉 구조를 따른 상관 교란을 사용한다.
`amplitude * length_scale`은 요청한 물리 위치 RMS 크기다.
비음수 경계와 multiplier 변화량 제한 때문에 실제 교란은 더 작을 수 있어 실제 크기도 기록한다.
`D_recovery`는 이 절차의 최종 KKT·위치 오차를 학습한다.
이 variant의 가중치는 `train.recovery.enabled_weight`로 조절한다(기본 1).
C/D의 `train.inner_rollout.weight`는 양수여야 하며, 잘못된 0 설정은 거부한다.
K=4에서 중간 step을 선택하며, prefix와 suffix의 gradient 연결을 유지한다.
별도의 recovery velocity 정답을 만들지 않는다.

교란 평가는 다음을 함께 본다.

- 교란이 추가한 물리적 위치 차이가 남은 호출 후 얼마나 줄었는가?
- 최종 KKT·위치 오차가 실제로 작아졌는가?
- 같은 난수와 교란 설정에서 비학습 local head와 PGS는 어떤 결과를 내는가?
- 교란 크기와 남은 호출 수는 얼마인가?

복구 비율만으로 학습 효과를 판단하지 않는다. Analytic head 자체도 교란을 줄일 수 있다.
CFM과 local의 중간 상태가 다르면 clipping 뒤 실제 교란도 다를 수 있다.
PGS는 각 중간 상태와 그 교란 상태에서 tolerance까지 풀며, 가상의 FM 시간을 부여하지 않는다.
물리적 크기가 거의 0인 multiplier 교란은 비율의 분모가 불안정하므로 별도로 다룬다.
같은 구조의 반복 회귀 objective와 비교하는 실험은 아직 추가하지 않았다.
따라서 이 비교만으로 CFM objective 자체의 우월성을 주장하지 않는다.

## 6. 실행 명령

아래 명령은 저장소 루트에서 실행한다. CUDA가 없다면 `--device cpu`를 사용한다.
공통 QP cache는 `runs/multiplier_cfm_ablation/shared_data`에 한 번 생성하고 재사용한다.
체크포인트·학습 로그·평가 결과는 `runs/multiplier_cfm_ablation/<variant>`로 분리한다.
처음에는 `--prepare-only`를 완료한 뒤 각 학습을 시작한다. 준비 중 다른 프로세스가
같은 cache를 쓰려고 하면 잠금 오류로 중단하며, 완성된 cache는 동시에 읽을 수 있다.

```powershell
# QP, 단일 접촉 head와 정확한 coupling 표현을 검증한다.
python eval_multiplier.py --config configs/multiplier_cfm_ablation.yaml --preflight

# 공통 데이터 준비. 네 모델마다 정답을 다시 생성할 필요가 없다.
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --prepare-only

# 먼저 전역 모델과 rollout loss를 합친 계산량을 확인한다.
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant D --profile-only --device cuda

# 첫 2x2 비교.
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant A --device cuda
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant B --device cuda
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant C --device cuda
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant D --device cuda

# D의 frozen-QP 평가. A/B/C도 같은 명령에서 variant만 바꾼다.
python eval_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant D --split test --device cuda

# 같은 물리 dt로 새 접촉을 구성하며 진행하는 physical rollout 평가.
python eval_multiplier_rollout.py --config configs/multiplier_cfm_ablation.yaml --variant D --device cuda

# 이후 교란 학습의 추가 효과를 비교한다.
python train_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant D_recovery --device cuda
python eval_multiplier.py --config configs/multiplier_cfm_ablation.yaml --variant D_recovery --split test --device cuda
```

체크포인트가 있는 학습 디렉터리는 덮어쓰지 않는다. 재실험에는 새 `--outdir`를 사용한다.
데이터 생성 설정을 바꿨다면 새 `cache_dir`를 지정한다. 다른 설정의 cache는 거부한다.
`--prepare-pilot`은 별도의 작은 데이터 준비 검사에 사용할 수 있다.
기존 `configs/multiplier_cfm.yaml`, `configs/multiplier_cfm_large.yaml`과 그 결과 경로는 유지한다.
새 checkpoint는 v3 형식이며, 모델 구조가 다른 checkpoint의 혼용을 검사한다.

## 7. 평가와 해석

`best.pt`는 고정 샘플 CFM validation loss로 선택한다.
`best_solver.pt`는 validation의 장면 유형·물체 수·source 유형과
호출 예산 `[1,2,4]`를 균등하게 반영한 성공률을 우선하고 projected-gradient 잔차로 동률을 해소한다.
Test 결과나 physical rollout 결과로 checkpoint를 선택하지 않는다.

Frozen-QP 평가는 CFM, 비학습 local head, PGS를 같은 tolerance로 비교한다.
호출별 잔차와 최종 KKT·위치 오차, 교란 복구를 확인한다.
NFE는 endpoint 호출 수이며 PGS sweep과 동일한 계산 비용이 아니다.
Frozen-QP의 solver 시간에는 접촉 후보·QP 구성 비용이 포함되지 않는다.
가속을 주장하려면 같은 성공률·오차에서 전체 구성·복원 비용까지 포함한 시간을 측정해야 한다.

Physical rollout은 매 프레임 새 접촉을 구성하고 위치 보정 뒤 유한 차분으로 속도를 복원한다.
접촉 후보 밖 새 관통, 법선 변화, 속도 오차, 미수렴 프레임도 확인한다.
현재 구현은 여러 물리 프레임을 통한 학습을 추가하지 않는다.
Inner rollout의 K번 호출은 하나의 고정 QP 안에서 이루어지며 물리 시간 `dt`를 전진시키지 않는다.

기존 평면 경계의 해석식과 접촉 후보 규칙을 유지한다.
비선형 접촉 재선형화, 마찰·반발·회전, CCD, 별도 fallback은 이번 실험 범위에 포함되지 않는다.
테스트와 짧은 smoke 실행은 구현 검증이며 PGS보다 빠르거나 잘 수렴한다는 근거가 아니다.

## 8. 코드와 검증

- `src/multiplier_flow/problem.py`: 고정 QP, analytic endpoint, 위치 복원, KKT 잔차.
- `model.py`: 국소/전역 통신, 접촉 head, CFM loss와 checkpoint 검사.
- `solvers.py`: 학습·평가의 공통 Euler step, PGS, 선택적인 Q guard.
- `data.py`: PGS 정답과 source-target 쌍 cache.
- `experiment.py`: variant 선택과 공통 cache·출력 디렉터리 해석.
- `training.py`: 정규화된 물리 endpoint loss, inner/recovery 적분 학습.
- `recovery.py`: 교란 생성과 남은 시간의 복구 평가.
- `evaluation.py`: QP/head preflight, CFM/local/PGS 비교, solver validation.
- `rollout.py`: 물리 시간 평가와 접촉 변화 진단.
- `train_multiplier.py`: CFM·inner/recovery 학습, 비용 측정과 checkpoint 선택.

```powershell
python -m unittest discover -s tests -p "test_multiplier*.py" -v
python -m unittest discover -s tests
```

핵심 검사는 단일 접촉 해석식, 정확한 coupling의 CFM 표현,
비음수성·multiplier 감소, 실제 parameter gradient,
시간을 유지한 suffix 적분, padding·순열·독립 성분,
전역 정보 전달, checkpoint 격리와 짧은 학습·평가 경로다.
