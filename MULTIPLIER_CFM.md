# Contact-structured Conditional Flow Matching (v2)

> **지금 진행할 실험은 별도의 비학습 homotopy preflight다.**
> 초기값 스케일링 + 선형/기하 μ 비교이며, 아래 기존 CFM 학습을 바꾸지 않는다.
> 상세 설정·결과 해석은 [HOMOTOPY_PREFLIGHT.md](HOMOTOPY_PREFLIGHT.md)를 참고한다.
>
> ```powershell
> python eval_multiplier.py --config configs/multiplier_homotopy_preflight.yaml --homotopy-preflight --device cpu
> ```
>
> 이 검증을 위해 `--prepare-only`나 학습 명령을 다시 실행할 필요는 없다.

## 1. 이번에 바꾼 것

기존 raw CFM v1은 PGS가 만든 정답을 학습했지만, 추론에서는 자유로운
`d(lambda)/d(tau)`를 출력했다. 접촉 보정식은 그 출력 안에 없었다.

v2는 **다른 접촉들의 남은 보정을 네트워크가 예측하고, 각 접촉의 보정은
물리 공식으로 계산하는 CFM**이다. 학습과 추론 모두 같은 projection head를 쓴다.
마지막 출력만 clipping하거나, FM 없이 Q/에너지 loss만 학습하는 방식이 아니다.

| 유지한 것 | 변경한 것 |
|---|---|
| 외력/감쇠로 proposal 생성, 최종 위치에서 FD 속도 복원 | 자유로운 field head → 물리 보정식을 포함한 endpoint head |
| 고정 법선 접촉 QP, PGS 정답, source-target 쌍 | Euler를 동일한 수식의 비음수 convex combination으로 계산 |
| 그래프 message passing, CFM tangent loss | 같은 구조에서 학습된 결합만 끈 `local` 비교군 |
| 관통/KKT/접촉 변화/렌더링 진단 | v2 checkpoint와 새 run 경로, 구조 자체의 preflight |

순차 PGS 전체를 그대로 실행하는 것은 아니다. **PGS의 단일 접촉 갱신식과
같은 projected coordinate 식을 병렬로 사용**한다. Jacobi형 갱신에 학습된
접촉 결합 예측을 넣는 구조라고 구분해야 한다.

## 2. 물리 문제가 정의되는 방식

한 물리 프레임의 원래 proposal `p`, 접촉 Jacobian `J`, inverse mass `W`를 고정한다.

```text
D = J W J^T
Q(lambda) = 0.5 lambda^T D lambda + c^T lambda
g(lambda) = c + D lambda
lambda >= 0
x = p + W J^T lambda
```

`c`는 proposal의 signed gap에 slop을 더한 값이다. `D_ii`는 자기 접촉의
질량 가중 스케일이고 `D_ij`는 다른 접촉의 보정이 이 접촉에 미치는 영향이다.
현재 QP의 PGS는 각 접촉을 순서대로 다음처럼 갱신한다.

```text
new_lambda_i = max(0, lambda_i - g_i / D_ii)
increment = new_lambda_i - lambda_i
g = g + D[:, i] * increment
```

누적 lambda는 비음수지만 **증분은 음수일 수 있다**. 불필요해진 접촉의
보정량을 되돌리는 데 필요하다. 중복 접촉에서는 multiplier 해가 비유일할 수
있으므로 최종 위치 오차와 KKT 잔차도 함께 평가한다.

## 3. 모델 내부에서 실제로 계산하는 것

`r_theta`는 다른 접촉의 남은 multiplier 변화율 예측이다. 네트워크 입력은
현재 lambda, 원래 c, 현재 g, D 대각 성분, tau이며 D로 message passing한다.

```python
r = network(lambda_tau, tau, original_problem)
predicted = lambda_tau + (1 - tau) * r

# 각 접촉 i에서 자기 prediction_i는 사용하지 않는다.
endpoint_i = relu(-(c_i + sum(D_ij * predicted_j for j != i)) / D_ii)

# CFM이 matching하는 field는 r 자체가 아니라 이 전체 계산의 출력이다.
u_theta = (endpoint - lambda_tau) / (1 - tau)
```

이는 아래 식과 같다.

```text
endpoint_i = max(0, lambda_i -
    (g_i + (1-tau) * sum_{j!=i} D_ij*r_j) / D_ii)
```

구현에서는 큰 자기 성분을 더했다 빼는 수치 상쇄를 피하려고 첫 번째 형태를 쓴다.
`D_off = D - diag(D)`를 명시적으로 만들고, 원소별 projection은 autograd 안에서 계산한다.

중요한 성질:

- 연결된 다른 접촉이 없으면 합이 0이다. **학습 없이도 단일 접촉의 해를 정확히 계산**한다.
- `endpoint >= 0`이지만 이것만으로 전체 접촉의 비관통/상보성/에너지 감소가 보장되지는 않는다.
- `r=0`이면 현재 다른 접촉 multiplier를 그대로 사용한 analytic local 갱신이다.
- 정확한 직선 경로 위에서 `r = target-source`이고 target이 KKT 해이면,
  predicted=target이므로 projection head 역시 target을 반환한다.
  따라서 새 출력 제약 때문에 CFM 정답을 표현할 수 없게 되는 것은 아니다.
- 일반적인 coupled 해에서는 학습된 field가 0이라는 구조적 보장은 없다.
  solved source도 학습에 포함하고, 실제 KKT 잔차를 별도로 검사한다.

## 4. 학습은 여전히 CFM이다

```python
lambda_tau = (1 - tau) * source + tau * target
target_rate = target - source
predicted_rate = model(lambda_tau, tau, problem)  # 물리 projection 포함
loss = mean_valid_contacts(((predicted_rate - target_rate) / scale)**2)
scale = length_scale / D_diagonal
```

source는 zero/under/over/mixed/near/solved를 균형 있게 포함한다.
target은 같은 원래 proposal 문제를 PGS로 충분히 푼 multiplier다.
PGS iterate 경로를 따라 그리는 학습은 아니지만 **해를 이용한 감독 학습**이므로
teacher-free라고 주장하지 않는다. 조건부 해가 유일한 문제에서 임의의 source로
물리 해의 다양성이 생긴다고 주장하지도 않는다.

네트워크는 target을 입력받지 않는다. endpoint MSE는 로그일 뿐 추가 loss가 아니다.
물리 projection을 포함한 field를 matching하므로 Q/KKT-only learned optimizer와 다르다.
학습에는 1차 parameter gradient만 필요하다. JVP나 spatial Hessian은 계산하지 않는다.

기본 tau 표본의 10%는 0, 나머지는 `[0, 0.999)` 균등 표본이다.
`tau_min_remaining: 0.001`은 유한 정밀도의 QP 정답 오차를 `1-tau`의 극소값으로
나누는 것을 피한다. **추론 종료 시간을 0.999로 자르는 설정은 아니다.**
ReLU projection에는 꺾임과 0 gradient 영역이 있다. 학습이 쉬워지거나 성공한다는
보장은 없으며, 해석해 테스트와 실제 학습 결과를 구분한다.

## 5. 추론과 물리 시간

Euler는 다음과 같은 동일한 식으로 구현한다.

```python
lam = source  # 실제 물리 rollout에서는 0, 시작 noise 없음
for s in range(K):
    tau = s / K
    endpoint = model.endpoint(lam, tau, problem)
    alpha = (1 / K) / (1 - tau)
    lam = (1 - alpha) * lam + alpha * endpoint

x_next = p + W @ J.T @ lam
v_next = (x_next - x_previous) / dt
```

`0 <= alpha <= 1`이고 두 항이 비음수이므로 각 accepted state도 비음수다.
field를 `tau=1`에서 계산하지 않는다. 마지막 Euler step의 alpha는 1이다.
비음수는 **이산 업데이트에서도 유지하는 성질**이며, 에너지 하강이나 수렴 보장과 다르다.

물리 시간 `dt`와 FM 시간 `tau`는 다르다. 이번 목표는 dt를 키우거나 물리 프레임을
건너뛰는 것이 아니라, **같은 물리 프레임 안에서 접촉 문제를 푸는 비용을 줄이는 것**이다.
K번 호출마다 외력을 다시 적용하지 않는다. K가 커지면 반드시 좋아진다는 보장은 없다.

보고서의 기존 이름을 유지한다.

- `cfm_k*_raw`: projection head는 포함. 추가 Q guard/PGS 마무리는 없음.
  v1 raw의 "projection 없음"과 의미가 다르므로 `model_format`도 확인한다.
- `local_k*_raw`: 같은 head와 tau 시간표, 학습된 다른 접촉 예측은 없음.
  NN forward도 생략한다. **고유 Jacobi 반복 알고리즘 자체와는 다른 비교군**이다.
- `pgs`: 원래 순차 PGS를 공통 tolerance까지 실행한 기준.
- `*_guarded`: 별도 진단. 같은 endpoint를 재사용하며 보폭을 줄여 Q 비증가를 검사한다.
  제한 안에 tau=1까지 못 가면 실패다. 에너지 보정/fallback을 숨겨 실행하지 않는다.

매 물리 프레임 접촉을 재구성하지만 프레임 내부 법선은 고정이다.
비선형 재선형화 solver, restitution/마찰, CCD 해결, warm-start 전달은 추가하지 않았다.
geometric/swept 검사는 진단일 뿐 trajectory를 수정하지 않는다.

## 6. 실행 순서

기존 config 파일명은 유지하고 저장 경로만 바꿨다.
작은 실험은 `runs/multiplier_contact_cfm`, Large는 `runs/multiplier_contact_cfm_large`이다.
**기존 raw CFM/B·C 결과를 지우거나 덮어쓰지 않는다.**

```powershell
# 1. QP/oracle + 단일 접촉 head + 정확한 coupling의 CFM 표현 가능성 검증
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --preflight

# 2. 공통 정답 데이터 준비
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --prepare-only

# 3. 새 contact-structured CFM 한 번 학습
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --device cuda

# 4. 분포 내 / 원 개수 일반화: CFM + local + PGS 함께 평가
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --split val --device cuda
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --split test --device cuda

# 5. 동일한 물리 dt의 300프레임 움직임, JSON/GIF
python eval_multiplier_rollout.py --config configs/multiplier_cfm_large.yaml --device cuda
```

먼저 작은 실행/비용만 확인하려면 다음 옵션을 쓴다.

```powershell
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --prepare-pilot 8
python train_multiplier.py --config configs/multiplier_cfm_large.yaml --profile-only --device cuda

# 새 디렉터리의 짧은 pipeline 검사. 본 실험 성능으로 해석하지 않는다.
python train_multiplier.py --config configs/multiplier_cfm.yaml --outdir runs/contact_cfm_smoke --max-updates 20 --device cuda

# 학습 없이 PGS 움직임만
python eval_multiplier_rollout.py --config configs/multiplier_cfm_large.yaml --pgs-only --device cpu

# 추가 Q guard는 별도 결과로 비교
python eval_multiplier.py --config configs/multiplier_cfm_large.yaml --split test --guarded --device cuda
```

Large 데이터/학습 예산은 기존과 동일하다: train 1,216개 문제/43,776쌍,
val/test 각 288개 문제/10,368쌍, 30,000 optimizer updates, batch 128,
hidden 64, message steps 3. Train/Val stack 3·5, Test 8·10.
데이터 규모와 모델 규모를 동시에 바꿔 구조 효과를 혼동하지 않도록 유지했다.

새 checkpoint 형식은 `multiplier_contact_cfm_v2`이다. raw CFM v1 및 B·C weights는
명시적으로 거부하며 **재학습이 필요**하다. 정답 pair 형식 자체는 변하지 않았다.
설정이 일치하는 기존 `pairs.pt`는 재사용 가능하지만 기본 새 디렉터리에서는 새로 준비한다.
학습 checkpoint가 이미 있는 outdir에서는 덮어쓰기를 거부한다. Resume은 지원하지 않는다.

## 7. 결과에서 무엇을 볼 것인가

`cfm/best_solver.pt`는 장면 유형·원 개수·source 모드별 validation 성공률을
균등 평균해 선택하고, 동률이면 projected-gradient 잔차를 사용한다.
`best.pt`는 고정 tau 표본의 CFM loss 기준이다. 둘 다 rollout을 보고 선택하지 않는다.

1. 단일 접촉이 정확한 것은 analytic head의 성과이며 FM 학습의 성과가 아니다.
2. CFM이 `local`보다 coupled 문제를 더 잘 풀어야 학습된 결합 예측의 이득이다.
3. 그것만으로 FM loss 자체의 우월성이 입증되지는 않는다. 동일 head의 다른 학습
   objective와 비교하는 것은 후속 실험이다. 이번에는 새 학습을 CFM 하나로 제한한다.
4. 최종 lambda 비음수, 관통량, KKT 잔차, 위치 오차를 구분한다.
5. 물리 rollout에서 접촉 생성/해제, 잔차, 속도, 실패 프레임을 함께 본다.
6. NFE는 endpoint 평가 수다. `neural_evals`는 local에서 0이며 PGS sweep와 동일 비용이 아니다.
   같은 tolerance/성공률에서 전체 실행 시간을 비교해야 가속을 주장할 수 있다.

## 8. 파일과 테스트

- `src/multiplier_flow/problem.py`: QP, `contact_endpoint`, 위치 복원, KKT 잔차.
- `model.py`: coupling rate → analytic endpoint → CFM field/loss, v2 checkpoint.
- `solvers.py`: PGS/oracle, convex-combination Euler, 선택적 Q guard.
- `evaluation.py`: 구조 preflight, local/CFM/PGS 비교, 그룹별 validation.
- `rollout.py`: 외력 → solver → 위치 → FD 속도, 접촉 변화와 렌더링.
- `data.py`: 기존 PGS 정답과 source-target cache 유지.

```powershell
python -m unittest discover -s tests -p "test_multiplier*.py" -v
python -m unittest discover -s tests
```

테스트는 단일 접촉의 가중치 독립성, 정확한 coupling의 CFM 경로 재현,
multiplier 감소, 비음수 적분, 실제 parameter gradient, padding/순열,
checkpoint 격리, 짧은 train/eval, FD/렌더링을 검증한다.
통과가 coupled 문제의 일반화나 PGS보다 빠른 solver라는 증거는 아니다.
