# D 방식: 전역 정보를 사용하는 CFM과 실제 적분의 최종 오차 학습

여기서 **D는 실험 variant 이름**이다. 접촉 행렬도 기존 코드에서 `D`로 표기하므로,
이 문서에서는 행렬을 `A = J W Jᵀ`라고 써서 구분한다.

## 1. 구성과 비교군

현재 통합 설정에서 D는 다음 조합이다.

| 요소 | D 설정 |
|---|---|
| 상태 | 각 접촉의 누적 multiplier λ |
| 조건 | 해당 물리 프레임에서 고정한 QP와 접촉 연결 구조 |
| 입력 | residual 특징 6개/접촉 |
| 신경망 | hidden 64, 국소 message 단계 3회, 4-head attention 층 2개 |
| 출력 head | 다른 접촉의 예측을 사용하는 analytic endpoint |
| 학습 | CFM loss + zero-start inner rollout의 최종 KKT/위치 loss |
| Inner 예산 | 매 update K∈{1,2,4} 균등 선택 |
| Recovery loss | 0; `D_recovery`에서만 별도로 활성화 |
| 현재 평가 예산 | K∈{1,2,4,8,16,32,64} |

B는 같은 전역 모델에서 inner loss가 없는 비교군이다. C는 inner loss가 있지만 전역
attention이 없는 비교군이다. D_direct는 D와 같은 backbone/학습 조건에서 최종 analytic
endpoint를 제거하고 신경망의 signed rate를 직접 FM field로 사용하는 비교군이다.

## 2. 한 번 푸는 물리 문제

제안 위치 p, 접촉 Jacobian J, 역질량 W, signed gap c를 고정한 문제를 푼다.

```text
A = J W Jᵀ
Q(λ) = ½ λᵀ A λ + cᵀ λ,  λ ≥ 0
g(λ) = c + A λ
x(λ) = p + W Jᵀ λ
```

KKT 조건은 λ≥0, g≥0, λᵢgᵢ=0이다. λ가 커지면 대응 접촉에 대한 위치 보정이 커지지만,
A의 비대각 성분 때문에 한 접촉의 변화가 다른 접촉에 영향을 준다.
한 stack의 바닥 접촉 보정이 위쪽 접촉까지 바꿀 수 있는 것이 전역 정보를 사용하는 이유다.

λ는 이 위치 보정 QP의 변수다. 물리 시간을 직접 예측하는 상태나 Box2D의 전체 접촉
알고리즘 자체를 의미하지 않는다. 마찰·회전·CCD까지 해결하는 범용 물리 엔진은 아니다.

## 3. 네트워크가 보는 정보

각 접촉 i의 고정 scale은 `sᵢ = length_scale / Aᵢᵢ`이며 기본 길이는 0.1이다.
입력은 다음 여섯 값이다.

1. 현재 multiplier `λᵢ / sᵢ`
2. 원래 gap `cᵢ / length_scale`
3. 현재 gap `gᵢ(λ) / length_scale`
4. 대각 계수 `log(1 + Aᵢᵢ)`
5. FM 시간 τ
6. 국소 analytic 보정과의 차이 `(λᵢ − endpoint_localᵢ) / sᵢ`

MLP로 각 접촉을 64차원 token으로 만든다. 먼저 정규화된 A를 사용하는 국소 메시지
전달을 3회 수행하고, 그 뒤 같은 연결 성분의 모든 접촉이 참여하는 attention을 2회 수행한다.

```text
attention score(i,j)
  = qᵢ·kⱼ / sqrt(head_dim) + learned_head_weight × Aᵢⱼ / sqrt(Aᵢᵢ Aⱼⱼ)
```

직접 결합 Aᵢⱼ가 0이어도 중간 접촉을 통해 같은 성분으로 연결되어 있으면 볼 수 있다.
서로 독립인 성분과 padding은 차단한다. 별도 super token으로 압축하지 않는다.
구현상 같은 접촉 token 집합에서 Q/K/V를 만들므로 **component 내부 self-attention**이다.
서로 다른 token 집합 간 cross-attention을 구현한 것은 아니다.

마지막 선형층은 각 접촉의 signed rate `rθ`를 출력한다. 이미 feasible하고 λ도 0인
독립 성분은 0을 유지한다. 같은 성분의 다른 접촉이 보정 중이면 자신의 잔차가 0이어도
출력을 허용한다. 따라서 개별 접촉의 현재 잔차로 전역 보정을 막지는 않는다.

## 4. 예측 rate가 실제 FM field가 되는 과정

D에서는 `rθ`를 그대로 λ에 더하지 않는다. 먼저 남은 시간 동안 다른 접촉이 어디까지
변할지 예측하고, 그 예측을 조건으로 각 접촉의 scalar QP를 해석적으로 푼다.

```text
z = λ + (1 − τ) rθ(λ, τ, QP)
eᵢ = max(0, −[cᵢ + Σⱼ≠ᵢ Aᵢⱼ zⱼ] / Aᵢᵢ)
uθ = (e − λ) / (1 − τ)
```

신경망은 전역적으로 결합된 변화율을 예측하고, analytic head는 그 예측에 맞춰 각 접촉의
비음수 endpoint를 만든다. 다른 접촉의 예측을 반영하므로 단순한 독립 scalar solver가 아니다.
동시에 계산되므로 PGS의 접촉별 순차 갱신과도 다르다.

단일 접촉에서는 합이 비어 `e=max(0,−c/A)`가 되어 신경망과 무관하게 해석해를 구한다.
이 장점은 analytic head가 제공한다. 반면 여러 접촉에서 전역 KKT 수렴이나 에너지 감소는
보장되지 않는다. λ는 비음수여도 `uθ`는 음수가 될 수 있어 기존 보정을 해제할 수 있다.

## 5. CFM으로 학습하는 대상

충분히 수렴한 CPU float64 PGS 해 `λ*`를 학습용 endpoint로 사용한다.
초기값 `λ₀`는 zero/under/over/mixed/near/solved 여섯 종류다.

```text
λτ = (1 − τ) λ₀ + τ λ*
u_target = λ* − λ₀
L_CFM = mean_QP mean_valid_contacts [ (uθ(λτ,τ) − u_target) / s ]²
```

손실은 analytic head까지 포함한 **최종 field uθ**에 적용된다. 내부 출력 rθ에 직접
정답 변화율을 맞추는 손실이 아니다. 입력에는 현재 λτ, τ, 고정 QP만 들어가며
추론 시 λ*는 필요 없다.

따라서 PGS 정답에 의존하는 endpoint 감독 학습은 맞다. PGS의 중간 iterate나 접촉별
업데이트 순서를 따라 하도록 학습하는 trajectory 증류는 아니다.
FM 시간이 실제 물리 시간이나 PGS 반복 번호를 의미하지도 않는다.

## 6. Inner rollout loss가 추가하는 학습

CFM 학습 경로의 상태만 잘 처리한다고 모델 자신의 추론 경로에서도 정확하다는 보장은 없다.
D는 매 update zero-start에서 실제 solver를 한 번 적분하고, 그 끝점도 학습한다.

```text
K ~ Uniform({1,2,4})
λ̂K = Integrate(uθ, λ=0, τ:0→1, K calls)
Rη(λ̂K) = λ̂K − max(0, λ̂K − η g(λ̂K))

L_residual = mean_QP mean_valid_contacts [Rη / (η × length_scale)]²
L_position = mean_QP mass_weighted_coordinate_MSE(x(λ̂K),x(λ*)) / length_scale²
L_D = L_CFM + L_residual + L_position
```

위 식의 가중치는 현재 모두 1이다. 중간 λ를 detach하지 않으므로 마지막 오차의 gradient가
앞선 호출까지 전달된다. 중간 FM 상태에 물리적 feasible 조건을 강제하지 않고,
완전한 [0,1] 적분의 끝점에 손실을 적용한다.

이 inner rollout은 한 고정 QP 안에서 모델을 여러 번 호출하는 것이다.
여러 물리 프레임을 연속으로 학습하는 physical rollout이나 교란 recovery 학습과 구분된다.

## 7. K=16에서 실행되는 것

등간격 `h=1/K`, `τk=k/K`로 Euler를 실행한다. Analytic head에서는 아래 안정적인
동치식을 사용한다.

```text
αk = h / (1 − τk) = 1 / (K − k)
λk+1 = (1 − αk) λk + αk eθ(λk,τk,QP)
```

K=16이면 τ=0,1/16,…,15/16에서 16회 신경망/field를 평가한다. 마지막 호출에서는
α=1이므로 그때의 endpoint가 최종 λ가 된다. τ=1에서는 field를 호출하지 않는다.
별도 PGS 마무리나 recovery는 없다. 비음수 상태와 endpoint의 convex combination이므로
상태의 비음수성이 유지된다.

K=16은 K=4 결과에 12번을 추가하는 것이 아니라 [0,1] 전체를 다른 간격으로 적분하는
별도 실행이다. 학습된 field의 오차 때문에 K를 늘린다고 반드시 성공률이 단조 증가하지는 않는다.
현재 inner loss는 K≤4를 사용하므로 K=16·32·64는 학습된 같은 field를 더 촘촘히 적분하는 평가다.

## 8. 연구 결과에서 구분할 주장

- D와 B의 차이: 같은 전역 모델에 inner loss를 추가한 효과.
- D와 C의 차이: 같은 inner 학습에서 전역 attention을 추가한 효과.
- D와 D_direct의 차이: 같은 backbone/학습 예산에서 analytic head를 제거한 효과.
- D K회와 PGS K sweep의 차이: 반복 예산당 오차 감소의 비교.
- 같은 정확도에서의 실행 시간: 실제 solver 가속의 비교.

현재 best_solver checkpoint는 validation source/장면 유형과 K={1,2,4}를 균등하게 반영해
선택한다. 평가 K를 확장해도 그 checkpoint 선택 기준까지 자동으로 바뀌지는 않는다.
K=16 중심 선택을 연구하려면 test가 아닌 validation에서 선택 기준을 따로 정해야 한다.

구현 위치: `src/multiplier_flow/model.py`, `training.py`, `solvers.py`, `experiment.py`.
