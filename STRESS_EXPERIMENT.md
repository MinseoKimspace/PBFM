# 대규모 원: PGS와 D 전역 보정 + PGS

원 **256·512·1,024·2,048개**로 비교한다. 모델은 기존 D의 전역 attention과
analytic head를 그대로 사용한다. 비교 대상은 PGS 단독과 D를 K번 실행한 뒤
동일 QP를 동일 PGS로 허용 오차까지 푸는 하이브리드다.

## 실행

기존 통합 YAML `configs/multiplier_cfm_ablation.yaml`의 `stress` 절을 사용한다.
기존 학습 설정과 체크포인트를 바꾸지 않는다. 기본 checkpoint는
`runs/multiplier_cfm_ablation/D/cfm/best_solver.pt`다.

```powershell
# 컴파일된 순차 PGS용 numba를 포함한 의존성
python -m pip install -r requirements.txt

# 같은 고정 QP: 전체 규모·3개 장면·3개 seed에서 정확도/시간/메모리 비교
python eval_multiplier_stress.py --device cuda --mode snapshot

# 먼저 한 장면을 실제 물리 시간으로 실행: 256개, 300프레임
python eval_multiplier_stress.py --device cuda --mode rollout --sizes 256 --scenes pile_drop --seeds 42 --hybrid-calls 4 8 16 --output runs/stress_rollout_256

# 전체 규모로 물리 rollout 확장
python eval_multiplier_stress.py --device cuda --mode rollout --sizes 256 512 1024 2048 --output runs/stress_rollout_full
```

`--mode snapshot`은 같은 초기 물리 상태에서 첫 proposal을 구성하고 **같은 QP**를 비교한다.
`--mode rollout`은 같은 초기 장면에서 출발하지만 각 solver의 실제 궤적을 따라간다.
두 모드의 시간 비율은 구분해서 해석해야 한다.

추가 옵션:

```powershell
# 특정 D 체크포인트와 별도 결과 경로
python eval_multiplier_stress.py --device cuda --checkpoint-file path/to/best_solver.pt --output runs/stress_other_checkpoint

# 더 엄격한 허용 오차와 큰 PGS 한도 (양쪽에 동일 적용)
python eval_multiplier_stress.py --device cuda --tolerance 0.0001 --max-sweeps 20000 --output runs/stress_tol1e4

# 기존 PyTorch device PGS 구현을 그대로 사용하는 비교
python eval_multiplier_stress.py --device cuda --pgs-backend torch --sizes 256 --seeds 42 --scenes pile_drop --output runs/stress_torch_pgs

# 짧은 실행 확인
python eval_multiplier_stress.py --device cuda --mode rollout --sizes 256 --scenes pile_drop --seeds 42 --hybrid-calls 4 --steps 10 --output runs/stress_quick
```

그 외 `--hybrid-calls`, `--timing-repeats`, `--no-render`, `--seeds`를 사용할 수 있다.
`--outdir`은 기존 학습 root, `--output`은 새로운 평가 결과 디렉터리다.
결과가 있는 디렉터리는 덮어쓰지 않는다. 자동 재개는 지원하지 않는다.

## 장면

| 장면 | 구성 |
|---|---|
| `pile_drop` | 여러 층의 엇갈린 원 더미. 상층이 더 빨리 하강하여 첫 프레임부터 전체를 압축 |
| `pile_impact` | 하강·압축 중인 더미의 상부 구간에 추가 낙하 간격과 속도를 부여 |
| `pile_shear` | 수직 압축과 동시에 이웃 층이 좌우 반대 방향으로 움직임 |

단순히 멀리 떨어진 원을 추가하지 않는다. 원 반지름 0.45와 기존 벽 너비를 유지하고,
원이 늘면 연결된 더미의 높이가 커진다. 초기 중심은 겹치지 않는다.
접촉 수, 연결 성분 수, 가장 큰 연결 성분의 접촉 수를 JSON에 기록한다.

이 장면들은 **의도적으로 압축 속도 구배를 준 stress test**다.
원 수가 늘면 높이와 상층 속도도 증가하므로, 결과는 물리적 부하와 규모가 함께 커지는 실험이다.
자연 낙하 자료의 대표성이나 원 개수만의 효과로 해석하지 않는다.
Seed는 작은 초기 속도 차이를 만든다. Snapshot은 크기·장면·seed별 QP 하나를 비교한다.

## 대규모 구성과 solver

기존 작은 장면용 geometry는 멀리 떨어진 원 쌍의 Jacobian도 만든다.
이 경로는 전체 pair distance를 확인하되, 기존 contact margin에 들어온 접촉만
Jacobian 행으로 만든다. gap·법선·slop·접촉 순서는 기존 구현과 같다.

- 모든 pair distance 검사: O(N²). 근접 판정 이전에 모든 pair Jacobian을 만들지 않는다.
- 접촉 행렬 `D=J W J^T`와 모델의 전역 attention은 여전히 dense다.
- 연결 성분은 D의 실제 nonzero 연결로 계산한다. Cubic transitive closure를 사용하지 않는다.
- `eta`는 `eta_fraction / max_i sum_j |D_ij|`로 정한다. 이는 최대 고유값의 안전한 상계를
  사용한 step이다. 양쪽에 동일 적용하며, QP·PGS coordinate 갱신·D analytic field는 바꾸지 않는다.
  기존 작은 장면 평가와는 projected residual의 step 값이 다를 수 있다.
- CPU/GPU 메모리는 무한히 확장되지 않는다. 특히 dense attention은 접촉 수의 제곱에 비례한다.

기본 `pgs_backend: sparse_cpu`는 Numba로 컴파일한 **순차 PGS**다.
같은 순서로 `lambda_i = max(0, lambda_i - g_i/D_ii)`를 갱신하고 residual을 갱신한다.
고정 D의 0인 항만 생략한다. Colored/Jacobi 방식이나 다른 relaxation을 사용하지 않는다.
각 sweep마다 residual을 다시 계산하고, 기존 projected KKT 허용 오차를 검사한다.
BLAS와 sparse 합산 순서의 차이 때문에 부동소수점 결과가 bitwise 같다는 보장은 없다.

PGS 단독과 하이브리드의 마무리는 **같은 backend**다.
하이브리드는 FM endpoint를 그대로 넘기며, 별도 clipping·정답 입력·재선형화·실패 fallback이 없다.
각 QP/물리 프레임은 multiplier 0에서 시작한다. 프레임 간 warm start는 하지 않는다.

## 측정

### Snapshot

- 실제 한 world, batch size 1.
- 접촉 구성 경로를 예열하고 각 방법 1회 solver warmup 후 기본 3회 반복. 반복마다 방법 실행 순서를 섞는다.
- Solver 시간은 중앙값. Hybrid의 FM·CPU/GPU multiplier 전송·PGS 마무리 전체를 포함한다.
- QP 구성, GPU→CPU D 전송 및 sparse 구조 구성은 별도의 setup 시간이다.
- 순수 solver 비율과 공통 setup을 포함한 비율을 모두 기록한다.
- 양쪽 KKT가 성공한 경우에만 speedup을 계산한다. 미수렴을 빠른 실행으로 취급하지 않는다.

### Rollout

- 매 물리 프레임 QP 재구성. PGS 단독과 각 hybrid K는 독립 궤적이다.
- 초기 상태를 진행하지 않고 전체 첫 프레임 경로를 예열한 뒤 각 프레임을 1회 실행한다.
  첫 방법에만 lazy import/JIT 비용을 부과하지 않는다. 시간 평균·중앙값·p95·최댓값을 저장한다.
- FM 적분 실패 또는 PGS sweep 소진 시 해당 프레임을 진행하지 않고 실패를 기록한다.
- 마지막 허용 상태를 보존하며 성공한 것처럼 나머지 프레임을 채우지 않는다.
- 실패한 solver 시도의 시간·sweep도 보고한다. OOM 등 시간 측정을 완료하지 못한 실패는 별도 표시한다.
- 두 궤적이 모두 전체 프레임을 완료한 경우에만 전체 rollout 시간 비율을 계산한다.

공통 정확도 지표는 projected KKT residual, 선형 접촉 침투량, 실제 원의 기하학적 침투량이다.
Snapshot의 `position_rmse_vs_pgs`는 PGS 결과와의 일치도이며 별도 정확한 정답에 대한 오차가 아니다.
진단은 FP32 solver 문제를 사용한다. FP64 oracle은 이 대규모 실행에 포함하지 않는다.

CUDA allocator의 최대 allocated/reserved 메모리와 시작점 대비 추가 peak를 기록한다.
이는 CPU 프로세스 최대 메모리가 아니며, PGS 쪽에도 상주 모델과 공통 QP가 포함된다.
메모리 부족 시 해당 방법/장면의 실패를 명시한다. 모델을 자동으로 작은 구조로 바꾸지 않는다.

**공통 QP 구성은 두 방법에 같은 dense 표현을 제공하기 위한 것이다.**
특히 PGS만을 위한 최적화된 엔진은 dense D 없이 구현할 수도 있다.
이번 시간 비율은 이 구현의 비교이며 Box2D 등 native 엔진 전체보다 빠르다는 뜻은 아니다.
현재 물리 모델에는 마찰·반발·회전·CCD와 프레임 내부 접촉 재선형화가 없다.

## 출력

기본 snapshot 경로는 `runs/multiplier_cfm_ablation/D/stress/best_solver_snapshot/`다.

- `stress.json`: checkpoint·환경·구성·장면별/방법별 결과. Rollout은 프레임 기록 포함.
- `stress.md`: 비교 표. 미수렴 사례의 speedup은 `n/a`.
- `*_initial.pt`: 실제 초기 상태와 반지름.
- `*_positions.pt`: snapshot의 각 방법 최종 위치.
- `*_trajectory.pt`: rollout 전체 상태와 반지름.
- `*_progress.json`: 긴 rollout의 10프레임 간격 진행 기록. 중간 기록임을 표시.
- `*_k4.gif` 등: PGS와 hybrid를 나란히 렌더링. 실패 시 STOPPED 표시.

초기 smoke에는 학습되지 않은 D 형태의 모델을 사용할 수 있지만, 그 결과는 구현·메모리 검증에만
사용해야 한다. 성능 실험은 실제 D `best_solver.pt`로 실행한다.

## 검증

```powershell
python -m unittest discover -s tests -p "test_multiplier_stress.py" -v
python -m unittest discover -s tests
```

기존 geometry와 compact QP 일치, exact 연결 성분, sparse/dense 순차 PGS 일치,
FM endpoint 전달과 실패 처리, 시간 집계, 2,048개 장면 생성, JSON 및 2패널 GIF를 검사한다.
