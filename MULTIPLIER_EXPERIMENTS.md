# 확대 학습 + 동적 접촉 진단

기존 pilot 설정·결과와 `contact_flow_v2`는 보존한다.
새 실행은 `configs/multiplier_flow_large.yaml`, `runs/multiplier_flow_large/`를 사용한다.
B는 구간 끝점 위치 loss, C는 **동일 모델·데이터 + Lagrangian flow-map matching**이다.
모델 크기와 matching weight는 유지한다. 새 목적함수나 PBD 마무리를 추가하지 않는다.

## 1. 확대 예산과 데이터

| 항목 | 설정 |
|---|---|
| Train | 3개 원 512장면 + 5개 원 512장면 + release 128장면 |
| Val / test | 각각 원 크기별 48장면 + release 128장면 |
| 크기 분리 | train/val 3·5개, test 8·10개; split seed 분리 |
| 구간 | 장면당 32개, train 총 36,864개 |
| 시작 상태 | zero / under / over / mixed / near, 장면당 각 6~7개 |
| 시간 | 50% log-uniform, 50% anchor 1·4·16·32·64·128·256; 장면별 첫 zero는 256 |
| 학습 | batch 128, lr 3e-4, 최대 30,000 optimizer updates |

B/C 초기 seed·샘플 순서·updates는 같다. C의 AD 비용까지 같다는 뜻은 아니다.
기존 `slot < 4`만 교란하던 코드를 `slot % 5` 유형 순환으로 바꿨다.
따라서 구간 32개 중 수렴 상태 28개가 생기던 문제를 막는다.
near도 해 주변 양쪽을 교란한다. 원래 anchor `p`는 유지한다.
각 시작점에서 유한 시간 참조 끝점을 새로 적분하며 QP 해로 label을 대체하지 않는다.

참조 생성은 CPU float64, 128구간씩 처리한다. 전체 구간에 J/D를 복제하지 않는다.
진척·시간·시작 유형 수·장면별 horizon 성공을 기록한다.
완료된 split은 `segments_parts/`에 저장해 준비 재실행 때 재사용한다.
진행 중이던 split 내부 chunk까지 재개하지는 않는다. 다른 설정의 캐시는 덮어쓰지 않는다.
기존 v1 cache는 기존 모델 평가용으로 읽을 수 있지만 신규 학습에는 v2 cache가 필요하다.

현재 장면 생성기는 수직/비스듬한 stack·자유낙하·추상 release다. 규모를 늘렸다고
임의 충돌 분포까지 확보한 것은 아니다. 새 동적 장면은 평가용이며 자동 재학습하지 않는다.

## 2. 확대 실험 실행

```powershell
# 네트워크 없이 참조 문제 검증
python eval_multiplier.py --config configs/multiplier_flow_large.yaml --preflight

# 작은 생성 비용 측정: 별도 prepare_pilot/에 저장, 학습하지 않음
python train_multiplier.py --config configs/multiplier_flow_large.yaml --prepare-pilot 8

# 공통 전체 데이터 생성은 한 프로세스에서 한 번만 수행
python train_multiplier.py --config configs/multiplier_flow_large.yaml --prepare-only
python train_multiplier.py --config configs/multiplier_flow_large.yaml --profile-only --device cuda

# B/C: GPU가 하나라면 차례로 실행
python train_multiplier.py --config configs/multiplier_flow_large.yaml --objective endpoint --device cuda
python train_multiplier.py --config configs/multiplier_flow_large.yaml --objective map --device cuda

# 분포 내 / 원 개수 일반화 평가
python eval_multiplier.py --config configs/multiplier_flow_large.yaml --objective endpoint --split val
python eval_multiplier.py --config configs/multiplier_flow_large.yaml --objective map --split val
python eval_multiplier.py --config configs/multiplier_flow_large.yaml --objective endpoint --split test
python eval_multiplier.py --config configs/multiplier_flow_large.yaml --objective map --split test
```

`--max-updates 20`은 smoke test용 정확한 갱신 수를 지정한다.
`--epochs`를 명시하면 config의 max_updates 대신 해당 epoch 수를 사용한다.
둘 다 주면 max_updates가 우선한다. 기존 checkpoint가 있는 학습 경로는 거부한다.
이 버전은 훈련 resume를 지원하지 않는다. 중단된 run의 checkpoint는 보존한다.
새 훈련에는 새 outdir을 사용한다. 설정이 같은 segments.pt는 복사해 재사용할 수 있다.

30,000회는 상한 예산이지 수렴 보장이 아니다. train/val zero-start horizon 중
tolerance 미달이 있으면 학습을 중단한다. 실패 장면을 제외하거나 label을 바꾸지 않는다.

## 3. 학습 중 진단과 checkpoint

`history.json`에 실제 updates, 경과 시간, loss를 남긴다.
첫 epoch, 이후 약 1,000 updates마다(epoch 경계), 마지막에 전체 validation을 평가한다.
새 참조 경로 생성 없이 raw K=8 map의 projection 수렴을 측정한다.

- `best.pt`: 기존 B/C 통제대로 validation 구간 끝점 MSE 최선.
- `best_solver.pt`: 자유낙하 유형 제외 zero/over/mixed validation 성공률 최선,
  동률이면 projected-gradient 평균 잔차 최선. 물리 rollout으로 고른 것은 아니다.
- `last.pt`: 마지막 상태 보존. 현재 구현은 재개 기능을 제공하지 않는다.
- `gradient_probe`: 고정 validation batch에서 endpoint/matching loss와 각각의
  **가중 gradient norm**. B에도 진단용 matching을 계산하되 업데이트에는 넣지 않는다.

solver-selected 비교는 B/C 모두 `--checkpoint best_solver`로 실행한다.
일반 best와 결과 파일명이 다르다. `max_scenes: 0`은 전체 장면이다.
양의 제한은 유형별 round-robin으로 뽑으며 per_scene/groups를 함께 기록한다.

guarded 결과에는 완료 여부, 도달 시간, NFE, backtracks, 최소 수용 h,
실패 코드(0 완료 / 1 재시도 한도 / 2 전체 반복 한도)를 장면별로 남긴다.
v2 guard는 수용한 h를 다음 반복에도 사용하므로 과거 guard timing과 직접 섞지 않는다.
목표 T에 미도달한 결과를 완성된 finite-time map처럼 해석하면 안 된다.

## 4. 실제 움직임 실행

```powershell
# 학습과 별개로 즉시 가능한 PGS 기준 300프레임
python eval_multiplier_rollout.py --objectives --device cpu --output runs/motion_pgs

# B/C checkpoint 생성 후: PGS, B K1/K8, C K1/K8
python eval_multiplier_rollout.py --device cuda

# 별도 비교 결과 경로
python eval_multiplier_rollout.py --steps 60 --output runs/motion_short
python eval_multiplier_rollout.py --checkpoint best_solver --output runs/motion_solver_selected
python eval_multiplier_rollout.py --guarded --output runs/motion_guarded
```

학습 GPU를 같이 쓰면 timing이 오염된다. PGS CPU 진단은 학습과 병행할 수 있지만
속도 비교용 B/C 평가는 다른 GPU나 학습 종료 후 실행한다.
`--objectives endpoint`로 B만, `--objectives` 뒤를 비우면 checkpoint 없이 PGS만 실행한다.
보고서 경로가 이미 차 있으면 거부한다. 기존 결과는 삭제하지 않는다.

### 공통 외부 루프

1. 현재 `(x,v)`에서 중력·damping으로 proposal `p` 생성.
2. 매 프레임 `p`에서 near contacts, 법선, 질량을 새로 계산.
3. lambda=0부터 PGS 또는 학습된 map으로 동일한 frozen QP 보정.
4. `x_next=p+W J^T lambda`, `v_next=(x_next-x)/dt`.
5. 실제 원 기하를 다시 계산해 진단하고 다음 프레임으로 이동.

장면: 바닥 낙하, 비스듬한 두 원 충돌, stack 위 낙하 충돌, 기울어진 stack 붕괴.
새 접촉은 다음 프레임에서 반영한다. 한 프레임 안에서 법선 갱신/접촉 재해결은 하지 않는다.
완성된 nonlinear PBD 가속이 아니다. 반발·마찰·CCD 해결, multiplier warm start,
위치/속도 clipping, PBD finish가 없다. bound 초과/비정상 값은 실패로 기록하고 정지한다.
guard가 T를 끝내지 못한 중간 결과로 물리 프레임을 진행하지 않는다.
유한하지만 미수렴한 raw/PGS 결과는 진단을 위해 진행하며 solver_converged=false로 기록한다.

### 지표와 렌더링

`rollout.json`, 장면별 `*_trajectories.pt`, 동기화된 비교 GIF와 마지막 PNG를 저장한다.
PGS 기준 고정 카메라를 사용하고 화면 밖 물체와 일찍 정지한 run은 표시한다.

- `local_qp_position_mse`, projected-gradient: 그 모델의 현재 입력에서 frozen solve 오차.
  진단용 고정밀 PGS 실패 시 오차는 null이다. 진단 정답은 추론에 넣지 않는다.
- `geometric_penetration`: 실제 원/경계에서 측정한 slop 이후 관통.
- `new_violations_outside_frozen_contacts`: 구성 때 없던 접촉의 새 관통.
- 선형화 gap 오차·법선 회전: 고정 법선 근사와 실제 기하의 차이.
- 접촉 생성/해제 수, 해제 직후 gap, 속도, 운동에너지와 projection에 의한 운동에너지 변화.
  에너지 증가는 진단량이지 무조건 물리 오류라는 판정은 아니다.
- `swept_endpoint_missed_pairs`: 이전→다음 위치의 직선 이동 가정에서만 나타난 원간 관통.
  FM 내부 경로 검사가 아니며 CCD를 해결한 것도 아니다.
- `pgs_rollout_*_mse`: 같은 프레임의 독립 PGS rollout과 위치·속도 차이.
  PGS도 완전한 물리 정답은 아니다. 조기 종료한 trace를 늘려 평균내지 않는다.
- free_flight/onset/contact/release 단계별 요약. 생성·해제가 동시에 있으면 onset으로
  분류하되 두 event count는 모두 기록한다.

timing에는 외력/접촉 구성/D·eta/solve/FD가 포함된다. oracle·진단·렌더링은 제외한다.
작은 순차 진단 harness이므로 최적화된 PBD 대비 speedup을 주장하는 benchmark는 아니다.
실패한 run은 완료 프레임 수와 함께 비교한다.

## 5. 검증

```powershell
python -m unittest discover -s tests -p "test_multiplier*.py" -v
python -m unittest discover -s tests
```

유형 균형, anchor 보존, 시간 anchor, 정확한 update budget, solver checkpoint,
자유낙하/정지 접촉 FD, swept crossing, guard 실패의 프레임 승격 금지, JSON·GIF를 검사한다.
테스트 통과나 짧은 smoke 학습은 B/C의 물리 성능이 좋다는 뜻이 아니다.
