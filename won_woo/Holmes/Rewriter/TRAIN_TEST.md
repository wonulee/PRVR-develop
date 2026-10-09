# Holmes 방식의 전체 train / test 실험

이 실행기는 기존 Holmes의 `train_one_epoch`, `val_one_epoch`,
모델·손실·옵티마이저·데이터 로더를 재사용합니다.
**전체 train 학습 → validation Rsum으로 best.ckpt 선택 → best.ckpt로 test 평가**
순서입니다. 재작성은 train에만 적용합니다.

## 전체 실행

실행 중 진행률과 상세 출력이 터미널에 바로 표시되며, 같은 내용이 `logs/`에도
저장됩니다. 학습에서는 Holmes의 배치별 loss, epoch별 R@K·Rsum·Best를 확인할 수 있습니다.


```bash
cd /home/viglab2/won_woo/Holmes
bash scripts/run_mca_train_test.sh --gpu 1
```

기본값:

- backbone: `multisrc`
- 조건: `original`과 `context`, 각각 별도 모델을 처음부터 학습
- train: 17,435개 비디오 / 원본 87,174개 쿼리 전부
- 최대 150 epoch, batch size 128 **비디오**, seed 0
- Holmes 설정의 early stopping, 학습률, 손실 가중치, warmup 사용
- 재작성: Qwen2.5-7B-Instruct, RoBERTa 특수 토큰 포함 30토큰 제한
- 출력: `/home/viglab2/won_woo/experiments/mca_train_test/`

`src`를 사용하려면 `--backbone src --output ../experiments/mca_src_train_test`를
지정합니다. src의 기본 최대 epoch는 100입니다.
평가 배치 기본값은 query 16 / context 64이며 점수 정의는 기존 Holmes와 같습니다.
DataLoader worker는 재개와 HDF5 접근을 단순하게 유지하기 위해 0입니다.

GPU 메모리에 따라 학습 batch size를 바꿀 수 있습니다. 처음 실행할 때
`--batch-size 64`처럼 지정하고 모든 조건에 동일하게 적용하세요.
각 contrastive batch에는 비디오가 최소 2개 있어야 합니다.

터미널 종료와 무관하게 실행하려면:

```bash
nohup bash scripts/run_mca_train_test.sh --gpu 1 \
  > ../mca_train_test.log 2>&1 &
tail -f ../mca_train_test.log
```

현재 설치된 holmes Python 환경과 로컬 모델 캐시를 사용합니다.
`PYTHON_BIN=/path/to/python`으로 Python 환경을 바꿀 수 있습니다.

## 조건과 학습 데이터

| 조건 | train 구성 |
|---|---|
| original | 원본 train 전부 |
| target-only | 원본 전부 + 문맥 없이 재작성한 채택 쿼리 |
| context | 원본 전부 + 같은 train 비디오의 형제 문맥으로 재작성한 채택 쿼리 |

세 조건을 모두 비교하려면 처음부터 다음과 같이 실행합니다.

```bash
bash scripts/run_mca_train_test.sh --gpu 1 \
  --conditions original target-only context \
  --output ../experiments/mca_three_conditions
```

재작성문은 `원본ID#ar`로 추가합니다. 실패하거나 변화 없는 문장은 추가하지 않고
원문을 보존합니다. 비디오 수와 epoch당 optimizer step 수는 조건 간 동일하며,
비디오당 학습 쿼리 수는 채택 건수에 따라 증가합니다.

원본 train/val/test 피처는 같은 로컬 RoBERTa로 한 번 추출합니다.
추가 문장만 별도로 추출하고, HDF5 external link로 하나의 데이터셋처럼 연결해
기존 로더에 전달합니다. 원본 피처와 추가 피처를 함께 보존해야 합니다.
val/test에는 원본 캡션과 피처만 사용합니다. 이번 경로에는 test 쿼리 재작성이나
DST 융합이 없으며, 해당 별도 추론 실험은 [EXPERIMENTS.md](EXPERIMENTS.md)입니다.

## 단계별 실행과 재개

```bash
# 전체 train 재작성 및 피처 준비
bash scripts/run_mca_train_test.sh --gpu 1 --stage prepare

# 준비된 데이터로 학습만
bash scripts/run_mca_train_test.sh --gpu 1 --stage train

# 각 조건의 validation-selected best.ckpt로 test만
bash scripts/run_mca_train_test.sh --gpu 1 --stage test
```

동일한 실행 옵션과 output을 사용하세요. 기본 `--stage all`은 **모든 재작성 → 모든 피처 준비 → 모든 조건 학습 → 모든 조건 test**
순서로 수행합니다. 원본 조건 학습도 재작성과 피처 준비가 모두 끝난 뒤 시작합니다. 위 세 단계도 같은 결과를 만듭니다.
`--check-only`는 전체 데이터와 환경 검사만 수행합니다.

같은 명령을 다시 실행하면 완료한 단계를 건너뜁니다.

- 재작성: 완료된 audit 기록 이후부터 이어서 생성.
- 피처 추출: 완료된 파일 재사용, 중단된 추출 단계는 처음부터 재실행.
- 학습: 마지막으로 완료된 epoch의 `last.ckpt`에서 모델·옵티마이저·epoch·
  early-stop 횟수·난수 상태를 복원. 중단된 epoch는 다시 실행.
- 평가: 완료된 test 결과 재사용.

`best.ckpt`와 `last.ckpt`는 임시 파일 저장 후 교체해 중단 시 이전 파일을 보존합니다.
CUDA 연산의 비결정성 때문에 재개 전후 bit 단위 동일 결과까지 보장하지는 않습니다.
코드·입력·epoch 예산·batch size 등 조건을 바꾸려면 새 output을 사용하세요.
GPU 번호와 실행 stage는 바꿔도 됩니다.

## 결과

```text
experiments/mca_train_test/
├── experiment.json          # 설정·입력·코드 식별
├── splits.json              # 실제 분할 개수와 중복 정보
├── original.hdf5            # 공유 원본 피처
├── original/
│   ├── dataset/             # 기존 Holmes 구조의 격리 데이터 경로
│   └── training/
│       ├── best.ckpt        # validation Rsum 최대 모델
│       ├── last.ckpt        # 재개용 마지막 완료 epoch
│       ├── progress.json
│       ├── train_complete.json
│       ├── test_metrics.json
│       └── log.txt
├── context/
│   ├── rewrites/audit.jsonl
│   ├── augmentation.json    # 원본·추가·총 학습 쿼리 수
│   ├── dataset/
│   └── training/            # original과 같은 체크포인트·평가 파일
├── logs/                    # 각 단계의 상세 출력
├── stages/                  # 완료 단계 기록
├── summary.csv              # 조건별 val_best/test R@K, Rsum, 증강 건수
└── complete.json            # 모든 조건의 test 결과 생성 완료
```

## 현재 데이터의 test 해석

현재 로컬 `tvrval.caption.txt`와 `tvrtest.caption.txt`는 **동일한 파일 내용**입니다.
Holmes의 기존 train/val/test 진입 방식은 그대로 실행하되,
`splits.json`, `test_metrics.json`, `summary.csv`에
`independent_test: false`를 표시합니다. 이 test 수치는 독립 held-out 성능이 아닙니다.

동일한 val/test ID는 피처를 한 번만 추출하며, 동일 ID의 텍스트가 다르면 오류로 처리합니다.
train 비디오가 val/test에 포함되면 실행을 거부합니다.
독립 test 파일과 피처를 준비하면 새 output으로 실행하고, 입력된 분할 그대로 평가합니다.

## 검증

```bash
python -m Rewriter.test_rewriter
python -m Rewriter.test_experiment
python -m Rewriter.test_training
python -m Rewriter.test_training --gpu 1 --backbone multisrc
python -m Rewriter.test_training --gpu 1 --backbone src
# 실제 LLM/피처 추출/학습/test/재실행까지 연결 검사
python -m Rewriter.test_training --gpu 1 --pipeline
```

GPU 검사는 합성 데이터로 실제 Holmes 역전파·체크포인트 저장·중단 후 재개·
best 체크포인트의 test 평가를 실행합니다. 이는 실행 경로 검사이며 검색 성능
개선에 대한 실험 결과는 아닙니다.

## 생성 중인 쿼리 확인

현재 작업을 재시작하지 않고 누적 통계와 최근 채택된 원문/생성문을 확인합니다.

```bash
python utils/inspect_rewrites.py --watch
```

`unchanged`는 모델이 원문과 같은 문장을 반환한 경우이며 학습 원문은 보존됩니다.
`temporal_or_negation_change`는 시간·부정 표현 목록이 달라진 경우입니다.
의미 왜곡을 확정하는 검사가 아니며 `then` 하나가 빠져도 제외됩니다.
특정 ID의 원문·후보와 검사된 표현을 보려면 `--id "비디오ID#enc#번호"`를 사용하세요.
`--csv 새파일.csv`로 완료된 기록의 비교표를 저장할 수 있습니다.
