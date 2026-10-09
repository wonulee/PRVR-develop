# Holmes 학습용 Adaptive Rewriting

`holmes_ar.py`는 기존 `src`의 모델, EDL 손실, 데이터 로더, 학습 루프를 재사용합니다. 기존 코드와 평가 쿼리는 변경하지 않습니다. `multisrc` 체크포인트는 지원하지 않습니다.

## 실행

`Holmes` 디렉터리에서 `conda activate holmes` 후 실행합니다. `CHECKPOINT`, `ROBERTA`는 실제 경로로 지정하세요. 출력 디렉터리는 새 경로여야 합니다.

```bash
CHECKPOINT=/absolute/path/to/best.ckpt
ROBERTA=/absolute/path/to/roberta_checkpoint
DATA=/home/viglab2/won_woo
WORK=/home/viglab2/won_woo/ar_experiment

# 1. 비디오 단위로 고정된 1/4 학습 집합을 추출하고 Holmes로 screening.
python utils/holmes_ar.py screen --data-root "$DATA" \
  --checkpoint "$CHECKPOINT" --fraction 0.25 --seed 0 \
  --output "$WORK/screen" --gpu 0

# 2. 불확실성 상위 25%에만 형제 문맥을 제공해 재작성.
python utils/holmes_ar.py rewrite --screening "$WORK/screen/screening.json" \
  --selection uncertain --ratio 0.25 --siblings --roberta "$ROBERTA" \
  --output "$WORK/context" --gpu 0

# 3. 원본+추가 쿼리 및 원본 val/test를 동일한 설정으로 특징 추출.
python utils/extract_tvr_roberta_features.py \
  --caption-file "$WORK/context/train.caption.txt" \
  --caption-file "$DATA/tvr/TextData/tvrval.caption.txt" \
  --caption-file "$DATA/tvr/TextData/tvrtest.caption.txt" \
  --model "$ROBERTA" --max-length 30 --layer -2 \
  --output "$WORK/context/features.hdf5" --device cuda:0

# 4. 별도 데이터 경로를 구성하고 기존 src/main.py로 학습.
python utils/holmes_ar.py train --data-root "$DATA" \
  --captions "$WORK/context/train.caption.txt" \
  --text-features "$WORK/context/features.hdf5" \
  --output "$WORK/context_training" --seed 0 --gpu 0
```

가능하면 같은 1/4 집합으로 먼저 기준 모델을 학습해 screening 체크포인트로 사용하세요. Screening 없이 그 집합을 만들려면:

```bash
python - <<'PY'
from pathlib import Path
from utils.holmes_ar import training_captions, subset, write_captions
write_captions(Path('/tmp/tvr_quarter.caption.txt'),
               subset(training_captions(Path('/home/viglab2/won_woo')), 0.25, 0))
PY
```

이 캡션으로 위의 특징 추출 및 `train` 단계를 먼저 실행합니다. 각 실험은 같은 추출 모델·레이어·길이·seed·학습 예산을 사용하세요. 제공된 원본 feature와 새 feature의 동등성을 가정하지 않습니다. `screen --text-features`로 기준 모델에 맞는 HDF5를 지정할 수 있습니다.

## 대조 실험

동일한 `screening.json`을 사용하고 출력 디렉터리만 다르게 지정합니다.

| 설정 | 역할 |
|---|---|
| `--selection none` | 원본만; LLM/토크나이저를 로드하지 않음 |
| `--selection random --ratio 0.25` | 무작위 선택 후 단일 쿼리 재작성 |
| `--selection uncertain --ratio 0.25` | 높은 불확실성 선택 후 단일 쿼리 재작성 |
| 위 설정에 `--siblings` 추가 | 같은 선택 집합에 학습 비디오의 다른 캡션 제공 |

`ratio`는 생성 시도 수를 맞춥니다. 필터 통과율에 따라 실제 추가 수는 달라지므로 `run.json`의 `selected`, `accepted`를 함께 보고하세요. 원본은 모두 유지하고 생성문 ID에 `#ar`를 붙입니다. 기존 로더는 첫 `#` 앞부분으로 비디오를 찾으므로 BatchPOT에도 같은 비디오의 추가 쿼리로 전달됩니다.

## 동작과 제한

- Train 비디오와 val/test 비디오의 중복을 거부합니다. 형제 문맥은 학습 부분집합 내부에서만 가져옵니다.
- `EDL_loss.get_alpha`를 재사용합니다. 고정된 후보 집합 전체의 strength로 `u=K/S`를 구하고 `max(u_frame,u_clip)`로 선택합니다. 이는 실험용 규칙이며 Holmes의 전체 UGI 분류를 복제하지 않습니다. 다른 후보 집합의 u를 직접 비교하지 마세요.
- 비디오 임베딩을 CPU에 보관하고 GPU에 블록 단위로 올립니다. 전체 쿼리×비디오 점수 행렬은 저장하지 않습니다. CPU 메모리는 후보 비디오 수에 비례합니다.
- 대상 쿼리 하나만 재작성합니다. RoBERTa 토큰 수는 특수 토큰을 포함해 최대 30입니다. 초과 시 한 번 재압축하고 다시 실패하면 추가하지 않습니다. 원본은 길이와 무관하게 보존합니다.
- 시간·부정 표현의 목록과 순서를 검사하지만 의미 보존을 증명하지는 않습니다. `audit.jsonl`의 생성문과 실제 비디오를 표본 검토하세요. JSON 파싱 실패나 생성 중 CUDA OOM도 기록하고 원본을 유지합니다.
- 중단 시 audit만 남을 수 있습니다. 완료된 실행에는 `train.caption.txt`와 `run.json`이 모두 있습니다. 자동 재개는 구현하지 않았습니다.
- 테스트 재작성, DST 융합, cross-attention은 이 학습 증강 실험에 포함하지 않습니다. 성능 개선은 실제 학습·평가로 확인해야 합니다.

검증: `python utils/test_holmes_ar.py`. `--smoke`를 추가하면 Holmes 환경에서 작은 합성 HDF5와 임의 초기화 체크포인트로 GPU screening 및 기존 데이터 로더 연결을 검사합니다. 실제 데이터/LLM은 사용하지 않습니다.
