# MCA Rewriter

**전체 train → validation best → test 실행:** [TRAIN_TEST.md](TRAIN_TEST.md)

전체 데이터 검색·융합 실험은 [EXPERIMENTS.md](EXPERIMENTS.md)를 참고하세요.

`Holmes/Rewriter/`는 `src/`와 `multisrc/`가 공유하는 독립 전처리 모듈입니다.
전처리 모듈은 기존 모델의 가중치를 변경하지 않습니다. 기존 학습 증강 및 불확실성
선별 실험은 `utils/holmes_ar.py`를 그대로 사용합니다.

## 실행

`Holmes` 디렉터리에서 `holmes` conda 환경으로 실행합니다.
출력 디렉터리는 새 경로여야 합니다. 먼저 `--limit 5`로 확인하고 전체 실행 시 제거하세요.

```bash
python -m Rewriter.mca_rewriter \
  --caption-file ../tvr/TextData/tvrtrain.caption.txt \
  --roberta ../tvr/query_only/roberta-base_tuned_model \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
  --limit 5 --output ../rewriting/mca_train_sample

python -m Rewriter.extract_roberta_feat \
  --caption-file ../rewriting/mca_train_sample/rewritten.caption.txt \
  --model ../tvr/query_only/roberta-base_tuned_model \
  --max-length 30 --layer -2 --device cuda:0 \
  --output ../rewriting/mca_train_sample/rewritten.hdf5

python -m Rewriter.test_rewriter
```

`--no-siblings`는 동일 대상의 단일 쿼리 재작성 대조군입니다. `--limit`은
재작성 대상 수만 제한하며 해당 비디오의 형제 문맥은 입력 파일 전체에서 가져옵니다.
비디오마다 쿼리 수가 달라도 처리하며 대상 자신은 형제 문맥에서 제외합니다.

## 출력 및 인터페이스

- `original.caption.txt`: 선택된 대상의 원문과 ID.
- `rewritten.caption.txt`: 검증을 통과한 재작성문만, 원본과 동일한 ID로 저장.
- `audit.jsonl`: 원문, 사용한 형제 문맥, 후보, 토큰 수, 수락 여부, 실패 이유.
- `run.json`: CLI 실행 완료 표시, 모델·토크나이저·옵션 및 처리 건수.

실패한 쿼리의 ID는 재작성 파일에 없습니다. 따라서 재작성 HDF5는 부분 집합이며
기존 전체 평가용 HDF5를 대체하지 않습니다. 후속 융합에서는 빠진 ID의 원본 점수를
그대로 사용해야 합니다. 전부 거절되면 빈 재작성 파일이 생성되므로 특징 추출을
실행하지 마세요. 실행 중 오류는 전파되며 audit만 남을 수 있습니다. `--resume`으로 동일 설정의
재작성을 이어서 실행할 수 있습니다.

```python
from Rewriter.context_manager import build_video_context_db, get_sibling_queries
from Rewriter.mca_rewriter import rewrite_with_context, validate_length, RewriteRejected

db = build_video_context_db(caption_path)
siblings = get_sibling_queries(db, video_id, target_idx)
rewritten = rewrite_with_context(target_q, siblings, llm_pipe, roberta_tokenizer)
assert validate_length(rewritten, roberta_tokenizer, max_tokens=30)
```

함수는 검증된 문자열을 반환하거나 `RewriteRejected`를 발생시킵니다.
LLM 로드는 CLI 실행 시에만 수행됩니다. 피처 추출은 기존
`utils/extract_tvr_roberta_features.py`를 재사용하며, 두 단계에 동일한
RoBERTa 토크나이저/체크포인트와 길이 설정을 지정해야 합니다.

## 6대 원칙과 실험 범위

첨부 설계에서 여섯 항목이 모두 확정되어 있지 않아 다음과 같이 명시했습니다.

1. 대상 사건·참여자·객체·의미 보존.
2. 부정 의미 보존. 부정어 일괄 삭제는 하지 않음.
3. 근거가 명시된 경우에만 대명사 해소.
4. 원문 또는 같은 사건의 문맥에 있는 시각 정보만 사용. 감정에서 행동을 추측하지 않음.
5. 형제 쿼리의 다른 사건을 혼합하지 않고 시간 순서 보존.
6. 불필요한 표현을 줄이고 특수 토큰 포함 RoBERTa 30토큰 이내로 작성.

길이를 초과하면 한 번 더 압축하고, 실패하면 제외합니다. 텍스트를 강제로 자르지
않습니다. 부정·시간 표현의 목록과 순서는 기존 학습 도구의 어휘 검사로 확인합니다.
이는 의미 동등성이나 환각 부재를 보장하지 않으므로 audit 표본 검토가 필요합니다.
변화 없는 문장은 `unchanged`로 제외합니다. 원문 보존 파일에는 길이 제한이 없습니다.

val/test 파일도 파싱할 수 있지만 정답 비디오 ID로 찾은 형제 캡션은 추가 정답 문맥을
제공합니다. 이 설정은 oracle-context 실험으로 구분하고, 일반 단일 쿼리 검색 성능과
동일 조건으로 보고하지 않아야 합니다. 실사용 문맥이 주어지는 경우에는 그 출처를
명시하세요. 전체 검색·DST 융합 비교는 `experiment.py`에서 실행합니다. 자세한 조건은
[EXPERIMENTS.md](EXPERIMENTS.md)에 있습니다.
