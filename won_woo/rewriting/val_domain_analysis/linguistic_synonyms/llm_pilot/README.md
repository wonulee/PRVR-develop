# WordNet 조건을 제외한 LLM 유의어 검증 pilot

이 검증은 사용자 선택에 따라 대표 사례만 대상으로 진행했습니다. 유의어 미발견 11,227개 caption 전체를 검증한 결과가 아닙니다.

## 대상과 방법

- 기존 실패 사유 3종 × action/object 2종에서 각 4개씩, 서로 다른 표현 24개를 선택했습니다. 각 범주에서 빈도가 높은 표현을 우선했습니다.
- 명확한 유의어 `handcuffed → cuffed`, `vomits → throws up`과 서로 다른 object `earphones → camera`를 대조 사례로 추가했습니다.
- Val 10,895개 caption에서 기존 pool에 속하는 head의 실제 단어·동사 particle 구·명사 수식 구를 수집했습니다. 후보 head와 Val 예문 ID를 기록했습니다.
- WordNet의 synset 일치 조건과 원문 문맥 점수 0.20 필터를 적용하지 않았습니다. 원문 표현 자체를 MiniLM에 입력하여 action/object별 가까운 Val 표현 20개를 검색하고, 각 표현의 실제 Val 예문을 함께 제공했습니다.
- 로컬 Qwen2.5-7B-Instruct가 후보를 제안하고, 별도 prompt에서 한 쌍씩 SYNONYM / SOURCE_CORRECTION / REJECT / UNCERTAIN으로 판정했습니다. 같은 어휘의 형태 변환은 추가로 거절했습니다.
- 마지막으로 Codex assistant(LLM)가 원문·치환 문장·실제 Val 예문을 질적으로 검토했습니다. 이는 사람 전문가의 ground truth 또는 영상 픽셀 검증이 아닙니다.

## 검토 결과

| 판정 | 대표 표현 수 |
| --- | ---: |
| 텍스트 문맥상 유의어 후보 | 2 |
| 원문 오타 교정 가능성 | 1 |
| 의미 차이 검토 필요 | 1 |
| 제안된 치환 거절 | 20 |
| 합계 | 24 |

새 유의어 후보:

1. `whereabouts → location`: Train의 특정 시점 소재와 Val의 사람 위치가 같은 의미입니다. 원문 `whereabouts`의 NNS 태그를 기계적으로 따라 `locations`로 만들지 않고, 문맥에 맞는 `location`을 기록했습니다.
2. `ability → skill`: Train의 `writing ability`와 Val의 `skills`가 능력·숙련의 의미를 공유합니다. 기존 `writing` 수식어를 유지합니다.

`complements → compliments`는 outfit을 칭찬하는 문맥에서 오타 교정일 가능성이 있으므로 유의어 쌍으로 추가하지 않았습니다. `encourages → induces`는 권유와 행동 유발의 강도·인과성 차이로 보류했습니다.

대조 사례 3개는 쌍별 판정과 마지막 텍스트 검토에서 모두 의도한 결과를 얻었습니다. 대조 사례는 출력의 기본 동작 확인이며 전체 검증 정확도를 추정하는 benchmark가 아닙니다.

## 발견한 오류와 검증 기준

첫 후보 제안에서 Qwen은 `zips → unzip`, `IV → IV stand` 등을 승인했습니다. 쌍별 판정에서 상당수가 거절됐지만, 서로 다른 Val 의미를 승인하거나 타당한 표현을 거절하는 오류가 남았습니다. 따라서 모델의 후보 제안 또는 단일 승인 결과를 곧바로 caption에 적용하지 않았습니다.

최종 검토에서:

- `zips → unzip`: 닫기와 열기의 반대 action
- `MRI → MRI machine`: 검사와 장비의 차이
- `app → application`: Train은 소프트웨어, 제시된 Val 예문은 오디션 신청서
- `assumes → supposes`: Train은 믿음, Val의 `wasn't supposed to`는 의무·허용 의미
- `mess → mess up`: 원문 `mess it up`에서 치환하면 `mess up it up`으로 particle 중복
- `thumbs → thumb`, `thanks → thank`: 동일 어휘의 형태 변화

를 거절했습니다. `SYNONYM`은 다른 표현, 같은 Val 문맥 의미, action/object 정체성과 구체성, polarity·시간·참여자·argument 구조, 치환 후 문법을 함께 검토한 판정입니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `llm_approved_pairs.csv` | 최종 텍스트 검토의 새 후보 2쌍; 기존 synonym_pairs 형식과 주요 열 호환 |
| `llm_semantic_review.csv` | 대표 24개와 대조 3개의 모든 판정·이유·실제 Val 예문 |
| `semantic_review_summary.json` | 최종 통계, 범위, 검토 주체와 한계 |
| `qwen_provisional_pairs.csv` | Qwen 쌍별 판정이 잠정 승인한 5개; 최종 승인 목록과 구분 |
| `llm_pair_review.csv`, `review_summary.json` | Qwen 쌍별 판정과 동일 어휘 제외 결과 |
| `llm_validation.csv`, `summary.json` | 최초 Qwen 후보 제안; 잘못된 승인·JSON 출력 오류도 보존 |
| `requests.jsonl`, `responses.jsonl` | 최초 입력·후보·원시 출력 |
| `pair_review_requests.jsonl`, `pair_review_responses.jsonl` | 쌍별 판정 입력·출력 |
| `system_prompt.txt`, `pair_review_prompt.txt` | 실행 prompt |

## 한계와 적용 범위

WordNet 후보가 없었던 표현에도 LLM으로 새 후보를 찾을 수 있음을 확인했습니다. 이번 검토 범위는 표현별 검색 후보 20개와 그중 Qwen이 제안한 한 쌍입니다. 거절은 해당 제안의 판정이며, Val 전체에 다른 유의어가 없다는 결론이 아닙니다.

검증은 텍스트 의미 수준입니다. 영상의 timestamp와 픽셀에 대응하는 action/object를 직접 확인하지 않았으므로 `requires_review=True`, `video_grounding_verified=False`를 기록했습니다. 기존 caption·merge·유의어 CSV와 학습 feature는 변경하지 않았습니다.

로컬 모델은 greedy decoding, seed 2026, GPU 0으로 실행했습니다. 아래 명령은 Qwen의 자동 단계만 재현하며 최종 Codex 질적 검토 결과는 `llm_semantic_review.csv`에 보존했습니다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 /home/viglab2/miniconda3/envs/holmes/bin/python rewriting/validate_val_synonyms_llm.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 /home/viglab2/miniconda3/envs/holmes/bin/python rewriting/review_llm_synonym_proposals.py
```
