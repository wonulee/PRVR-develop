# Train Object/Action 추가 캡션

현재 `tvr/TextData/tvrtrain.caption.txt`를 기준으로 Train→Val 유의어 치환 쿼리 **161개**를 생성했습니다.
Object 119개, Action 42개이며 출발 캡션은 105개입니다. LLM을 사용하지 않았습니다.

| 파일 | 내용 |
| --- | --- |
| [tvrtrain.additional.caption.txt](tvrtrain.additional.caption.txt) | 추가 쿼리 161개만 저장. `video_id#enc#번호 문장` 형식 |
| [tvrtrain.caption.with_additional.txt](tvrtrain.caption.with_additional.txt) | 현재 원본 87,174개 + 추가 161개 = 87,335개 |
| [additional_queries.csv](additional_queries.csv) | 추가 문장 전체와 치환·사전·Val·시간·WordNet 근거 |
| [object_additional_queries.csv](object_additional_queries.csv) | Object 치환 119개 |
| [action_additional_queries.csv](action_additional_queries.csv) | Action 치환 42개 |
| [additional_queries.jsonl](additional_queries.jsonl) | 추가 쿼리와 타임스탬프를 배열로 저장한 메타데이터 |
| [candidate_audit.csv](candidate_audit.csv) | 출발 표현별 후보 1,456건의 생성/제외 사유 |
| [verification.json](verification.json) | 집계, 입력·출력 SHA-256, 생성 기준과 검증 결과 |

## 생성 기준

- [object_B_train.csv](../ABC_csv/object_B_train.csv)와 [action_B_train.csv](../ABC_csv/action_B_train.csv)의 `WORDNET_SYNONYM`만 사용했습니다. A/C에는 반대편 유의어 연결이 없고 B의 참고 연결에는 치환 후보가 없으므로 생성에 사용하지 않았습니다.
- CSV에서 실제 Val 표현과 치환 표현이 같고 양쪽 Penn POS가 같은 경우만 선택했습니다. 출발·대치 표현 모두 같은 Object/Action 공식 JSON 사전에 존재하며 공유 WordNet synset이 있어야 합니다.
- 기존 `select_diagram_synonyms.py`의 유의어군과 문맥 필터를 재사용했습니다. 복합어, 구동사, 뜻이나 논항이 바뀔 수 있는 용법, 관사 불일치를 제외했습니다. 대치 뒤에도 품사·Penn POS·의존관계가 유지되는지 확인했습니다.
- 대명사, 부정, 내부 감정·비시각적 동작이 포함된 문장은 기존 규칙으로 제외했습니다. 임의로 이름이나 영상 동작을 추가하지 않았습니다.
- 쿼리 하나에 한 토큰만 대치했습니다. 14–20단어, 특수 토큰을 포함한 RoBERTa 토큰 수 30 미만을 확인했습니다.
- 같은 영상의 현재 Train 문장, 치환 전 백업 문장, 이미 생성한 문장과 중복되지 않아야 합니다. 규칙을 통과한 824개 중 이전 원본과 같은 663개를 제외해 161개를 저장했습니다.

## 현재 Train과 원본 사전의 차이

현재 Train의 1,031개 문장은 이전 작업에서 이미 치환되었습니다. 공식 JSON의 caption ID별 주석은 치환 전 문장을 기준으로 합니다.
현재 문장의 기존 대치 표현을 다시 치환할 때는 같은 토큰 위치의 원래 표현·POS 주석과 현재 표현의 사전 증거를 모두 확인했습니다.
`source_annotation_basis=projected_previous_substitution`은 이 경우이며 157개입니다.
기존 ID별 주석의 표현을 직접 사용한 `original_id_annotation`은 4개입니다.
원본 사전의 Object/Action 및 POS를 변경하거나 새 단어를 추가하지 않았습니다.

## CSV에서 확인할 열

| 열 | 의미 |
| --- | --- |
| `new_caption_id`, `source_caption_id`, `video_id` | 추가 쿼리 ID, 출발 ID, 동일 원본 영상 |
| `original_caption`, `additional_caption` | 현재 Train의 치환 전/후 문장 |
| `term_type`, `source_expression`, `replacement_expression` | Object/Action 종류와 치환 전/후 표현 |
| `source_pos`, `replacement_pos` | 실제 문맥에서 확인한 Penn POS |
| `source_annotation_expression`, `source_annotation_pos`, `source_annotation_basis` | caption ID별 원본 JSON 주석과 현재 표현의 연결 근거 |
| `source_dictionary_caption_id`, `replacement_dictionary_caption_id` | 각 표현·POS가 공식 JSON에 존재하는 근거 ID |
| `source_span_start`, `source_span_end` | 현재 원본 문장에서 바꾼 토큰의 문자 위치. 0부터 시작, 끝 위치 제외 |
| `mapping_id`, `mapping_csv` | 원래 Train→Val 대응 CSV의 행을 찾는 키 |
| `val_caption_id`, `val_caption` | 실제 Val 표현의 대표 문장 근거 |
| `source_timestamp` | 원본 Train 영상의 동작 시간 `[시작, 끝]` (초) |
| `word_count`, `roberta_token_count` | 추가 문장의 단어/토큰 수 |
| `shared_wordnet_synsets`, `meaning` | 공유 synset과 원래 CSV의 의미 |
| WordNet 8개 열 | 출발 표현의 원래 후보·synset 조회 정보 그대로 보존 |
| `requires_video_review` | 텍스트 규칙을 통과한 결과이며 영상 픽셀 확인은 수행하지 않아 True |

타임스탬프는 `tvr_train_release.jsonl`에서 **원본 영상 ID와 치환 전 백업 문장이 일치하고 시간값이 유일한 경우**에만 사용했습니다.
기존 `train_timestamp_relations.jsonl`의 일부 ID별 시간 연결은 release 문장과 불일치하므로 시간 근거로 사용하지 않았습니다.
전체 원본 중 시간값을 유일하게 찾지 못한 문장은 523개이고, 실제 치환 후보에서는 8건이 시간 확인 실패로 제외되었습니다.

## 검증 및 사용

기존 입력 파일과 사전·대응 CSV의 SHA-256이 유지되는지 확인했습니다.
병합본의 원본 부분은 현재 Train 파일의 바이트를 그대로 보존하며, 새 ID는 기존 Train/Val/Test ID와 충돌하지 않습니다.
단일 토큰 대치, 사전 주석·표현·POS 근거, 공유 WordNet synset, 길이 제한, 중복 제외, 영상·시간 연결을 검증했습니다.

학습에 사용하려면 추가 ID와 문장에 맞는 쿼리 특징을 새로 추출해야 합니다.
동일 영상·시간의 양성 관계를 유지한 표현 변형이므로 Holmes의 교차모달 증거에 문장 표현을 추가하는 자료로 사용할 수 있습니다. 불확실성이나 검색 성능의 변화는 별도 실험이 필요합니다.

재생성 명령 (작업 공간 루트에서 실행):

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/generate_additional_train_queries.py
```

로컬 spaCy POS/구문 분석, WordNet 3.0, 로컬 RoBERTa 토크나이저를 사용했고 시드는 2026입니다.
