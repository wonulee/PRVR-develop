# Train / Val 표현별 매핑

대표 캡션 문장과 WordNet 정보를 한 CSV에서 보려면 [ABC_csv 안내](../ABC_csv/README.md)를 사용하세요. 기존 매핑의 모든 행과 20열 값을 유지한 31열 보기용 CSV입니다.

한 행에서 **Train 표현·POS ↔ Val 표현·POS**를 확인할 수 있습니다.
기본형별 표현 목록을 합친 이전 보기와 달리, 각 행에는 정확한 표현·POS 대응 한 가지가 들어 있습니다.
원본 CSV 합계 3.12 GB에 대해 매핑 요약 CSV는 합계 **6.04 MB**입니다.
전체 캡션 연결과 문장·WordNet 자료를 포함한 CSV 합계는 **60.72 MB**입니다.
LLM은 사용하지 않았으며 기존 JSON 사전 표현만 사용하는 범위를 유지했습니다.

## 현재 A/B/C 집계 기준

**표현·POS의 대응을 중복 제거한 매핑 1행**을 한 건으로 셉니다.
B에서는 같은 출발 표현이 서로 다른 반대편 표현과 연결되면 각각 한 건입니다.
A/C는 반대편 연결이 없으므로 출발 표현·POS별 한 행입니다.
분류는 기존과 같이 동일 기본형 또는 같은 Object/Action 품사의 WordNet synset 연결 여부로 정합니다.
`NN`과 `NNS`처럼 실제 POS가 달라도 같은 기본형으로 연결될 수 있으며, 양쪽 POS는 그대로 기록합니다.

| 영역 | Object | Action |
| --- | --- | --- |
| A·Train | [3,316행](main/object_A_train.csv) | [1,250행](main/action_A_train.csv) |
| B·Train | [6,525행](main/object_B_train.csv) | [12,375행](main/action_B_train.csv) |
| B·Val | [4,779행](main/object_B_val.csv) | [7,648행](main/action_B_val.csv) |
| C·Val | [0행](main/object_C_val.csv) | [0행](main/action_C_val.csv) |

[세부 집계](../reports/ABC_mapping_counts.csv)에는 매핑 행 수, 출발 표현·POS 수, 고유 기본형 수, 원본 행 수를 구분해 기록했습니다.
[새 다이어그램 PNG](../diagrams/object_action_regions.png) · [SVG](../diagrams/object_action_regions.svg) · [PDF](../diagrams/object_action_regions.pdf)

WordNet과 정확한 Train/Val 캡션 연결은 유지했습니다. 원본 1,494,233행과 유의어 쌍 1,379,891행의 보존 검증도 유지됩니다.
C·Val=0은 기존 JSON 사전 표현으로 제한한 범위의 결과입니다.

## 바로 열기

| 대상 | Train 출발 | Val 출발 |
| --- | --- | --- |
| Object | [object_B_train.csv](main/object_B_train.csv) | [object_B_val.csv](main/object_B_val.csv) |
| Action | [action_B_train.csv](main/action_B_train.csv) | [action_B_val.csv](main/action_B_val.csv) |

A영역: [Object](main/object_A_train.csv) · [Action](main/action_A_train.csv)

`train_expression`, `val_expression`을 나란히 놓았습니다. `train_pos`, `val_pos`도 나란히 표시합니다.
캡션 문장을 반복해서 저장하지 않고 대표 캡션 ID를 요약에 표시합니다. 모든 원본 ID 연결은 `links/`에서 확인합니다.

## 대응 예시

| mapping_id | Train 표현 | Train POS | Val 표현 | Val POS | 연결 종류 |
| --- | --- | --- | --- | --- | --- |
| 3317 | drink | NN | beverage | NN | WORDNET_SYNONYM |
| 3323 | pong | NN | pong | NN | DIRECT_LEMMA_REFERENCE |

이는 실제 출력에 존재하는 후보와 참고 연결입니다. WordNet 의미는 문맥·영상으로 검증하기 전입니다.

## main/ — 정확한 표현별 매핑 요약

| 주요 열 | 의미 |
| --- | --- |
| `mapping_id` | 매핑과 전체 캡션 연결을 잇는 고유 번호 |
| `train_expression`, `val_expression` | 각 split에서 실제 관측된 표현. 서로 독립된 목록으로 합치지 않음 |
| `train_pos`, `val_pos` | 위 표현에 대응하는 실제 Penn POS |
| `train_lemma`, `val_lemma` | 각 표현의 기본형 |
| `source_split` | 원본 CSV의 출발 split. `train` 또는 `val` |
| `replacement_expression` | 원본 유의어 후보에서 출발 표현을 바꿀 때의 표현. 반대 split의 실제 표현과 굴절형이 다를 수 있음 |
| `mapping_type` | 아래 표의 연결 종류 |
| `source_coverage_status` | 출발 기본형의 기존 영역 분류 근거 |
| `full_rows` | 해당 정확한 연결에 묶인 원본 CSV 행 수 |
| `source_caption_count` | 이 연결의 출발 캡션 수 |
| `example_train_caption_id`, `example_val_caption_id` | 실제 연결된 대표 Train/Val 캡션 ID |
| `source_wordnet_id`, `meaning_id` | WordNet 조회 정보와 의미 설명을 찾는 번호 |

Train 출발이면 `replacement_expression`은 Train 표현을 바꾸기 위한 후보입니다.
Val 출발이면 Val 표현을 바꾸기 위한 후보입니다. `train_expression`과 `val_expression`은 항상 실제 관측된 표현입니다.

| mapping_type | 의미 |
| --- | --- |
| WORDNET_SYNONYM | 원본에서 내보낸 유의어 쌍. 실제 Train/Val 표현·POS·캡션 연결을 그대로 보존 |
| DIRECT_LEMMA_REFERENCE | 원본 유의어 쌍은 없지만 반대 split에 같은 기본형이 있는 B항목. 실제 반대편 표현과 대표 캡션 하나를 참고 근거로 표시 |
| WORDNET_COVERAGE_REFERENCE | 원본 쌍 필터를 통과한 치환 후보는 없지만 공유 synset으로 연결되는 B항목. 실제 반대편 표현과 대표 캡션 하나를 참고 근거로 표시 |
| NO_MAPPING | A/C 항목. 반대편 표현·ID는 빈칸 |
| EMPTY / UNCLASSIFIED | 기존 빈 캡션·분류 불가 원본 항목을 보존 |

REFERENCE 두 종류는 연결 근거이며 새 치환 후보로 취급하지 않습니다. `replacement_expression`은 빈칸이고 원본 `synonym_rank`도 0입니다.
반대편에 가능한 모든 캡션 조합을 새로 생성하지 않았습니다. 원본에 있던 연결 전체와, 빈 B행의 참고 근거 한 개씩을 제공합니다.
모든 `requires_review=True`입니다. 현재 C·Val=0은 기존 사전 어휘로 제한한 범위의 결과입니다.

## links/ — 모든 캡션 ID 연결

`main/` 파일과 같은 이름의 `links/` 파일을 엽니다.
`mapping_id`로 필터링하면 해당 표현 쌍의 전체 연결을 볼 수 있습니다. 원본 1,494,233행 모두에 대응하는 연결 행이 있습니다.

| 열 | 의미 |
| --- | --- |
| `mapping_id` | `main/`에서 고른 매핑 번호 |
| `train_caption_key`, `val_caption_key` | 실제 Train/Val 캡션을 찾는 번호 |
| `source_data_row_number` | 원본 37열 CSV에서 헤더를 제외한 1부터 시작하는 행 번호 |
| `synonym_rank` | 원본 후보 번호. 0은 유의어 쌍 없는 원본 행 |

각 연결 행의 두 caption key는 한 쌍입니다. Train ID 목록과 Val ID 목록을 따로 합치지 않았습니다.
[caption 표](captions.csv)의 `caption_key`로 찾아 실제 `caption_id`와 문장을 확인합니다.
캡션은 원본 분석과 동일한 수정 전 Train 백업 및 Val 원문을 기준으로 합니다.

## WordNet 정보

[wordnet.csv](wordnet.csv)에서 `source_wordnet_id`로 등록 여부, 후보 수·목록, 반대 split 후보, 전체·선택 synset 등을 확인합니다.
[meanings.csv](meanings.csv)에서 `meaning_id`로 원본 유의어 쌍의 의미 설명을 확인합니다.
빈 의미 번호는 원본에도 쌍 의미가 없었던 행입니다.
반복되던 값을 한 번씩 저장했으며 원본 WordNet 값이 모두 일치하는지 행별로 검증했습니다.

## 전체 영역 파일

| Object/Action | 영역 | 출발 split | 매핑 요약 | 요약 크기 | 전체 캡션 연결 |
| --- | --- | --- | --- | ---: | --- |
| object | A | train | [3,316행](main/object_A_train.csv) | 373.8 KB | [전체 ID 연결](links/object_A_train.csv) |
| object | B | train | [6,525행](main/object_B_train.csv) | 1,143.9 KB | [전체 ID 연결](links/object_B_train.csv) |
| object | B | validation | [4,779행](main/object_B_val.csv) | 824.2 KB | [전체 ID 연결](links/object_B_val.csv) |
| object | C | validation | [0행](main/object_C_val.csv) | 0.3 KB | [전체 ID 연결](links/object_C_val.csv) |
| object | EMPTY | train | [1행](main/object_EMPTY_train.csv) | 0.4 KB | [전체 ID 연결](links/object_EMPTY_train.csv) |
| object | EMPTY | validation | [1행](main/object_EMPTY_val.csv) | 0.4 KB | [전체 ID 연결](links/object_EMPTY_val.csv) |
| object | UNCLASSIFIED | train | [16행](main/object_UNCLASSIFIED_train.csv) | 2.3 KB | [전체 ID 연결](links/object_UNCLASSIFIED_train.csv) |
| object | UNCLASSIFIED | validation | [0행](main/object_UNCLASSIFIED_val.csv) | 0.3 KB | [전체 ID 연결](links/object_UNCLASSIFIED_val.csv) |
| action | A | train | [1,250행](main/action_A_train.csv) | 143.3 KB | [전체 ID 연결](links/action_A_train.csv) |
| action | B | train | [12,375행](main/action_B_train.csv) | 2,206.8 KB | [전체 ID 연결](links/action_B_train.csv) |
| action | B | validation | [7,648행](main/action_B_val.csv) | 1,345.2 KB | [전체 ID 연결](links/action_B_val.csv) |
| action | C | validation | [0행](main/action_C_val.csv) | 0.3 KB | [전체 ID 연결](links/action_C_val.csv) |
| action | EMPTY | train | [1행](main/action_EMPTY_train.csv) | 0.4 KB | [전체 ID 연결](links/action_EMPTY_train.csv) |
| action | EMPTY | validation | [1행](main/action_EMPTY_val.csv) | 0.4 KB | [전체 ID 연결](links/action_EMPTY_val.csv) |
| action | UNCLASSIFIED | train | [17행](main/action_UNCLASSIFIED_train.csv) | 2.5 KB | [전체 ID 연결](links/action_UNCLASSIFIED_train.csv) |
| action | UNCLASSIFIED | validation | [0행](main/action_UNCLASSIFIED_val.csv) | 0.3 KB | [전체 ID 연결](links/action_UNCLASSIFIED_val.csv) |

[파일별 집계](file_counts.csv) · [생성 검증](verification.json) · [원본 행별 복원 검증](roundtrip_verification.json) · [전체 자료 설명](../README.md)

## 검증과 재생성

원본 유의어 쌍 1,379,891행의 표현·POS·양쪽 캡션 ID와 문장이 모두 일치합니다.
참고 연결도 해당 캡션의 실제 주석 표현·POS와 일치하고, 같은 기본형 또는 공유 synset이 있는지 확인했습니다.
원본 캡션 98,069개의 문장과 WordNet 정보를 별도 표에 보존했습니다.

```bash
cd /home/viglab2/won_woo
python rewriting/export_mapping_diagram_csv.py --output-dir /tmp/diagram_mapping_view
python rewriting/verify_mapping_diagram_csv.py --mapping-dir /tmp/diagram_mapping_view
```

재생성 시 새 출력 폴더를 지정합니다. 모든 CSV는 UTF-8 BOM 형식입니다.

현재 매핑 기준 다이어그램 재생성:

```bash
cd /home/viglab2/won_woo
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/summarize_mapping_regions.py
```
