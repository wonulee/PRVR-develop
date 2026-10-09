# Object / Action A·B·C 최종 자료

2026-10-09에 중복 복사본과 이전 산출물을 정리했습니다. 표현·POS 매핑, 캡션 연결, WordNet 근거, 현재 추가 쿼리와 검증 결과를 유지합니다.

먼저 [ABC_csv 안내](ABC_csv/README.md)를 확인하세요. Train/Val 표현·POS, 대표 캡션과 WordNet 의미를 한 CSV에서 볼 수 있습니다.

## 남긴 자료

| 경로 | 용도 |
| --- | --- |
| [ABC_csv/](ABC_csv/README.md) | 현재 31열 보기용 CSV 16개, 매핑 35,930행 |
| [mapping/](mapping/README.md) | 정확한 표현·POS 매핑, 전체 캡션 연결, 캡션·WordNet·의미 표 |
| [additional_captions_rank1/](additional_captions_rank1/README.md) | 현재 1순위 유의어 기반 추가 쿼리와 생성 연결·검증 자료 |
| [additional_captions/](additional_captions/README.md) | 별도 규칙 적용 실험의 161개 후보와 거절 사유 감사 자료 |
| [diagrams/](diagrams/object_action_regions.png) | 현재 매핑 기준 그림 PNG·SVG·PDF |
| [reports/](reports/summary.json) | 집계, 사전 범위, 원본 항목·캡션 포함 정보와 검증 결과 |
| csv/ | 동일 원본 37열 CSV 16개를 가리키는 상대 심볼릭 링크 |
| [package_verification.json](package_verification.json) | 현재 남은 파일의 SHA-256 및 원본 CSV 연결 정보 |
| [cleanup_20261009.json](cleanup_20261009.json) | 삭제 목록, 원본 동일성 검사와 정리 전후 용량 |

`csv/`의 중복 복사본 16개는 모두 원본과 SHA-256이 같은지 확인한 뒤 삭제하고 링크로 교체했습니다. 원본 위치는 아래와 같습니다.

`../rewriting/val_domain_analysis/dictionary_only/diagram_synonym_pairs/`

기존 `csv/<파일명>` 경로는 계속 사용할 수 있습니다. 이 폴더를 다른 위치로 옮길 때는 연결된 원본 CSV도 함께 가져와야 합니다.

## 현재 A/B/C 집계

표현·POS의 대응을 중복 제거한 매핑 1행을 한 건으로 셉니다. 같은 출발 표현이 여러 반대편 표현과 연결되면 각각 한 건입니다.

| 영역 | Object | Action |
| --- | --- | --- |
| A·Train | [3,316행](ABC_csv/object_A_train.csv) | [1,250행](ABC_csv/action_A_train.csv) |
| B·Train | [6,525행](ABC_csv/object_B_train.csv) | [12,375행](ABC_csv/action_B_train.csv) |
| B·Val | [4,779행](ABC_csv/object_B_val.csv) | [7,648행](ABC_csv/action_B_val.csv) |
| C·Val | [0행](ABC_csv/object_C_val.csv) | [0행](ABC_csv/action_C_val.csv) |

EMPTY·UNCLASSIFIED 항목도 유지합니다. 전체 캡션 연결은 `mapping/links/`에서 `mapping_id`로 찾고, 실제 문장은 `mapping/captions.csv`의 `caption_key`로 조회합니다.

분석 범위는 기존 Object/Action JSON 사전의 정확한 표면형·Penn POS입니다. 사전 밖 Val 표현은 추가하지 않았고 LLM도 사용하지 않았습니다. C·Val=0은 이 제한 범위의 결과입니다. WordNet 후보는 문맥·시각적 의미 검증 전 자료입니다.

[현재 PNG](diagrams/object_action_regions.png) · [SVG](diagrams/object_action_regions.svg) · [PDF](diagrams/object_action_regions.pdf) · [집계 기준](reports/ABC_mapping_counts.csv)

## 추가 쿼리와 연구 실험 연결

현재 rank1 결과는 Object 185,934개·Action 180,510개, 총 366,444개입니다. 원문 87,174개를 포함한 병합본은 453,618개입니다.

[추가 TXT](additional_captions_rank1/tvrtrain.additional.caption.txt) · [병합 TXT](additional_captions_rank1/tvrtrain.caption.with_additional.txt) · [치환 대응](additional_captions_rank1/replacement_pairs.csv)

이 rank1 생성에서는 길이·토큰·POS·문맥 필터를 적용하지 않았습니다. 검증 PASS는 ID·대치 재구성·파일 무결성을 뜻하며 시각적 의미 보존이나 학습 성능을 보장하지 않습니다. 규칙을 적용한 161개 결과는 별도 비교 자료로 유지했습니다.

`experiments/multi_query_selection_20261008`의 입력 해시와 기존 스크립트가 참조하는 `additional_captions_rank1/`, `ABC_csv/`, `mapping/` 경로 및 데이터는 보존했습니다.

## 정리한 자료

이전 전체 후보 생성본 `additional_captions_unfiltered/`, 기본형 집계 중복본 `compact/`, 이전 기본형 그림 `diagrams/lemma_previous/`, 중복 문서 `read.me`를 삭제했습니다. 원본 CSV·캡션·모델·실험 결과는 수정하지 않았습니다.

기존 세부 검증 JSON은 생성 당시의 감사 기록으로 유지하고, 현재 파일 목록과 링크 상태는 `package_verification.json` 및 `cleanup_20261009.json`에서 확인합니다.
