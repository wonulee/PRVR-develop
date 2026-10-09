> 이전 자동 POS 확장 및 aspect 이동을 포함한 자료입니다. 현행 공식 사전 범위를 충족하지 않아 사용을 중단하고 dictionary_only 재생성 결과로 대체했습니다.

# 문맥 검토 반영: 최종 Object / Action 영역 CSV

남은 검토 657건을 반영했습니다. 기본형·품사 교정 341건, 유지 214건, Object→Action 이동 43건, 문맥상 비시각적 표현 제외 59건입니다.
이동 43건 중 1건은 기존 Action 추출과 합쳤고 42건은 새로 추가했습니다. 원본 문장과 추출 데이터는 그대로 보존했습니다.

## 최종 영역과 단위

- A: Train 단어 중 Val에 동일 기본형도, 같은 품사 WordNet synset 연결도 없는 단어.
- B: 반대 split에 동일 기본형 또는 같은 품사 WordNet synset 연결이 있는 단어.
- C: Val 단어 중 Train에 위 연결이 없는 단어.
- EMPTY: 해당 Object/Action 추출 표현이 없는 쿼리. 유의어가 없다는 의미가 아닙니다.
- EXCLUDED: 검토에서 제외한 원본 표현 또는 Object에서 Action으로 옮긴 표현의 원본 기록. A/B/C 집계에 포함하지 않습니다.
- UNCLASSIFIED: 해소되지 않은 추출·품사·기본형 오류. A/B/C 집계에 포함하지 않습니다.

| 종류 | A Train | B Train | B Val | C Val |
| --- | ---: | ---: | ---: | ---: |
| Object | 2,875 | 2,991 | 2,209 | 322 |
| Action | 882 | 1,595 | 1,006 | 81 |

표와 다이어그램의 숫자는 종류·split별 **고유 기본형 단어 수**입니다. Train B와 Val B는 각 split의 단어 수이므로 서로 다릅니다.
한 쿼리 안의 여러 단어가 다른 영역에 들어갈 수 있습니다. 아래 쿼리 수를 영역 간 합산하면 중복될 수 있습니다.
WordNet 연결은 같은 품사의 동일 synset을 기준으로 합니다. 상하위어 또는 여러 단계를 거친 연결은 사용하지 않습니다.
단어별 영역은 그 split의 유효한 출현에서 선택한 의미를 합쳐 계산합니다. 개별 유의어 행은 그 원본 출현의 선택 의미와 반대 split 근거 출현의 선택 의미가 공유되는지 확인합니다.
A/C에도 전체 WordNet 후보는 있을 수 있으나 반대 split 연결은 없습니다. B에서도 동일 기본형만 겹치거나 적절한 다른 후보·굴절형이 없으면 유의어 쌍이 없을 수 있습니다.

## 영역별 CSV

| 종류 | 출처 | 영역 | 고유 단어 | 쿼리 | CSV 행 | 실제 후보 쌍 | 파일 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Object | Train | A | 2,875 | 6,484 | 6,992 | 0 | [object_A_train.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_A_train.csv) |
| Object | Train | B | 2,991 | 80,467 | 368,154 | 311,673 | [object_B_train.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_B_train.csv) |
| Object | Val | B | 2,209 | 10,308 | 62,255 | 56,445 | [object_B_val.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_B_val.csv) |
| Object | Val | C | 322 | 701 | 724 | 0 | [object_C_val.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_C_val.csv) |
| Action | Train | A | 882 | 1,733 | 1,761 | 0 | [action_A_train.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/action_A_train.csv) |
| Action | Train | B | 1,595 | 85,541 | 1,181,034 | 1,159,074 | [action_B_train.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/action_B_train.csv) |
| Action | Val | B | 1,006 | 10,837 | 251,664 | 249,179 | [action_B_val.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/action_B_val.csv) |
| Action | Val | C | 81 | 85 | 85 | 0 | [action_C_val.csv](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/action_C_val.csv) |

EMPTY / EXCLUDED / UNCLASSIFIED CSV도 같은 폴더에 보존했습니다. 압축하지 않은 UTF-8 BOM CSV입니다.

## 열과 읽는 방법

Train 파일의 앞 13개 열은 `tvrtrain_synonym_pairs` 형식입니다. Val 파일의 대상 열은 `train_expression`, `train_caption_id`, `train_caption`입니다.
WordNet 조회 7개, 영역·이력 4개, 문맥 검토 8개 열을 추가해 **총 32개 열**입니다.

| 열 | 의미 |
| --- | --- |
| source_expression / source_lemma | 원본 표현 / 교정 후 조회 기본형 |
| has_wordnet_entry | 같은 품사의 WordNet 항목 유무. 조회할 원본 단어가 없으면 빈칸 |
| wordnet_candidate_count / wordnet_candidates | 전체 사전 의미에서 찾은 후보 수 / 기본형 목록 |
| wordnet_synsets | 같은 품사의 전체 사전 synset |
| selected_wordnet_synsets | 이 출현에서 사용하는 synset. 검토하지 않은 출현은 전체 의미를 사용 |
| context_wordnet_candidates | 선택한 의미의 후보. 검토되지 않은 AUTO 행은 문맥 검증을 뜻하지 않음 |
| other_split_candidate_count / other_split_candidates | 반대 split의 유효한 출현과 선택 의미가 공유되는 후보 수 / 목록. 굴절 필터 적용 전 |
| term_region / term_coverage_status | 현재 단어 영역 / 연결 근거 또는 제외 상태 |
| query_region | **이전 문장 전체 분류 이력**. 현재 단어의 영역이 아님 |
| original_source_lemma | 문맥·형태 교정 전 기본형 |
| source_pos / original_source_pos | 현재 / 원래 Penn POS |
| original_aspect | 해당 표현이 원래 추출된 Object/Action |
| review_id / review_decision / review_reason | 검토 ID / KEEP·CORRECT·MOVE·EXCLUDE 또는 미검토 AUTO / 이유 |

후보 목록은 ` | `로 구분합니다. 같은 표현·굴절 차이·철자 변형은 대체 후보에서 제외합니다.
`synonym_rank >= 1`은 실제 반대 split 근거가 있는 후보 행이며, `synonym_expression`은 원본 POS에 맞춘 굴절형입니다.
`synonym_rank = 0`은 후보 쌍이 없는 보존 행입니다. 후보·뜻·대상 표현/ID/문장은 빈칸입니다.
`meaning`은 해당 원본과 근거 표현의 선택 의미에서 공유되는 synset 정의입니다. 순위는 사전순이며 적합도 점수가 아닙니다.
`no_pair_annotations.csv`의 파일명과 헤더 제외 1부터 시작하는 행 번호로 후보가 없는 이유를 확인할 수 있습니다.
`query_membership.csv`의 문장 기준 coverage/uncovered 값은 이전 분류 이력입니다. 현재 단어 분류에는 `term_region`을 사용하세요.

## 문맥 검토에서 바뀐 내용

- glasses: 안경류 263건, 음료잔 152건, 대화에서만 언급된 2건 제외. 3D/VR/시야 차단용 특수 안경 16건은 Object로 유지하되 일반 교정용 안경의 synset 후보를 사용하지 않습니다.
- leaves: 출발·퇴장의 동사 41건은 Action `leave`, 실제 식물 잎 1건은 Object `leaf`.
- heaves / pinches: 각각 Action `heave` / `pinch`로 이동. 한숨을 내쉬는 heave의 선택 의미는 Train에 연결되지 않아 Action C입니다.
- scrubs / pliers / lenses: 수술복 / 집게 도구 / 광학 렌즈의 의미와 기본형으로 교정.
- swinging / smelt / saw / shined: 동작 문맥에 따라 swing / smell / see / shine의 기본형·의미로 교정.
- 감정·시간·관계·금융 개념 또는 대화에서만 언급한 대상 등 59건을 제외. EXCLUDED 파일의 102행에는 이동 전 원본 기록 43행도 포함됩니다.

예전 질문의 `friends_s09e23-24_seg02_clip_33#enc#3`에서 **door→doorway는 Object B**입니다. `query_region=A`는 이전 문장 전체 분류입니다. 같은 문장의 scuba는 A, door는 B로 나뉩니다.

## 검증과 파일

독립 검사에서 1,880,989행, 실제 유의어 후보 1,776,371행을 원본과 대조했습니다.
영역·후보·근거 표현·굴절형·행 누락의 계산 검사 오류는 0건입니다. 원본 98,069개 쿼리, query×aspect 196,138개, 표현/EMPTY 그룹 408,739개를 확인했습니다.
검토 큐 657건 모두 최종 CSV에 추적 가능하며 해당 형태 모호성 검토의 미처리 항목은 0건입니다.
원본 캡션·추출 CSV·Train JSON의 해시와 내용은 이전 검사와 동일합니다. WordNet 3.0과 seed 2026을 유지했습니다.

- [최종 다이어그램 PNG](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_action_regions.png) · [SVG](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_action_regions.svg) · [PDF](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs/object_action_regions.pdf)
- [최종 단어별 영역·WordNet 후보 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/final_term_regions.csv) — 품사·split·기본형마다 한 행. 선택 의미/후보는 출현별 의미의 합집합
- [영역별 수치 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/final_region_summary.csv)
- [657건의 결정과 최종 CSV 행 위치](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/final_audit/review_resolution.csv)
- [표현별 검토 요약 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/review_summary.csv)
- [검토 결정 원본 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/decisions.csv)
- [독립 검사 수치와 해시](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/final_audit/summary.json)

657건의 검토는 캡션 문맥과 일부 동일 클립의 다른 캡션을 근거로 했습니다. 영상을 직접 확인한 결과는 아닙니다.
이 검토 목록 밖의 표현은 자동 품사·WordNet 의미 조회를 유지합니다. AUTO 행의 전체 의미 후보나 B 분류가 문장·영상에 맞는 대체 표현을 보장하지 않습니다.
모든 실제 후보의 `requires_review=True`를 유지했으며 원본 문장 전체를 재작성하지 않았습니다.

이전 자동 교정본은 `/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review`에 보존했습니다. 그 이전 초기 파일은 `diagram_synonym_pairs_before_independent_audit/`에 있습니다.
재계산은 `rewriting/export_diagram_synonym_pairs.py`, 독립 검사는 `rewriting/audit_diagram_regions.py`를 사용합니다.
현재 exporter가 생성하는 기본 README보다 이 문맥 검토 요약을 우선 참고하세요.
