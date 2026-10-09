**이 문서는 문맥 검토 전 자동 교정 결과입니다.** 남은 657건을 반영한 [최신 최종 결과](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/README.md)와 [검토 완료 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/context_review/final_audit/review_resolution.csv)를 확인하세요. 아래 657건의 “남은 검토”는 최신 결과에서 처리 완료했습니다. 이 문서의 이전 단계 CSV는 `diagram_synonym_pairs_before_context_review/`에 보존했습니다.

# A/B/C 전수검사 결과

이전 CSV에서 분류·기본형·후보 검사가 필요한 행 8,349개를 확인했습니다. 자동 교정 가능한 부분을 수정하고 1,882,441행, 실제 후보 1,778,296개를 원본 데이터·WordNet에서 독립적으로 재검산했습니다.
현재 분류·후보·근거 문장·굴절형·행 누락 검사 오류는 0건입니다. 아래 단어/품사/WordNet 규칙에 대한 검증이며 문맥·영상 의미 검증은 포함하지 않습니다.

## 문제와 처리

- 기본형 교정: 129개 표현·품사 조합, 원본 표현 1,646개. miss→mis, kiss→kis처럼 잘못 잘린 기본형을 바로잡고 singing→singe를 sing으로 교정했습니다.
- 영역 변경: 원본 표현 498개. A→B 165개, C→B 2개, B→A 5개, B→C 1개, A/B/C→UNCLASSIFIED 325개입니다.
- 기존 A/B/C에 잘못된 추출·기본형 325개가 섞여 있었습니다. 기존 검토 대상 1개를 합쳐 UNCLASSIFIED는 표현 326개입니다.
- 이전 CSV에서 beinged, sanging, childrens 등 검증되지 않는 생성형 5,300행을 확인했습니다. 잘못된 반대 split 근거가 있는 후보 6,111행도 확인해 유효한 원본 표현으로 다시 계산했습니다.
- 한 행에 여러 문제가 나타날 수 있어 범주별 수를 더하면 중복됩니다. 이전 CSV에서 중복을 제거한 검토 행 수는 8,349개입니다.
- 원본 쿼리 98,069개, query×aspect 196,138개, 표현/EMPTY 그룹 408,701개를 보존했습니다. 원본 캡션·추출 CSV의 SHA-256은 전후 동일하며, Train 원본 JSON 내용도 해당 원본 추출값과 일치합니다.

## 영역 기준

- A: Train 단어 중 Val에 동일 기본형도 없고 같은 품사 WordNet synset을 공유하는 단어도 없는 경우.
- B: 반대 split에 동일 기본형 또는 같은 품사 WordNet synset 연결이 있는 경우.
- C: Val 단어 중 Train에 위 연결이 없는 경우.
- EMPTY: 해당 Object/Action 추출 표현이 없는 쿼리. WordNet 후보가 없다는 뜻이 아닙니다.
- UNCLASSIFIED: 구두점, 문장에 없는 문자열 조각, 해소되지 않은 축약형, 잘못된 기본형/품사 등 검토할 표현. A/B/C에 포함하지 않습니다.

A/C에도 자체 WordNet 유의어는 있을 수 있습니다. 반대 split과 연결되지 않는다는 의미입니다.
B의 후보는 동일 품사 synset으로 연결되는 실제 반대 split 표현이며 모든 의미를 대상으로 찾습니다. 상하위어·다단계 연결은 사용하지 않습니다.
B에서도 다른 유의어가 없거나 굴절 필터를 통과하지 못하면 rank=0일 수 있습니다.

아래는 고유 source_lemma 수입니다. Train/Val의 B는 각각 해당 split의 단어 수이므로 서로 다를 수 있습니다. 쿼리 수 또는 CSV 행 수와 비교하지 마세요.

| 종류 | A Train | B Train | B Val | C Val |
| --- | ---: | ---: | ---: | ---: |
| Object | 2,872 | 2,998 | 2,211 | 323 |
| Action | 882 | 1,596 | 1,007 | 80 |

## door 행 확인

`friends_s09e23-24_seg02_clip_33#enc#3`의 door→doorway 행은 **term_region=B**입니다. Val의 doorway와 같은 명사 synset을 공유합니다.
이 문장의 scuba는 A이며 다른 겹치는 단어는 B로 나뉩니다.
`query_region=A`는 교정 전 문장 전체 분류 이력입니다. 현재 단어 영역은 `term_region`을 보세요.
`query_membership.csv`의 `query_coverage_status`·`query_uncovered_lemmas`도 기존 문장 기준 이력이며 교정 후 단어 판단에 사용하지 않습니다.
`original_source_lemma`는 교정 전 값이고 `source_lemma`가 현재 조회 기본형입니다.

## 남은 의미·품사 검토

형태 사전으로도 기본형을 하나로 확정하지 못하는 27개 표현·품사 조합, 원본 표현 657개는 별도 CSV에 기록했습니다.
예: glasses의 안경/잔 구분, swinging의 swing/swinge 구분, leaves가 실제 동사인데 명사로 추출된 경우 등입니다. 657개가 모두 오류라는 뜻은 아닙니다.
제공된 NN/NNS를 Object, 동사 Penn POS를 Action으로 사용했으므로 추상 명사·품사 오분석이 남을 수 있습니다.
WordNet 미등록 기본형 1,555개 aspect×split 기록도 따로 저장했습니다. 미등록 단어는 EMPTY가 아니며 오탈자·이름·새 단어 등을 검토해야 합니다.
문맥의 의미, 구동사, 영상의 객체·동작·시간 구간은 자동 검증하지 않았습니다. 모든 requires_review=True를 유지했습니다.

## 결과 파일

| 종류 | A Train | B Train | B Val | C Val |
| --- | --- | --- | --- | --- |
| Object | [A Train](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/object_A_train.csv) | [B Train](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/object_B_train.csv) | [B Val](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/object_B_val.csv) | [C Val](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/object_C_val.csv) |
| Action | [A Train](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/action_A_train.csv) | [B Train](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/action_B_train.csv) | [B Val](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/action_B_val.csv) | [C Val](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/action_C_val.csv) |

- [수정된 다이어그램](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_context_review/object_action_regions.png)
- [이전 CSV의 문제 행 전체](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/previous_csv_findings.csv) — 파일명·헤더 제외 행 번호는 아래 이전 보존본 기준입니다.
- [단어 영역 변경 498개](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/after/region_changes.csv)
- [기본형 교정 1,646개](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/after/normalization_corrections.csv)
- [남은 문맥/품사 검토 657개](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/after/ambiguous_source_occurrences.csv)
- [잘못된 추출/기본형 326개](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/after/rejected_source_terms.csv)
- [WordNet 미등록 단어](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/after/terms_without_wordnet_entry.csv)
- [최종 전수검사 수치와 CSV 해시](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_region_audit/after/summary.json)
- [이전 CSV 보존본 설명](/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_independent_audit/README.md)

압축하지 않은 CSV입니다. 수정본은 `/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs`, 이전 파일은 `/home/viglab2/won_woo/rewriting/val_domain_analysis/diagram_synonym_pairs_before_independent_audit`에 있습니다.
`before/`는 초기 검사 결과이며, `after/`가 최종 결과입니다. `confirmed_errors.csv`는 처음 확인한 304행 목록이며 최종 전체 목록은 `previous_csv_findings.csv`입니다.

## 재현

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/export_diagram_synonym_pairs.py
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/audit_diagram_regions.py --output-dir rewriting/val_domain_analysis/diagram_region_audit/after
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/audit_previous_diagram_pairs.py
```

WordNet 3.0과 기존 random seed 2026을 사용했습니다. 독립 검사는 부모 term_coverage 표 대신 원본 표현에서 어휘와 synset 연결을 재구성합니다.
이전 term_coverage 표는 이전 기본형 기준의 계산 자체는 맞았지만 잘못된 기본형이 있었으므로 최종 어휘 참조에 사용하지 않습니다.
검사 실행 시 요약 원본은 `after/execution_summary.json`에 보존했고 `after/summary.json`의 CSV 경로는 실제 수정본 위치로 갱신했습니다.
