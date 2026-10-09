# TVR Object / Action 영역 분류

원본 train 87,174개와 validation 10,895개 쿼리의 기존 추출 결과를 재사용합니다.
Object는 NN/NNS 기반 명사, Action은 동사입니다. 시각적으로 검증된 객체/동작만을 뜻하지 않습니다.

## 다이어그램과 영역

점의 색은 데이터 출처, 큰 원은 해당 split에서 만든 lemma + WordNet 후보 커버 범위입니다.
반대 split에 동일 정규화 lemma가 있거나, 같은 품사에서 하나 이상의 WordNet synset을 공유하면 커버 후보입니다.
WordNet은 모든 의미를 고려합니다. 문맥상 의미 일치가 검증된 결과가 아닙니다.

- A: train 쿼리에서 해당 aspect 용어가 하나 이상 반대 split에 미커버.
- B: 해당 aspect 용어가 하나 이상 있고 모두 반대 split에 커버됨. Train/validation은 split 열로 구분.
- C: validation 쿼리에서 해당 aspect 용어가 하나 이상 train에 미커버. 관심 대상.
- EMPTY: 해당 aspect 추출 용어 없음. B로 취급하지 않음.

일부 용어만 미커버인 쿼리도 A/C입니다. coverage_status의 PARTIAL/NONE으로 구분합니다.
no_wordnet_sense_lemmas는 별도 진단이며 A/C와 같지 않습니다. 동일 lemma가 반대 split에 있으면 WordNet에 없어도 커버됩니다.
Object C와 Action C는 서로 겹칠 수 있습니다. Query 전체의 새로움, 검색 난도 또는 불확실성을 직접 증명하지 않습니다.

## 파일

- object_action_regions.png / .svg / .pdf: Object와 Action을 분리한 개념 다이어그램. 원의 면적은 비율을 의미하지 않음.
- object_A/B/C/EMPTY.csv, action_A/B/C/EMPTY.csv: 각 aspect와 영역의 실제 쿼리. 원문/ID 보존.
- query_regions.csv: 전체 결과. 한 행은 쿼리 × aspect이며 총 196,138행.
- query_region_pairs.csv: 한 쿼리의 Object/Action 영역을 함께 기록. CSV 간 중복 식별에 사용.
- term_coverage.csv: split별 lemma 단위 판정, 공유 synset, 반대 split의 근거 쿼리 ID.
- object_B_synonym_terms.csv / action_B_synonym_terms.csv: 실제 B 쿼리의 용어별 유의어 여부, 후보, synset·뜻·반대 split 근거 캡션.
- B_synonym_summary.csv: B 쿼리에서 유의어 후보가 있는 비율. 하나 이상의 용어/모든 용어를 구분.
- region_summary.csv: 각 영역의 쿼리 수와 전체 해당 split 쿼리 수 기준 비율. EMPTY 포함 분모.
- methodology.json: 입력 해시, WordNet 버전, 추출 방법과 한계, 교차 영역 집계.

CSV는 Excel 호환 UTF-8 BOM입니다. 배열/사전 셀은 JSON입니다.
lemma_only_region은 WordNet 확장 없이 동일 lemma만 비교한 영역입니다.
requires_sense_review=1은 WordNet 후보에 의존하는 쿼리/용어입니다.
원본 그림의 1.57%는 검증된 기준이 없으므로 재사용하지 않았습니다.

## B 유의어 후보

B는 커버리지 분류입니다. 동일 lemma로 커버되거나 WordNet에 원본만 있는 경우도 있으므로 교체 가능성을 보장하지 않습니다.
B CSV의 wordnet_synonym_candidates는 용어별 전체 WordNet 유의어 후보입니다.
other_split_synonym_candidates는 반대 split에서 관측된 다른 lemma 중 동일 synset을 공유하는 후보입니다.
원본 자체·단순 굴절·UK/US 철자 변형은 유의어 후보에서 제외합니다. 동사·명사 의미는 섞지 않습니다.
has_any_*는 해당 aspect 용어 중 하나 이상에 후보가 있음, all_terms_have_*는 모든 용어에 후보가 있음을 뜻합니다.
용어별 CSV의 has_wordnet_entry / has_wordnet_synonym / has_other_split_synonym으로 등재·다른 유의어·반대 split 후보를 따로 구분합니다.
wordnet_synonyms_by_sense는 의미별 후보와 뜻, other_split_evidence는 실제 관측 캡션을 제공합니다.
NOT_EVALUATED_NON_B 행의 후보 열은 비어 있으며 유의어가 없다는 뜻이 아닙니다.
후보는 기본형 표현입니다. 문맥·시각적 동작·구동사·문법·동작 시점은 검증하지 않았으므로 자동 치환하지 않습니다.
실제 쿼리 재작성에는 객체 강조·대명사 해소·가시적 상태·감정의 동작화·14–20단어/30 미만 subword·시점별 동작 보존 규칙을 적용해야 합니다.

기존 train 사전과 validation spaCy POS 추출은 추출 방식이 다릅니다.
다의어, 추상 명사/비시각적 동사, POS·lemma 오류, 구동사 손실이 결과에 영향을 줄 수 있습니다.
rewriting이나 모델/데이터 split 변경은 수행하지 않습니다.

## 재현

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/build_object_action_regions.py
```

무작위 샘플링/모델 초기화 없이 원본 순서와 정렬된 근거를 사용합니다.
스크립트는 입력 해시·원문 일치·영역 분할 합계와 핵심 빈 용어/부분 커버 사례를 검증합니다.
