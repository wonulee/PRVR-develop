# 다이어그램 영역별 synonym-pairs 형식 CSV

Object / Action 각각 A_train, B_train, B_val, C_val, EMPTY_train, EMPTY_val 파일로 나눕니다.
Train 파일은 tvrtrain_synonym_pairs의 기존 13개 열을 보존하고, 뒤에 WordNet 조회용 7개 열과 영역 설명용 3개 열을 추가합니다.
Val 파일은 대상 열을 train_expression / train_caption_id / train_caption으로 바꿉니다.

| 추가 열 | 의미 |
| --- | --- |
| source_lemma | 원본 표현을 해당 Object/Action 품사로 정규화한 조회 단어 |
| has_wordnet_entry | 같은 품사 WordNet synset 존재 여부. True/False, 조회할 단어 또는 지원 품사가 없으면 빈칸 |
| wordnet_candidate_count | 전체 WordNet 유의어 후보 수 |
| wordnet_candidates | 전체 WordNet 유의어 후보. ` | `로 구분한 기본형 목록 |
| other_split_candidate_count | 반대 split에서 관측된 WordNet 유의어 후보 수 |
| other_split_candidates | Train 행에서는 Val, Val 행에서는 Train에 등장한 후보의 정규화 기본형 목록 |
| wordnet_synsets | 원본 단어를 해당 품사로 조회한 모든 synset ID. ` | `로 구분 |
| term_region | 이 행의 원본 단어 영역. CSV 파일의 영역과 일치 |
| term_coverage_status | DIRECT_LEMMA / WORDNET_CANDIDATE / UNCOVERED / EMPTY / UNCLASSIFIED |
| query_region | 원본 문장 전체를 분류했을 때의 영역. 문장에 미겹침 단어가 하나라도 있으면 A/C였던 기존 기준 |

후보 목록은 원본 단어의 모든 WordNet 의미를 포함합니다. 원본 자체·굴절만 다른 표현·철자 변형은 제외합니다.
전체 WordNet 후보에 반대 split 등장 여부를 요구하지 않습니다. 반대 split 후보는 실제 관측 lemma에 한정합니다.
후보 목록은 문장 내 굴절 전 기본형이며, synonym_expression은 원본 POS에 맞게 굴절된 개별 후보입니다.
반대 split 후보 목록은 굴절 등 실제 쌍 내보내기 필터 적용 전이므로 rank=0 행에도 목록이 있을 수 있습니다.
후보가 없으면 수는 0이고 목록은 빈칸입니다. 원본 단어가 없는 EMPTY 행은 조회 여부도 빈칸입니다.

한 행은 원본 표현 × 유의어 후보이며, 후보가 없는 표현도 빈 후보 행으로 보존합니다.
synonym_rank >= 1: 실제 반대 split 관측 근거가 있는 유의어 후보.
synonym_rank = 0: 유의어 쌍 없음. synonym_expression / meaning / 대상 표현·ID·캡션은 모두 빈칸.
EMPTY 쿼리는 source_expression도 빈칸인 한 행으로 보존합니다.
no_pair_annotations.csv의 csv_file + data_row_number(헤더 제외 1부터)로 빈 행의 사유를 찾을 수 있습니다.
query_membership.csv는 쿼리 × aspect × 출력 영역별로 실제 파일·단어·후보 수를 기록합니다.
한 쿼리 안의 단어가 서로 다른 영역에 속하면 해당 쿼리가 여러 행에 나타납니다.

A는 Train 단어 중 반대 split의 동일 lemma 및 같은 품사 WordNet synset 연결이 모두 없는 단어만 포함합니다.
C는 Val 단어 중 위 연결이 없는 단어만 포함합니다. A/C의 반대 split 후보 수는 모두 0이며 실제 유의어 쌍도 없습니다.
B는 반대 split과 동일 lemma 또는 같은 품사 synset으로 연결되는 단어입니다. 기존 A/C 문장의 겹치는 단어도 B로 이동합니다.
예: 기존 A 문장 안의 scuba는 A지만, chair/door/mask는 B입니다. door→doorway 행도 B입니다.
EMPTY는 해당 aspect의 추출 단어가 없는 쿼리만 포함합니다.
유효한 품사/lemma 분류가 없는 원본 행은 UNCLASSIFIED 파일에 따로 보존합니다. 이는 다이어그램 영역이 아닙니다.
B도 동일 lemma로 커버되면서 다른 유의어가 없는 표현이 있습니다. 따라서 B에서도 rank=0 행이 있을 수 있습니다.
수정된 다이어그램의 A/B/C 수는 품사·split별 고유 source_lemma 수입니다. EMPTY 수는 쿼리 수입니다.
한 쿼리의 서로 다른 단어가 A/B 또는 B/C로 나뉠 수 있으므로 영역별 고유 caption_id 수는 서로 중복될 수 있습니다.

B의 실제 후보 행의 기존 13개 열은 기존 B_synonym_pairs 파일에서 그대로 재사용합니다.
유의어 쌍은 동일 품사 synset 기준에서 반대 split의 다른 표현을 찾습니다. 원본 자체·굴절·철자 변형은 후보에서 제외합니다.
후보 표현은 원본 Penn POS에 맞게 굴절시킵니다. 원본 쿼리 문장은 보존하고, 파일의 영역만 단어 기준으로 배정합니다.
synonym_rank는 사전순 번호이며 문맥 적합도 점수가 아닙니다. 의미 설명은 공유 synset의 모든 뜻을 |로 연결합니다.
문맥·시각적 동작·구동사·문법·시간 구간 검증 전 후보입니다. 모든 requires_review=True입니다.
UTF-8 BOM CSV이며, 빈 후보는 추출 오류/WordNet 누락/상대 split 미등장/굴절 실패 등 여러 원인일 수 있습니다.

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/export_diagram_synonym_pairs.py
```
