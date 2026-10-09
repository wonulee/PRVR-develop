# 다이어그램 영역별 synonym-pairs 형식 CSV — 공식 사전만 사용

최종 형식 기준: `/home/viglab2/won_woo/diagram_synonym_pairs/object_B_val.csv`의 37열 구성과 순서.
[전체 파일 집계](file_counts.csv) · [다이어그램 PNG](object_action_regions.png) · [SVG](object_action_regions.svg) · [PDF](object_action_regions.pdf)

`dictionary_scope_verification.json`에서 모든 사전 원본 항목 보존과 사전 밖 용어 0건을 검증했습니다.
`reference_format_verification.json`에서 16개 CSV의 기준 열 일치와 다이어그램 집계를 확인했습니다.

Object / Action 각각 A_train, B_train, B_val, C_val, EMPTY_train, EMPTY_val 파일로 나눕니다.
UNCLASSIFIED_train / UNCLASSIFIED_val도 같은 형식으로 보존합니다.
모든 영역은 기준 object_B_val.csv와 같은 37개 열을 사용합니다.
기존 13개 쌍 열 + WordNet 7개 열 + 영역 4개 열 + 검토 이력 8개 열 + 공식 사전 근거 5개 열입니다.
Val 파일은 대상 열을 train_expression / train_caption_id / train_caption으로 바꿉니다.
Train 파일의 대상 열은 val_expression / val_caption_id / val_caption이며, 나머지 열과 순서는 동일합니다.
file_counts.csv에 모든 파일의 행·후보·빈 후보·고유 기본형 수를 기록합니다.
csv_schema.json에 두 split의 열 이름과 순서를 기록합니다.

| 추가 열 | 의미 |
| --- | --- |
| source_lemma | 원본 표현을 해당 Object/Action 품사로 정규화한 조회 단어 |
| has_wordnet_entry | 같은 품사 WordNet synset 존재 여부. True/False, 조회할 단어 또는 지원 품사가 없으면 빈칸 |
| wordnet_candidate_count | 동일 aspect 공식 사전에 실제로 존재하는 WordNet 후보 표현 수 |
| wordnet_candidates | 공식 사전의 실제 표면형으로 제한한 후보 목록. ` | `로 구분 |
| other_split_candidate_count | 반대 split에서 관측된 WordNet 유의어 후보 수 |
| other_split_candidates | 반대 split에서 관측되고 같은 aspect 공식 사전에 존재하는 후보 표면형 |
| wordnet_synsets | 원본 단어를 해당 품사로 조회한 모든 synset ID. ` | `로 구분 |
| term_region | 이 행의 원본 단어 영역. CSV 파일의 영역과 일치 |
| term_coverage_status | DIRECT_LEMMA / WORDNET_CANDIDATE / UNCOVERED / EMPTY / UNCLASSIFIED |
| query_region | 원본 문장 전체를 분류했을 때의 영역. 문장에 미겹침 단어가 하나라도 있으면 A/C였던 기존 기준 |
| original_source_lemma | 교정 전 기존 spaCy 기본형. 현재 source_lemma와 비교해 기본형 변경을 확인 가능 |

같은 품사 WordNet 형태 분석 결과 기본형이 하나면 그 기본형을 사용합니다. miss→mis, access→acces처럼 잘못 잘린 기본형을 교정합니다.
기본형 후보가 둘 이상이면 원래 Penn POS로 굴절했을 때 원래 표현으로 돌아오는 후보가 하나인지 확인합니다.
예: sing→singing, singe→singeing이므로 singing/VBG는 sing으로 교정합니다.
이 방법으로도 하나를 정할 수 없거나 WordNet에 없으면 기존 기본형을 유지합니다. 이 경우 문맥에 따른 의미·기본형 검토가 필요할 수 있습니다.

추출과 후보의 유일한 허용 어휘는 tvr_id_object_dict.json / tvr_id_action_dict.json입니다.
Train은 caption ID의 원본 항목·품사를 그대로 보존합니다. 두 사전에는 Val ID가 없으므로
Val은 동일 aspect 사전에 존재하는 정확한 표면형·Penn POS 쌍만 선택합니다.
새 단어·새 굴절형·대소문자 변형을 추가하지 않으며 Object↔Action 이동도 허용하지 않습니다.
후보 목록은 원본 단어의 WordNet 의미와 연결되는 공식 사전 항목으로 제한합니다.
원본 자체·굴절만 다른 표현·철자 변형은 제외합니다. 반대 split 후보는 실제 관측 항목에 한정합니다.
synonym_expression도 원본 POS를 가진 공식 사전의 기존 표면형 중에서 선택하며 생성하지 않습니다.
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
유효한 품사/lemma 분류가 없는 행, 문장에 실제 토큰으로 존재하지 않는 표현, 구두점만 남은 lemma는 UNCLASSIFIED에 보존합니다.
예: 's를 잘못 정규화한 ' 또는 명사로 잘못 추출된 -는 C가 아니라 검토 대상입니다. 이는 다이어그램 영역이 아닙니다.
기본형으로 해소되지 않은 're 등의 축약형과 잘린 보조동사 s도 UNCLASSIFIED에 보존합니다.
being/children처럼 아직 굴절형이 남아 기본형으로 확인되지 않는 lemma도 UNCLASSIFIED에 보존합니다. 해당 원본의 철자·품사를 먼저 검토해야 합니다.
didn't 안의 did처럼 spaCy가 실제로 분리한 토큰은 유효한 관측으로 인정합니다. 단순 문자열 일부가 존재하는 것만으로 인정하지 않습니다.
B도 동일 lemma로 커버되면서 다른 유의어가 없는 표현이 있습니다. 따라서 B에서도 rank=0 행이 있을 수 있습니다.
수정된 다이어그램의 A/B/C 수는 품사·split별 고유 source_lemma 수입니다. EMPTY 수는 쿼리 수입니다.
한 쿼리의 서로 다른 단어가 A/B 또는 B/C로 나뉠 수 있으므로 영역별 고유 caption_id 수는 서로 중복될 수 있습니다.

모든 실제 후보 행은 교정된 기본형과 원본 split의 실제 표현·근거 문장에서 다시 계산합니다.
유의어 쌍은 동일 품사 synset 기준에서 반대 split의 다른 표현을 찾습니다. 원본 자체·굴절·철자 변형은 후보에서 제외합니다.
후보 표현은 원본 Penn POS와 일치하는 공식 사전의 기존 표면형에서 선택합니다. 원본 쿼리 문장은 보존하고, 파일의 영역만 단어 기준으로 배정합니다.
후보가 공식 사전의 해당 aspect·POS에 없으면 내보내지 않습니다.
synonym_rank는 사전순 번호이며 문맥 적합도 점수가 아닙니다. 의미 설명은 공유 synset의 모든 뜻을 |로 연결합니다.
문맥·시각적 동작·구동사·문법·시간 구간 검증 전 후보입니다. 모든 requires_review=True입니다.
UTF-8 BOM CSV이며, 빈 후보는 추출 오류/WordNet 누락/상대 split 미등장/굴절 실패 등 여러 원인일 수 있습니다.

dictionary_file과 source_dictionary_caption_id는 원본 항목의 사전 근거입니다.
synonym_dictionary_caption_id는 synonym_expression의 사전 근거,
target_dictionary_caption_id와 target_source_pos는 반대 split 표현의 사전 근거입니다.
dictionary_terms.csv는 실제 사전 표면형·POS·기본형·근거 ID를 연결합니다.
source_lemma는 형태 정규화 값이며 새로운 추출 항목이 아닙니다.
final_term_regions.csv는 이 사전 제한 범위의 A/B/C 기본형 집계입니다.

두 사전은 Train 전용이므로 이 범위에서 C가 0이어도 Val 전체에 새로운 단어가 없다는 뜻은 아닙니다.
사전 밖 Val 단어는 추출하지 않았으므로 그 개수와 유의어 전환 가능성은 이 결과로 판단할 수 없습니다.
이전 자동 POS 확장 및 Object→Action 교정 결과와 그에 따른 집계는 이 결과로 대체합니다. LLM은 사용하지 않았습니다.

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/export_diagram_synonym_pairs.py
```
