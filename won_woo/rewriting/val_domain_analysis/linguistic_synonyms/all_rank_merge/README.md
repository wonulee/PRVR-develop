# TVR train: 원본 + 전체 순위 유의어 후보 merge

전체 후보 결과 파일: `/home/viglab2/won_woo/tvr/TextData/tvrtrain.caption_replace_merge_all.txt`

1순위만 포함한 파일은 `/home/viglab2/won_woo/tvr/TextData/tvrtrain.caption_replace_merge.txt`입니다. 원본 87,174개와 1순위 후보 1,365개를 합친 88,539개 caption입니다.

원본 `tvrtrain.caption.txt`의 87,174개 행을 보존하고, `tvrtrain_synonym_pairs.csv`의 모든 순위 후보로 생성한 문장 1,606개를 추가했습니다. 결과는 총 88,780개 caption입니다. 원본 파일과 후보 CSV는 변경하지 않았습니다.

## 생성·번호 규칙

- 각 후보는 CSV의 원본 caption에서 생성합니다. 치환 결과에 다른 치환을 연쇄 적용하지 않습니다.
- 같은 원문에 여러 표현의 후보가 있으면 가능한 모든 조합을 생성합니다. 예를 들어 action 후보 2개와 object 후보 2개가 있으면 4개 문장을 생성합니다.
- 원본 caption의 기존 ID를 유지합니다. 같은 video 안에서 원본의 최대 `enc` 다음 번호부터 후보 문장에 연속 번호를 부여합니다.
- 한 video의 원본 caption을 기존 순서로 놓고, 후보는 출처 caption 순서로 추가합니다. 같은 출처 caption에서는 조합의 최대 순위, 순위 조합, 후보 표현 순으로 정렬합니다.
- 같은 video 안에서 공백을 정규화한 문장이 이미 있으면 추가하지 않습니다. 실제 추가가 있을 때만 `enc` 번호를 증가시킵니다.
- 이전 사용자 선택에 따라 문장 길이 제한으로 후보를 제외하지 않습니다.

## 개수

| 항목 | 개수 |
| --- | ---: |
| 원본 caption | 87,174 |
| 최대 후보 순위 1인 추가 문장 | 1,365 |
| 최대 후보 순위 2인 추가 문장 | 221 |
| 최대 후보 순위 3인 추가 문장 | 20 |
| 전체 추가 문장 | 1,606 |
| merge 결과 | 88,780 |

2순위 유의어 쌍 218개, 3순위 유의어 쌍 20개는 모두 포함했습니다. 2순위 추가 문장이 221개인 이유는 여러 표현의 후보 조합을 함께 생성하기 때문입니다. CSV의 1,628개 쌍 중 실제 변경 가능한 1,625개 쌍을 포함했습니다. `pinkies → pinkies` 2건과 이름 `Irv` 내부의 `v → volt` 1건은 새 문장을 만들지 않습니다.

## 비교 CSV

`caption_comparison.csv`는 후보 조합마다 한 행입니다.

| 열 | 의미 |
| --- | --- |
| `source_caption_id`, `video_id`, `source_enc` | 출처 원본 caption과 video, 원래 번호 |
| `variant_index` | 출처 caption 안에서 생성한 후보 조합의 순서 |
| `added_caption_id`, `added_enc` | merge에 추가된 새 ID·번호; 미추가 행은 빈 값 |
| `max_synonym_rank` | 조합에 포함된 유의어 중 최대 순위 |
| `original_caption`, `candidate_caption` | 비교할 원문과 후보 문장 |
| `substitutions` | 표현별 원문·후보·순위·치환 횟수·의미·Val 예문 ID의 JSON 배열 |
| `status` | `added`: 추가, `unchanged`: 원문과 동일, `duplicate`: 같은 video의 기존 문장과 중복 |
| `duplicate_of_caption_id` | 동일 문장이 이미 있으면 해당 ID |

`synonym_pair_coverage.csv`는 입력 CSV의 쌍마다 한 행으로, 후보 순위와 실제 추가 문장에 포함된 횟수(`added_variant_count`)를 확인할 수 있습니다. `status`는 `included`, `same_expression`, `source_not_standalone_token`, `duplicate_caption`입니다.

`summary.json`은 개수·생성 규칙·입출력 SHA-256을 기록합니다. `tvrtrain.caption_replace_merge.before_all_ranks.txt`는 1순위 merge 파일의 백업이며, `tvrtrain.caption_replace_merge_all.preview.txt`는 전체 후보 결과와 같은 사본입니다.

새 `enc` ID를 학습에 사용하려면 해당 ID의 text feature를 추출해야 합니다.
