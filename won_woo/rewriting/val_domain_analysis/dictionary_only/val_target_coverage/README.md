# 공식 사전 기준 Val 용어 조사

tvr_id_object_dict.json / tvr_id_action_dict.json의 항목만 사용한 diagram을 조사했습니다. LLM은 사용하지 않았습니다.

두 사전은 Train ID 87,174개만 포함하고 Val ID는 없습니다. Val은 동일 aspect 사전에 있는 정확한 표면형·POS 쌍만 추출했습니다.
따라서 사전 밖 Val 신규 어휘의 개수와 치환 가능성은 이 결과로 평가할 수 없습니다. C=0을 Val 전체의 신규 어휘 0개로 해석하면 안 됩니다.

Train 용어 집합은 공식 ID 사전과 연결된 원본 캡션 기준이며, 현재 수정된 Train 원문 전체의 재추출 결과가 아닙니다.

| 종류 | Train A | Train B | Val 용어 | Train 기본형 없음 | Val C | 유의어 후보 쌍 | 후보가 있는 원본 Train 캡션 |
|---|---:|---:|---:|---:|---:|---:|---:|
| object | 2,994 | 2,876 | 1,996 | 0 | 0 | 264,466 | 66,931 |
| action | 953 | 1,525 | 864 | 0 | 0 | 922,119 | 80,593 |

WordNet 후보는 문맥·동작·시간 구간을 검증한 치환 가능 건수가 아닙니다. CSV의 requires_review=True를 유지합니다.
이전 자동 POS 확장에 따른 596개 미포함 용어 / 95개 후보 용어 집계는 철회하고 이 사전 제한 결과로 대체합니다.

val_term_coverage.csv는 Val 기본형별 관측 표면형·근거·후보 수를 담습니다. 사전 밖 항목은 추출하지 않았습니다.
검증 근거는 diagram의 dictionary_scope_verification.json과 dictionary_terms.csv를 확인하세요.
