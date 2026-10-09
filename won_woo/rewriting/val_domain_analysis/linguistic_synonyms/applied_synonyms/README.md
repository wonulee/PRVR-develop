# TVR train caption 유의어 적용 기록

입력은 상위 디렉터리의 `tvrtrain_synonym_pairs.csv`입니다. 각 caption의 표현별 `synonym_rank=1` 후보를 적용했습니다. 사용자 선택에 따라 단어 수·token 수로 변경 대상을 제한하지 않았으며, 유의어 외의 원문 문장 구성은 유지했습니다.

- 원본 caption: 87,174개
- 변경 caption: 1,365개
- 적용한 표현 쌍: 1,387개, 실제 치환 횟수: 1,399회
- 나머지 85,809개 caption 행은 바이트 단위로 동일합니다. 전체 ID·순서·줄바꿈을 유지했습니다.
- `pinkies → pinkies` 2건은 동일 표현입니다. `Irv` 안의 `v → volt` 1건은 독립된 단어가 아니므로 적용하지 않았습니다.

치환은 대소문자를 구분하고 단어 경계를 확인하여 한 번에 수행합니다. 한 표현이 같은 caption에 여러 번 나타나면 모든 일치 위치를 변경합니다. 치환된 결과를 다른 후보로 다시 치환하지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `tvrtrain.caption.original.txt` | 적용 전 원본의 전체 백업 |
| `tvrtrain.caption.rewritten.txt` | 적용한 전체 caption 파일과 동일한 사본 |
| `caption_changes.csv` | 변경된 caption의 원문·결과·치환 목록·단어 및 token 수 |
| `replacement_audit.csv` | 1순위 후보 전체의 적용 여부와 일치 횟수 |
| `summary.json` | 입력·출력 SHA-256, 파일 경로, 개수, 적용 상태 |

`caption_changes.csv`의 `substitutions`는 JSON 배열입니다. 각 항목은 `term_type`, `source`, `target`, `occurrences`, `meaning`, `val_caption_id`를 담습니다. `word_budget_pass`는 변경 후 14–20단어, `token_budget_pass`는 변경 후 30 미만 RoBERTa token 여부입니다. token 수에는 시작·종료 특수 token을 포함합니다. 이 두 열은 진단값이며 이번 적용의 필터로 사용하지 않았습니다.

`replacement_audit.csv`의 `status`는 `applied`(치환), `same_expression`(동일 표현), `source_not_standalone_token`(독립 단어로 일치하지 않음)입니다.

수정 대상은 `/home/viglab2/won_woo/tvr/TextData/tvrtrain.caption.txt`입니다. 기존 action/object 추출 사전 및 미리 계산된 text feature는 이번 작업에서 갱신하지 않았습니다. 변경된 문장을 학습에 반영하려면 사용하는 text feature를 다시 추출해야 합니다.
