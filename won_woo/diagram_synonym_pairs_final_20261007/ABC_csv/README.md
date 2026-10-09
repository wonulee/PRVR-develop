# A/B/C Object·Action 보기용 CSV

한 행에서 Train/Val 표현·POS, 대표 캡션 ID·문장, WordNet 후보·synset·의미를 함께 확인합니다. 기존 표현별 매핑 20열에 대표 문장 2열, 의미 1열, WordNet 8열을 붙인 31열 CSV입니다. Excel에서 바로 열 수 있도록 UTF-8 BOM으로 저장했습니다.

| 영역 | Object | Action |
| --- | --- | --- |
| A·Train | [3,316행](object_A_train.csv) | [1,250행](action_A_train.csv) |
| B·Train | [6,525행](object_B_train.csv) | [12,375행](action_B_train.csv) |
| B·Val | [4,779행](object_B_val.csv) | [7,648행](action_B_val.csv) |
| C·Val | [0행](object_C_val.csv) | [0행](action_C_val.csv) |

EMPTY·UNCLASSIFIED도 별도 CSV로 보존했습니다. [파일별 행 수](file_counts.csv)와 [검증 결과](verification.json)를 확인할 수 있습니다.

행 수는 표현·POS 대응을 중복 제거한 매핑 수입니다. B에서 한 표현이 여러 표현과 연결되면 여러 행입니다. `full_rows`는 묶인 원본 행 수, `source_caption_count`는 출발 캡션 수입니다. 전체 16개 파일의 35,930행과 기존 20열의 값·순서를 그대로 보존했습니다.

`example_train_caption`과 `example_val_caption`은 실제 연결된 대표 문장입니다. 전체 캡션 연결은 같은 파일명의 [mapping/links](../mapping/links/) CSV에서 `mapping_id`로 확인하며, 문장은 [captions.csv](../mapping/captions.csv)의 `caption_key`로 찾습니다. 원본 1,494,233행에 대한 전체 연결 자료는 그대로 유지했습니다.

WordNet 열은 출발 split의 기존 조회 결과입니다. `mapping_type`의 `WORDNET_SYNONYM`은 기존 후보, `DIRECT_LEMMA_REFERENCE`·`WORDNET_COVERAGE_REFERENCE`는 B영역의 실제 반대편 표현 참고 연결입니다. 참고 연결의 `replacement_expression`은 비어 있습니다. `requires_review` 조건도 그대로 유지했습니다.

기존 JSON 사전의 동일 Object/Action 표현·POS 범위를 유지했습니다. C·Val=0은 이 제한 범위의 결과입니다. LLM을 사용하지 않았고 캡션을 수정하지 않았습니다.

재생성: `python rewriting/export_ABC_readable_csv.py` (작업 공간 루트에서 실행).
