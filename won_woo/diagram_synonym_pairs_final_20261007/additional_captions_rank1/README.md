# 1순위 유의어만 사용한 Train 추가 캡션

추가 쿼리 **366,444개**를 생성했습니다. Object **185,934개**, Action **180,510개**입니다.
현재 원본 87,174개를 포함한 병합본은 **453,618개**입니다.
사용자 요청대로 길이·토큰·사전 재검사·POS·문맥 및 유의어군 필터는 적용하지 않았고, LLM을 사용하지 않았습니다.

## 파일

| 구분 | 추가 쿼리 CSV | 행 수 |
| --- | --- | ---: |
| Object | [object_additional_queries_0001.csv](object_additional_queries_0001.csv) | 185,934 |
| Action | [action_additional_queries_0001.csv](action_additional_queries_0001.csv) | 180,510 |

- [추가 캡션 TXT](tvrtrain.additional.caption.txt): 추가 쿼리 366,444개만 저장
- [원본 포함 병합 TXT](tvrtrain.caption.with_additional.txt): 원본과 추가 쿼리 453,618개
- [선택한 1순위 표현 대응](replacement_pairs.csv): Object 2,419개·Action 3,111개, 총 5,530개 대응
- [파일별 집계](file_counts.csv) · [검증 및 비활성화한 필터](verification.json)
- [치환 발생 연결](links/): 생성·중복 연결 372,393건을 두 CSV에 보존

## 1순위 선택 기준

순위는 [mapping/links/object_B_train.csv](../mapping/links/object_B_train.csv)와 [mapping/links/action_B_train.csv](../mapping/links/action_B_train.csv)의 **원본 `synonym_rank=1`**을 사용했습니다.
이 순위는 [원본 Object CSV](../csv/object_B_train.csv)와 [원본 Action CSV](../csv/action_B_train.csv)의 순위를 보존한 값입니다.

Object/Action별로 같은 정확한 출발 표현에 대치 표현 **하나**를 남겼습니다.
같은 출발 표현의 서로 다른 1순위 기록이 있는 경우에는 **원본 CSV에서 먼저 등장한 후보**를 선택했습니다.
선택한 표현에 연결된 모든 1순위 `mapping_id`는 유지했습니다. WordNet이나 LLM으로 순위를 다시 계산하지 않았습니다.

`replacement_pairs.csv`와 추가 쿼리 CSV에 `synonym_rank=1`, `rank1_source_data_row_number`를 기록했습니다.
후자의 행 번호는 해당 Object/Action 원본 CSV의 헤더를 제외한 1부터 시작하는 데이터 행 번호입니다.
`mapping_ids`는 [ABC_csv](../ABC_csv/README.md)의 정확한 표현·POS·대표 Val 캡션·WordNet 정보를 연결합니다.

## 쿼리 생성 방식

현재 Train 문장에서 출발 표현이 정확한 대소문자의 독립된 문자열로 나타나는 각 위치에 선택한 1순위 `replacement_expression`을 대치했습니다.
쿼리 하나에서 표현 하나의 한 위치만 바꾸었습니다. 원본 캡션에 서로 다른 출발 표현이나 여러 위치가 있으면 각각의 추가 쿼리가 생깁니다.
Object/Action 구분은 원래 CSV의 구분을 기록한 값이며 새 문장의 POS를 판단하지 않았습니다.

372,393개의 치환 발생 건에서 같은 영상의 생성 중 중복 5,949건을 합쳐 366,444개의 쿼리를 저장했습니다.
기존 Train 파일의 문장·ID·바이트는 그대로 유지했고, 병합본에만 추가 쿼리를 덧붙였습니다.
새 쿼리의 ID는 같은 영상의 기존 Train/Val/Test ID와 충돌하지 않도록 인코딩 번호를 늘렸습니다.

시간을 원본 release의 영상·치환 전 백업 문장으로 유일하게 확인한 경우 `source_timestamp`에 기록했습니다.
시간이 확인되지 않은 **783개도 제외하지 않았고**, `timestamp_status=unresolved_kept`로 표시했습니다.

CSV에 기록된 치환을 재구성할 수 있는지, 1순위의 원래 행·매핑이 맞는지, 표현별 후보가 하나인지, ID·행 수·파일 보존이 맞는지만 확인했습니다.
길이·사전·POS·문맥 품질 검사는 수행하지 않았습니다. 학습에 사용할 때는 이 버전의 추가 ID·문장에 맞는 쿼리 특징을 추출해야 합니다.

## 재생성

작업 공간 루트에서 실행합니다.

```bash
python rewriting/generate_additional_train_queries.py --no-rule-filters --rank-one-only
```

이전 전체 후보 생성본은 2026-10-09 정리에서 삭제했습니다. [규칙 적용 161개 비교 자료](../additional_captions/README.md)는 별도 실험의 감사 자료와 함께 유지했습니다.
