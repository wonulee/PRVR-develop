# Val 유의어 후보 미발견 caption 분석

대상은 기존 `tvrtrain_synonym_candidates.csv`에서 `synonym_status=NO_VAL_SYNONYM`인 11,227개 caption입니다. 모든 OOD 표현의 후보 목록이 비어 있는 caption만 포함합니다. 일부 표현에 후보가 있는 `PARTIAL_VAL_SYNONYMS` 284개는 이 목록에서 제외했습니다.

원문·caption ID·순서를 유지했습니다. 기존 caption 파일과 유의어 CSV는 변경하지 않았습니다.

## 규모

- Caption: 11,227개
- OOD 표현 발생: 12,781건 — 같은 caption에서 같은 표현은 한 번 집계
- 서로 다른 `(action/object, 원문 표현)` 조합: 5,940개
- Action 표현 발생: 3,430건; object 표현 발생: 9,351건
- Object만 OOD: 7,914개 caption; action만 OOD: 2,755개; 둘 다 OOD: 558개

## 사유

| 사유 코드 | 해석 | 표현 발생 건수 | 포함 caption 수 |
| --- | --- | ---: | ---: |
| `no_wordnet_sense` | 기존 정규화 조회어와 해당 품사로 WordNet 의미를 조회하지 못함 | 1,825 | 1,766 |
| `weak_source_sense` | 원문 문맥과 가장 가까운 WordNet 정의의 cosine 점수가 0.20 미만 | 1,289 | 1,264 |
| `no_distinct_synonym_in_matching_val_usage` | 원문 의미 점수는 통과했지만 기존 Val 후보 생성 조건을 통과한 다른 표현이 없음 | 9,667 | 8,706 |

표현 발생 건수는 합계 12,781건입니다. 같은 caption에 서로 다른 실패 사유가 함께 있을 수 있으므로 caption 수 열은 합산하지 않습니다.

마지막 사유는 Val에서 관측된 표현과 문맥, 동일 WordNet 의미, 같은 단어·철자 변형 제외, 반의어 제외, 원문 품사로의 형태 조정 조건을 묶은 결과입니다. 이번 분석에서는 원문 측 의미와 점수를 재현하고 기존 빈 후보 목록을 이용해 이 잔여 사유를 분류했습니다. Val 후보 탐색 전체를 다시 실행하지 않았습니다.

기존 분석과 같은 spaCy, WordNet, MiniLM 모델과 seed 2026을 사용했습니다. 부분 후보 caption의 실패 표현까지 포함한 전체 사유 합계가 기존 `linguistic_synonyms/summary.json`의 9,918 / 1,865 / 1,330건과 정확히 일치하는지 검증했습니다.

## 원문 조회와 정규화 조회의 차이

198건의 표현, 197개 caption에서 원문 표현으로는 WordNet 의미를 찾을 수 있지만 기존 정규화 조회어로는 찾지 못했습니다. 예:

- `binoculars → binocular`: 45개 caption
- `whereabouts → whereabout`: 20개 caption
- `mess → mes`: 11개 caption
- `express → expres`: 10개 caption

이 사례는 `raw_surface_has_sense_but_lookup_does_not`로 표시했습니다. 원문 조회의 의미가 존재한다는 사실만으로 Val 안의 적절한 유의어가 보장되지는 않지만, 후보 부재를 검토할 때 조회어부터 확인할 필요가 있습니다.

## 파일과 주요 열

| 파일 | 용도 |
| --- | --- |
| `tvrtrain_no_val_synonym_captions.csv` | 11,227개 원문 caption, OOD 표현, 사유와 검토 표시를 한 행에 정리 |
| `tvrtrain_no_val_synonym_terms.csv` | 표현별 12,781행; 문맥·실제 조회어·품사·선택 의미·점수·사유 |
| `no_val_synonym_word_frequency.csv` | 5,940개 표현의 빈도·사유 분포·최대 3개 예문; 빈도 내림차순 |
| `failure_reason_summary.csv` | 사유별 표현 발생 수·caption 수·action/object 수 |
| `summary.json` | 통계, 검증 결과, 입력 파일 SHA-256과 재현 설정 |

`source_expression`은 원문 표현이고 `lookup_expression`은 기존 분석이 WordNet 조회에 사용한 정규화 형태입니다. `wordnet_sense_count`는 조회어의 의미 수, `raw_surface_wordnet_sense_count`는 원문 표현을 소문자로 조회했을 때의 의미 수입니다. `source_sense_score`는 원문 문맥과 선택한 정의의 embedding cosine 점수이며 Holmes의 evidential uncertainty 값은 아닙니다. 조회 가능한 의미가 없으면 의미와 점수 열을 비워 둡니다.

`missing_expression_details`, `failure_reason_codes`, `review_flags`, 빈도 파일의 예문·사유 분포 열은 JSON입니다. Caption 파일의 상세 JSON에는 표현별 원문·조회어·품사·문맥·선택 의미·점수·한국어 사유가 포함됩니다.

검토 표시는 확정된 오류 판정이 아닙니다.

- `source_not_standalone_token`: 단어 경계 기준으로 원문에서 표현을 찾지 못함
- `single_letter_expression`: 한 글자 알파벳 표현
- `uppercase_expression`: 모두 대문자인 표현; 약어 등 문맥 검토
- `source_pos_differs_from_spacy`: 제공 사전 품사와 이번 spaCy 태그가 다름
- `raw_surface_has_sense_but_lookup_does_not`: 원문 조회에는 의미가 있지만 정규화 조회에는 없음

현재 기준으로 후보를 찾지 못했다는 결과이며, 영어에 유의어가 없거나 해당 caption의 영상 의미가 새롭다는 결론으로 사용하지 않습니다.

재현:

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/summarize_missing_val_synonyms.py
```
