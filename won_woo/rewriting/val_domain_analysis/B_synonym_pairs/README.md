# B 영역 유의어 pair CSV

한 행은 원본 B 쿼리의 원본 Object/Action 표현과 반대 split에서 관측된 유의어 후보 한 쌍입니다.

- tvrtrain_synonym_pairs.csv: Train B → Val. 기존 linguistic_synonyms/tvrtrain_synonym_pairs.csv와 동일한 13개 열.
- tvrtrain_object_synonym_pairs.csv / tvrtrain_action_synonym_pairs.csv: 같은 결과를 Object/Action별 분리.
- tvrval_synonym_pairs.csv: Val B → Train. 대상 열은 train_expression / train_caption_id / train_caption.
- tvrval_object_synonym_pairs.csv / tvrval_action_synonym_pairs.csv: Val 쪽 Object/Action별 분리.
- B_query_pair_status.csv: 모든 B 쿼리 × aspect의 pair 수, 후보가 있는/없는 원본 표현, 건너뛴 추출 표현.
- summary.json: 집계, 건너뛴 사유, 방법, 입력 해시, 버전.

원본 표현과 원문을 보존하며 후보 기본형은 원본 Penn POS에 맞게 굴절시킵니다.
대상 표현과 대상 캡션은 반대 split의 실제 관측 근거입니다. 반드시 그 캡션에서 같은 의미로 사용된다는 보장은 없습니다.
원본 자체·단순 굴절·철자 변형은 이전 B 후보 생성 단계에서 제외했습니다.
synonym_rank는 굴절된 표현의 사전순 번호이며 의미 적합도 순위가 아닙니다. Top-k 제한 없이 후보를 내보냅니다.
meaning에는 공유 synset의 뜻을 모두 |로 연결합니다. 문맥에 맞는 의미를 하나 선택하지 않았습니다.

기존 reference 파일은 문맥 의미 선택과 점수·top-k를 사용합니다. 이번 파일은 같은 열 구조로 기존 B lexical 후보를 펼친 결과입니다.
모든 requires_review=True입니다. 실제 치환 전 문맥·가시적 동작·목적어/구동사·문법·시간 구간을 검토해야 합니다.
원본에 없는 표현, 품사 불일치, 실제 반대 split 근거 없음, 굴절 실패 등은 제외하고 사유를 집계합니다.
Pair가 없는 B 쿼리는 현황 CSV에 남아 있습니다. 기존 원본 파일과 B 분류는 변경하지 않습니다.
CSV는 Excel 호환 UTF-8 BOM입니다.

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/export_B_synonym_pairs.py
```
