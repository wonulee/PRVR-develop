# Train / Val 채택 조건 전수 분석

이 보고서는 audit.jsonl에 실제 기록된 원문 flag와 채택 검사를 집계합니다. 원문 flag는 실제 수행된 변환 유형이나 채택 원인을 의미하지 않습니다. 각 채택 쿼리의 ID·원문·rewrite·flag 조합·통과 검사·LLM 설명은 split별 CSV에서 확인할 수 있습니다.

| 구분 | 전체 쿼리 | 채택 | 전체 대비 채택률 |
|---|---:|---:|---:|
| train | 87,174 | 11,064 | 12.69% |
| val | 10,895 | 1,290 | 11.84% |

## 채택된 쿼리의 원문 flag 조합

**분모: train 11,064개, val 1,290개. 아래 조합은 서로 배타적이며 split별 합계 100%입니다.**

| 원문 flag 조합 | Train 개수 | Train 비율 | Val 개수 | Val 비율 |
|---|---:|---:|---:|---:|
| 세 flag 모두 없음 | 4,606 | 41.63% | 588 | 45.58% |
| 대명사 | 5,629 | 50.88% | 617 | 47.83% |
| 부정어 | 86 | 0.78% | 3 | 0.23% |
| 감정·심리 | 55 | 0.50% | 7 | 0.54% |
| 대명사 + 부정어 | 409 | 3.70% | 47 | 3.64% |
| 대명사 + 감정·심리 | 201 | 1.82% | 18 | 1.40% |
| 부정어 + 감정·심리 | 6 | 0.05% | 0 | 0.00% |
| 대명사 + 부정어 + 감정·심리 | 72 | 0.65% | 10 | 0.78% |

세 flag가 없다는 것은 원문에 코드의 대명사·부정어·감정 keyword가 검출되지 않았다는 뜻입니다. Object flag가 없으므로 4가지 전략이 모두 미적용됐다는 의미로 해석할 수 없습니다.

## 개별 flag 포함 비율: 중복 허용

| 원문 flag | Train | Val |
|---|---:|---:|
| 대명사 | 6,311 (57.04%) | 692 (53.64%) |
| 부정어 | 573 (5.18%) | 60 (4.65%) |
| 감정·심리 | 334 (3.02%) | 35 (2.71%) |

flag는 코드의 keyword 정규식입니다. 감정·심리 목록은 upset/angry/happy/sad/afraid/worried/think/believe/want/feel 및 일부 활용형으로 제한되어 confused/shocked 등을 놓칩니다. 부정어 정규식도 cannot 같은 표현을 놓칩니다. 따라서 이는 모든 자연언어 감정·부정 표현의 의미 기반 분포가 아닙니다.

## 실제 공통 채택 검사

채택된 모든 쿼리는 비어 있지 않은 변경 문장, 최대 22단어, 특수 토큰 포함 최대 29 RoBERTa 토큰, temporal marker 검사, grounding schema/문자 포함 검사, 영상 내 채택 후보 중복 방지 검사를 통과했습니다. LLM verifier는 event/participant/negation/temporal 보존 true, sibling_event_imported false, unsupported_visual_details []로 판정했습니다. 모든 검사 통과 비율은 train과 val 모두 100%입니다.

최소 16단어는 강제하지 않았습니다. 16단어 미만 채택은 train 10,241개, val 1,203개입니다. 빈 grounding은 train 12개, val 0개이며 기존 코드가 허용한 결과입니다.

**통과 비율 100%는 코드상 판정 통과를 뜻하며, 의미 보존의 정확도 100%를 뜻하지 않습니다.** 이전에 검토한 val 의미 변경 사례는 CSV의 previously_reviewed_semantic_issue에 표시했고, 전체 오류율로 확대 해석하지 않습니다.

## 실제 4가지 변환별 비율의 해석 한계

현재 audit에는 applied_strategy / transformation_type / causal_acceptance_route가 없습니다. 따라서 “대명사 변환으로 채택된 비율”이나 “Object 강조로 채택된 비율”을 로그만으로 확정할 수 없습니다. 모든 쿼리에 rewriting을 시도하고 공통 검사를 통과하면 채택하는 구조입니다.

예: he를 생략한 문법 축약은 pronoun flag 감소여도 이름으로 대체한 해소가 아닙니다. upset → sad도 감정 flag가 있어도 구체 표정/제스처 변환이 아닙니다. 4가지 실제 유형을 보고하려면 원문/rewrite 변경 구간과 근거를 별도로 분류·검토해야 하며 그 추정 결과를 이 확정 집계와 구분해야 합니다.

## 재현

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python /home/viglab2/won_woo/experiments/acceptance_condition_analysis_20261003/analyze.py
```

원본 caption, audit, rewriting 결과, HDF5, checkpoint를 변경하지 않습니다. CSV는 Excel에서 열 수 있도록 UTF-8-SIG로 저장했습니다.
