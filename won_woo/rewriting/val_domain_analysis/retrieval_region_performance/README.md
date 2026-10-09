# TVR: A/B/C 영역과 검색 성능 비교

입력은 `/home/viglab2/won_woo/tvr_test_results.csv`입니다. 요청한 reslts 철자 파일은 없고 `tvr_test_results.csv`가 확인됐습니다.
10,895개 쿼리를 최종 문맥 검토 후 영역 CSV와 caption_id로 연결했습니다. Train ID와 겹치는 평가 쿼리는 0개입니다.
정답 순위는 1부터 시작하며 R@K는 정답의 순위가 K 이하인 쿼리 비율입니다. 원본의 네 성공 표시와 gt_rank가 모두 일치합니다.
원본 캡션과 다른 6건은 공백 차이뿐입니다. 현재 tvrval.caption.txt와 tvrtest.caption.txt의 내용 동일 여부는 True입니다.

## 비교 단위와 분류

- **A:** Train 단어 영역. 이 결과 CSV에는 Train inference가 없어 Recall은 미측정입니다. 0%로 해석하지 않습니다.
- **C_PRESENT:** 해당 쿼리의 Object/Action에서 최종 C 단어가 하나라도 등장하는 경우. 다른 B 단어가 함께 있어도 C로 분류합니다.
- **B_ONLY:** 해당 기준에서 C 단어가 없고 유효한 B 단어가 있는 경우.
- **EMPTY:** 그 항목의 추출 결과가 비어 있는 쿼리.
- **REVIEW_ONLY:** 추출 기록은 있으나 남은 표현이 EXCLUDED/UNCLASSIFIED뿐인 쿼리. EMPTY나 C로 넣지 않습니다.

Combined는 두 항목 중 하나라도 C이면 C_PRESENT입니다. 각 평가 쿼리는 각 비교 기준에서 한 번만 계산합니다.
Object C 701개와 Action C 85개 중 10개가 겹쳐 Combined C는 776개입니다.
기존 단어 영역 CSV의 B 쿼리 수에는 C 단어가 함께 있는 문장이 포함됐으므로 아래 B_ONLY 수와 다릅니다.

## 전체와 Combined 비교

모든 Recall 값은 %입니다.

| 평가 쿼리 구분 | 쿼리 수 | R@1 | R@5 | R@10 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 10,895 | 18.38 | 40.69 | 52.02 | 87.49 |
| B: C 단어 없음 | 10,119 | 18.25 | 40.65 | 52.15 | 87.56 |
| C: C 단어 포함 | 776 | 19.97 | 41.24 | 50.39 | 86.60 |

A는 Train 쿼리 중 8,024개에 A 단어가 있으나 이 CSV의 평가 쿼리는 0개입니다. Object A 포함 Train 쿼리는 6,484개, Action은 1,733개입니다.
이 숫자는 고유 기본형 단어 수(기존 다이어그램의 Object A 2,875 / Action A 882)와 구별해야 합니다.

## Object / Action 별 비교

| 기준 | 쿼리 영역 | 쿼리 수 | R@1 | R@5 | R@10 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Object | B_ONLY | 9,672 | 18.54 | 40.88 | 52.42 | 87.69 |
| Object | C_PRESENT | 701 | 19.69 | 41.94 | 50.93 | 86.88 |
| Object | EMPTY | 512 | 13.48 | 35.74 | 46.48 | 85.35 |
| Object | REVIEW_ONLY | 10 | 20.00 | 20.00 | 30.00 | 50.00 |
| Action | B_ONLY | 10,756 | 18.28 | 40.68 | 52.03 | 87.50 |
| Action | C_PRESENT | 85 | 23.53 | 36.47 | 47.06 | 84.71 |
| Action | EMPTY | 52 | 28.85 | 48.08 | 59.62 | 90.38 |
| Action | REVIEW_ONLY | 2 | 50.00 | 50.00 | 50.00 | 100.00 |

Object와 Action은 서로 다른 기준으로 같은 쿼리를 분석하므로 두 표의 쿼리 수를 합산하지 않습니다.

## C의 실패가 더 많은가

Combined C의 R@1 실패는 621/776개(80.03%), R@100 실패는 104/776개(13.40%)입니다.
B의 R@100 실패는 1,259/10,119개(12.44%)입니다.
지표별 Recall 차이의 방향과 크기는 다음 표에서 확인할 수 있습니다.

| 지표 | C−B Recall 차이(%p) | 클립 재표집 95% 구간(%p) |
| --- | ---: | --- |
| R@1 | +1.72 | [-1.26, +4.93] |
| R@5 | +0.59 | [-3.17, +4.44] |
| R@10 | -1.76 | [-5.46, +2.06] |
| R@100 | -0.96 | [-3.41, +1.60] |

동일 클립의 여러 캡션을 함께 재표집하는 video-cluster bootstrap 2,000회(seed 2026)를 사용했습니다.
Combined 네 지표 중 4개의 차이 구간이 0을 포함합니다. 이 데이터에서는 C의 실패가 일관되게 더 높다는 근거가 뚜렷하지 않습니다.
C 쿼리 672개는 실제로 R@100에서 정답을 찾았습니다. C의 WordNet 미연결 자체로 기존 학습에서 극복 불가능한 샘플이라고 결론 내릴 수 없습니다.
추가 학습으로 개선되는지는 동일 평가 쿼리에서 증강 전후를 비교하는 실험이 필요합니다.

## 해석 범위

최종 영역은 657개 문맥 검토를 반영했으며 나머지 품사와 WordNet 의미는 자동 조회 기준입니다. 영상을 직접 확인한 분류는 아닙니다.
UNCLASSIFIED가 함께 있는 쿼리는 287개이며 해당 쿼리를 제외한 민감도 표도 별도 저장했습니다.
Combined 평균 단어 수는 B 12.08, C 13.52입니다. 길이·TV 시리즈·동작 난이도를 통제한 인과 분석은 아닙니다.
재표집 구간은 기술적·탐색적 비교입니다. 구간에 0이 포함된다고 두 영역의 성능이 같다고 증명한 것은 아닙니다.
두 결과 CSV의 체크포인트 정보는 파일에 기록돼 있지 않습니다. 추가 CSV의 성능 차이를 특정 학습·증강 전략의 효과라고 해석하지 않습니다.
기록된 metrics.json의 비교는 저장된 점수 대조이며 이 파일들에는 영역별 쿼리 결과가 없어 영역별 개선을 계산하지 않았습니다.

## 파일

- [A/B/C 성능 다이어그램 PNG](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/A_B_C_performance_diagram.png) · [SVG](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/A_B_C_performance_diagram.svg) · [PDF](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/A_B_C_performance_diagram.pdf)
- [Recall 비교 그래프](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/region_recall_comparison.png)
- [영역별 전체 성능 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/region_recall.csv)
- [10,895개 쿼리의 영역·정답 순위·성과·C 단어·WordNet 후보](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/query_region_results.csv)
- [A/B/C 단어별 성능 CSV](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/term_recall.csv) — A는 평가 수 0 / Recall 빈칸. 한 쿼리가 여러 단어에 기여하므로 단어별 점수를 평균내 전체 점수를 계산하지 않습니다.
- [Object×Action 영역 교차표](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/object_action_matrix.csv)
- [C의 R@1 실패 621개](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/C_failure_queries_R1.csv)
- [C의 R@100 실패 104개](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/C_failure_queries_R100.csv)
- [C−B 차이·실패율·재표집 구간](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/C_vs_B_comparison.csv)
- [UNCLASSIFIED 동반 쿼리 제외 성능](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/region_recall_without_unclassified.csv)
- [다른 결과 CSV와 쿼리별 성능 대조](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/paired_result_comparison.csv)
- [저장된 전체 실험 성능 대조](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/recorded_performance_comparison.csv)
- [검증·방법·입력 해시](/home/viglab2/won_woo/rewriting/val_domain_analysis/retrieval_region_performance/summary.json)

쿼리 CSV의 `*_C_wordnet_candidates`는 전체 사전 의미 후보, `*_C_context_wordnet_candidates`와 `*_C_selected_wordnet_synsets`는 해당 기본형의 출현별 선택 의미를 합친 값입니다.
검토되지 않은 출현의 후보는 문맥에서 검증됐다는 의미가 아닙니다. C의 반대 split 후보는 없지만 자체 사전 후보는 있을 수 있습니다.
모든 출력은 압축하지 않은 CSV/그림입니다. 원본 평가·추출 데이터, 모델, 특징 파일, 학습 설정은 변경하지 않았습니다.

## 재현

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/viglab2/miniconda3/envs/holmes/bin/python rewriting/analyze_region_retrieval.py
```
