# A 영역의 학습 노출과 Val 원문 출현 진단

검증할 가설은 “Train 전용 A 표현에 대한 학습이 Val 검색에 악영향을 주는가”입니다. 앞선 Val B/C 성능 비교는 이 질문을 직접 검증하지 않습니다.
Val에 없다는 이유만으로 유해한 학습 표현이라고 확정하지 않습니다. 이 보고서는 검토 후보와 분류 문제를 찾는 데이터 진단입니다.

- A 포함 Train 쿼리: 8,024/87,174 (9.20%).
- A만 있는 쿼리: 30. A와 B가 함께 있는 쿼리: 7,994.
- 고유 A 기본형: 3,757. 한 Train 쿼리에만 나타나는 A 기본형: 2,317 (61.67%).
- 추출 쿼리×유형(Object/Action)×기본형 출현 중 A 비중: 2.51%. 실제 학습 손실·토큰 비중은 아닙니다.
- A 중 Val 원문에 같은 기본형/관측 표현 문자열이 있는 항목: 436. 없는 항목: 3,321.

현재 A는 **반대 split에서 추출된 동일 품사 기본형·선택 WordNet 의미와 연결되지 않는 항목**입니다. Val 원문에서 해당 문자열이 완전히 없다는 뜻과 구별해야 합니다.
Val 원문에서 발견된 A에는 다른 품사·다른 의미일 가능성, 고유명사 제외, 품사 오분류·추출 누락이 함께 있을 수 있습니다. 문자열 출현만으로 B로 재분류하지 않았습니다.
A 포함 쿼리를 통째로 삭제하면 대부분의 쿼리에서 B 학습 신호와 해당 영상·시간 구간의 감독도 함께 줄어듭니다.

## 영향 검증에 필요한 비교

| 실험 | 비교 조건 | 목적 |
| --- | --- | --- |
| 기준 학습 | 원래 Train과 설정 유지 | 기준 Val R@1·5·10·100 |
| A 개입 | A 포함 쿼리의 학습 기여를 줄임 | A 포함 감독의 영향 측정 |
| 대조 개입 | 같은 규모의 B 쿼리에 같은 기여 감소; 길이·시리즈·영상 분포를 맞춤 | 단순 학습량 감소 효과와 구별 |

같은 초기 가중치, 학습 seed, 업데이트 수, 평가 캡션·특징·영상 후보를 유지하고 반복 seed의 결과를 함께 확인해야 합니다.
A 개입이 기준 학습과 대조 개입보다 개선될 때 A 포함 감독의 부정적 전이 가능성을 검토할 수 있습니다. 단일 A 단어가 원인이라는 증명은 아닙니다.
단어 자체의 영향을 보려면 동일 영상·타임스탬프의 객체·동작 의미를 보존한 표현 대체가 필요합니다. 임의 삭제로 고유 동작을 바꾸지 않습니다.
Holmes에서는 검색 Recall과 함께 Val 실패 시 evidential confidence/uncertainty도 비교하면 학습 표현 편향이 과신으로 이어지는지 확인할 수 있습니다.
이 단계에서는 학습 코드, 손실, 쿼리 원문과 모델을 수정하거나 재학습하지 않았습니다.
이 진단은 현재 `tvrtrain.caption.txt`와 `tvrval.caption.txt`를 기준으로 합니다. 평가 CSV를 만든 체크포인트의 실제 학습 캡션·텍스트 특징이 이 입력과 동일한지는 확인되지 않았습니다.
현재 Val과 Test 캡션 파일은 동일합니다. 이 Val에 맞춰 A를 선택하는 실험 결과는 별도의 독립 평가 split에 대한 일반화 근거가 아닙니다.

## 파일

- [A 단어별 학습 빈도·Val 원문 출현·WordNet 후보](/home/viglab2/won_woo/rewriting/val_domain_analysis/train_A_exposure/A_word_exposure.csv)
- [A 포함 Train 쿼리 8,024개와 A/B 구성](/home/viglab2/won_woo/rewriting/val_domain_analysis/train_A_exposure/A_train_queries.csv)
- [Val 원문에서 같은 관측 표현/기본형 문자열을 찾지 못한 A 단어](/home/viglab2/won_woo/rewriting/val_domain_analysis/train_A_exposure/A_words_without_literal_val_match.csv)
- [수치·정의·입력 해시](/home/viglab2/won_woo/rewriting/val_domain_analysis/train_A_exposure/summary.json)

검토 후보이며 “이 단어를 학습하면 성능이 낮아진다”는 판정 목록이 아닙니다. 실제 Train 손실·gradient나 A 개입 후 모델의 Val 결과는 현재 CSV에 없습니다.
최종 A 분류의 657개 문맥 검토 밖 항목은 자동 POS/WordNet 조회 기준입니다. 실제 문자열 부재 검사는 어휘 진단이며 영상 의미 검증은 아닙니다.
