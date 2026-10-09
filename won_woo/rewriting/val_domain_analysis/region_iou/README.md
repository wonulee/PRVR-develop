# Object / Action Train–Val IoU

IoU = B / (A+B+C). A=Train 전용, B=공통 교집합, C=Val 전용이며, 각 기준의 동일 단위 집합으로 다시 계산했습니다.

| 기준 | 유형 | A | B | C | 합집합 | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lemma | object | 3,795 | 2,071 | 460 | 6,326 | 32.74% |
| selected_wordnet_synset | object | 5,632 | 7,048 | 596 | 13,276 | 53.09% |
| lemma | action | 1,526 | 951 | 136 | 2,613 | 36.39% |
| selected_wordnet_synset | action | 2,138 | 4,403 | 139 | 6,680 | 65.91% |

## 기존 A/B/C와 다른 이유

기존 WordNet 연결 기준 B는 Train에서 연결되는 기본형 수와 Val에서 연결되는 기본형 수를 각각 셉니다. 여러 표현이 같은 의미에 연결되므로 두 B 수가 다릅니다.
따라서 기존 A/B/C 기본형 수 중 Train B 또는 Val B 하나를 골라 B/(A+B+C)에 넣으면 집합 IoU가 아닙니다. 기존 영역 파일은 변경하지 않았습니다.
lemma 기준은 동일 기본형끼리만 교집합으로 셉니다. 기존 WordNet 기준 B에 있던 유의어의 서로 다른 기본형은 이 기준에서는 A 또는 C에 들어갈 수 있습니다.
selected_wordnet_synset 기준은 단어가 아니라 선택 의미 ID 집합의 겹침입니다. 한 기본형에 여러 의미가 있어 단어 수보다 많을 수 있습니다.
문맥 미검토 항목은 전체 같은 품사 의미를 유지하므로 synset IoU가 문장 문맥에서 검증된 의미 일치율을 뜻하지 않습니다.
선택 synset이 없는 기본형은 synset 집합에 포함되지 않습니다. 그 수는 methodology.json에 있습니다. lemma 기준에는 포함됩니다.
EMPTY / EXCLUDED / UNCLASSIFIED는 단어 집합에서 제외합니다. 고유 기본형 집합이므로 출현 빈도는 가중하지 않습니다.
단어 IoU를 보기 위해 Train/Val의 서로 다른 caption_id를 직접 교집합으로 세지 않았습니다. 이 수치는 검색 성능이나 A 학습의 유해성을 측정하지 않습니다.

[결과 CSV](object_action_ABC_iou.csv) · [계산 기준·입력 해시·검증](methodology.json)

재계산: `python rewriting/calculate_region_iou.py`
