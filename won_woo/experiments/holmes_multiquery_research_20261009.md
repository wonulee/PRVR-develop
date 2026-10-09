**Holmes에서 multi-query로 성능을 높이는 방법 — 검토 및 실험 제안**

검토일: 2026-10-09. 논문, 현재 구현, 기존 실험 산출물을 확인한 연구 메모다. 아래 새 설계의 성능은 아직 검증하지 않았다. 모델 및 데이터는 수정하지 않았다.

가장 먼저 권하는 실험은 동결된 Holmes에 원문과 같은 사건의 재작성문 2개를 입력하고, 점수/순위 결합과 시간 구간의 합의를 비교하는 것이다. 그다음 중복 증거로 인한 과신을 제어하고, 이득이 확인되면 사건별 묶음을 반영한 학습으로 확장한다.

**확인한 논문과 구현**

- [Holmes 원문](https://arxiv.org/html/2605.06083v1): inter-video Dirichlet evidence, query-adaptive label calibration, intra-video FOT를 학습에 사용한다. 논문의 최종 검색은 frame/clip similarity의 가중합이다.
- `Holmes/multisrc/Models/holmes/model.py:293`: 두 query encoder는 동일한 입력 문장을 받는다. README의 dual-query branch는 외부 재작성문 여러 개를 사용하는 multi-query와 구별해야 한다.
- `Holmes/multisrc/Validations/validations.py:112`: native inference는 clip/frame score를 가중합한다. UGI, DST, FOT를 test-time query selector로 호출하지 않는다.
- `Holmes/Rewriter/evaluate.py:151`: original+rewrite 평균, DST, adaptive DST 비교가 이미 있으므로 기존 평가 경로를 확장할 수 있다.
- 논문 Eq.3은 `exp(tanh(s/tau))`이지만 `Holmes/multisrc/Losses/loss.py:53`과 `Holmes/src/Losses/loss.py:50`은 `exp(tanh(s)/tau)`이다. 기존 체크포인트 비교에서는 코드의 변환과 checkpoint tau를 유지하고 이 차이를 명시해야 한다.
- `graphify --help`를 먼저 시도했으나 명령이 설치 경로에서 발견되지 않아 파일 탐색과 직접 코드 읽기로 확인했다.

**선행연구가 뒷받침하는 범위**

Exa 검색 결과 20건을 두 검색 흐름에서 검토했다. 결과에는 같은 논문의 중복 URL이 포함된다. 직접 관련된 원문은 다음과 같다.

| 연구 | 확인한 결과 | 적용 범위 |
| --- | --- | --- |
| [GQE](https://arxiv.org/html/2408.07249v1) | MSR-VTT R@1: 원문 46.7, concatenation 45.4, feature mean 45.9, similarity mean 46.1, rank voting 49.6. 원문에 선택한 rewrite 2개를 더하면 50.2. | query 선택과 late fusion의 강한 baseline. 일반 T2VR 결과이며 Holmes 개선량으로 이전할 수 없다. |
| [Multi-Query Video Retrieval, ECCV 2022](https://arxiv.org/html/2201.03639v2) | 여러 실제 caption을 함께 사용하고 query aggregation을 학습한다. | 다른 순간의 caption과 같은 사건의 synthetic rewrite를 구별해야 한다. |
| [LLM-based query paraphrasing for video search](https://arxiv.org/html/2407.12341v1) | person/action/object 등의 QA 일치 검증으로 생성문을 필터링하고 검색 결과를 결합한다. | 표현의 cosine similarity에 더해 actor/action/object/시간 관계를 검증할 근거다. |

**1. 재작성 집합은 원문의 같은 사건을 유지한다**

q0는 원문이고 q1, q2는 동일한 actor/action/object/시간 관계를 유지한 전체 문장이다. 행동 표현과 객체 상호작용의 강조점을 달리할 수 있지만 필수 의미를 삭제한 object-only/action-only 문장은 첫 실험에서 사용하지 않는다. 이런 축약은 부정확한 video에도 쉽게 일치할 수 있다.

생성문은 사용자 규칙을 모두 적용한다: 객체의 시각적 상호작용 강조, 근거가 있는 명시적 명사로 대명사 해소, 부정의 의미를 유지하는 가시 상태 표현, 근거가 있는 표정/자세로 감정 표현, 14–20단어 및 실제 RoBERTa tokenizer의 subword token 30개 미만, 고유 action과 timestamp 보존. 특수 토큰을 포함한 모델 입력 길이도 별도로 기록한다.

원문에 없는 이름, 물체, 동작, 표정, 장소를 생성해 채워 넣지 않는다. 시각 상태를 확정할 근거가 없으면 후보를 거절한다. 원문은 항상 보존하고 필터 탈락도 전체 평가 분모에 남긴다. 재작성은 표현 때문에 생기는 모델의 이해 실패를 줄일 수 있지만, 원문에 없는 관측 정보를 제공하지는 못한다.

**2. 첫 비교는 score mean과 rank fusion이다**

비디오 표현은 한 번 캐싱하고, 모든 query는 동일한 encoder/checkpoint/feature layer/정규화로 처리한다. 원문과 생성문의 텍스트 feature를 같은 추출 경로에서 새로 계산해야 한다.

원문 점수 s0와 생성문 점수 sm의 결합은 다음처럼 시작할 수 있다.

`S(v) = (1-eta) * s0(v) + eta * sum_m w_m * s_m(v)`

`sum_m w_m = 1`, 후보가 없으면 `eta=0`이다. eta는 개발 분할에서 결정한다. 같은 수의 생성문을 넣는 균등 평균, 원문 비중을 높인 평균, rank voting/RRF를 비교한다. RRF는 새 제안이며 GQE의 voting과 동일한 방법이라고 주장하지 않는다. text embedding 평균과 문장 concatenation도 필요하면 대조군으로만 둔다.

**3. PRVR에서 가장 중요한 확장은 같은 시간 구간의 합의다**

clip branch가 계산하는 `s_m(v,t)`를 temporal max 직전에 보존한다. 현재 위치는 `Holmes/multisrc/Models/holmes/model.py:255`다. 동일한 사건의 rewrite끼리 다음 두 순서를 비교한다.

`A(v) = sum_m w_m * max_t s_m(v,t)`

`B(v) = max_t sum_m w_m * s_m(v,t)`

A는 query마다 서로 다른 순간에서 얻은 최고 반응을 합칠 수 있다. B는 동일한 시간 위치에서 여러 표현의 지지를 먼저 합친다. 비디오의 서로 다른 장면이 문장의 일부 조건만 만족하는 false positive를 줄일 수 있다는 가설이다. 이 식은 우선 clip branch에서 비교하고 frame branch 점수는 기존 계산을 사용한다.

같은 action의 object/state cue가 인접 프레임에 나타날 수 있으므로 exact argmax 일치만 강제하지 않는다. 인접 temporal window나 soft temporal distribution의 overlap도 비교한다. GT timestamp가 없는 설정에서는 predicted alignment agreement를 surrogate로 보고, 실제 grounding 정확도라고 주장하지 않는다. 원문의 localization이 잘못됐을 수 있으므로 원문과의 일치만을 hard filter로 삼지 않고 rewrite끼리의 합의도 함께 검사한다.

**4. 중복 evidence가 uncertainty를 낮추는 현상을 제어한다**

같은 query evidence를 M번 단순 합산하면 `u = K / (K + M * sum_v e(v))`가 되어 새 정보 없이 u가 감소한다. 표현이 비슷한 rewrite를 독립 관측으로 가정해 DST를 반복 적용하는 경우도 별도 검증이 필요하다.

보수적인 첫 evidence 결합은 clip/frame branch별로 다음처럼 정의한다.

`e*(v) = (1-eta) * e0(v) + eta * sum_m w_m * e_m(v)`

`alpha*(v) = 1 + e*(v)`

중복/거의 중복인 후보는 먼저 병합하고 묶음당 총 가중치를 고정한다. 단순히 모든 후보에 softmax 가중치를 주면 특정 후보의 복제본이 다른 후보 대비 총 비중을 늘릴 수 있다. 필요하면 의미 검증과 prediction/temporal disagreement에 따라 전체 evidence를 rho in [0,1]로 discount하는 변형을 비교한다. 이 결합을 정확한 Bayesian posterior라고 해석하지 않는다.

normalized evidence 평균은 의도적으로 query 수 자체를 strength 증가의 근거로 삼지 않는 baseline이다. similarity score 결합과 evidence 결합은 서로 다른 실험이며, 후자의 성능이 더 좋다고 가정하지 않는다. 후보 gallery와 evidence scale은 고정하고 calibration도 함께 평가한다.

**5. test-time routing은 정답 없이 구성한다**

Holmes UGI의 label consistency와 threshold 계산은 GT를 사용한다(`Holmes/multisrc/Losses/loss.py:372`). 이를 inference에 그대로 이식할 수 없다.

선택/생성 여부에는 원문 uncertainty, top1-top2 margin, clip/frame disagreement, rewrite 간 ranking/temporal agreement를 후보 신호로 사용한다. 학습용 정답으로 이 신호가 실제 오류 또는 개선 가능성을 예측하는지 검증하고, 평가 시에는 정답을 사용하지 않는다. 낮은 u만을 선택 기준으로 삼으면 광범위한 false positive에 높은 점수를 주는 과신 query를 고를 수 있다. 신호가 오류를 예측하지 못하면 고정 원문 비중의 결합을 유지한다.

생성 비용을 줄였다고 보고하려면 gate가 LLM 생성 전에 작동해야 한다. 모든 생성문을 미리 계산한 후 결합만 선별한 경우에는 offline selection의 비용과 실제 생성 비용을 구분한다.

**6. 학습 확장에서는 사건별 묶음을 반영한다**

현재 QDL은 같은 video의 query 쌍을 diagonal 제외하고 밀어낸다(`Holmes/multisrc/Losses/loss.py:195`). 원문과 같은 사건의 rewrite를 그대로 추가하면 서로 가까워야 하는 표현에 diversification penalty가 생긴다.

- `source_caption_id`에 따른 사건별 묶음 ID를 추가하고, 같은 묶음 내부 쌍은 QDL에서 제외한다. 같은 video의 서로 다른 사건 간 diversification은 유지한다.
- 묶음당 loss를 평균해 rewrite가 많은 사건이 학습을 지배하지 않게 한다. contrastive positive/negative mask에서도 같은 묶음과 같은 video 관계를 검토한다.
- 현재 FOT는 GT video의 모든 caption을 함께 배정한다(`model.py:100`). rewrite를 새로운 독립 row로 추가하면 OT marginal과 query 간 경쟁이 바뀐다(`partialot.py:184`). 원래 사건당 하나의 대표 query로 assignment를 만든 뒤 검증된 rewrite들의 temporal distribution을 그 target에 맞추는 방식을 먼저 비교한다. 대안은 사건별 총 marginal을 고정해 여러 rewrite에 나누는 것이다.
- inter-video prediction consistency와 intra-video temporal consistency를 작은 계수로 추가할 수 있다. 서로 다른 원래 timestamp의 caption에는 같은 temporal target을 강제하지 않는다.
- BatchPOT 반환 `pot_mass`는 row normalization 후의 합이다(`partialot.py:226`). 이것을 새 grounding confidence로 바로 해석하지 않는다.

multi-query teacher의 이득을 single-query student로 distill하면 inference query 수를 늘리지 않는 후속 방향도 가능하다. 의미 검증 및 검증된 temporal target을 먼저 확보해야 한다.

**기존 로컬 결과의 해석**

`tvr/TextData/effective_rewritten1.holmes_eval/metrics.json`에는 Val 10,895 queries / 2,179 videos에서 original SumR 192.281, 원문+rewrite 1:1 mean 199.073이 저장되어 있다. 같은 캐시에 대한 추가 CPU 진단은 rewrite-only 197.402, max 199.119이며, GT로 원문/rewrite의 더 좋은 rank를 선택하는 oracle은 211.849다. oracle은 성능 상한 진단이며 배포 가능한 selector의 성능이 아니다.

그러나 metrics가 가리키는 `effective_rewritten1.hdf5`가 현재 없고, 캐시 점수가 다른 query는 10,662개인데 인접 `tvrval.effective.caption.txt`에서 원문과 다른 텍스트는 1,296개다. 이 텍스트 파일이 캐시를 생성한 원본이라고 확정할 수 없다. 따라서 +6.792 SumR을 순수한 rewriting의 검증된 이득으로 귀속하지 않는다. 원문과 rewrite를 동일 경로로 재추출한 비교가 필요하다.

`experiments/multi_query_selection_20261008/README.md`에는 Train 원문 87,174개에 추가 1,371개를 선택한 결과가 있다. 2개 후보를 선택한 원문은 4개다. 선택에 Train GT와 Val 어휘를 사용했고 length filter도 현재 사용자 규칙과 다르므로 일반화 성능이나 새 증강 학습의 성공을 증명하지 않는다. 학습은 실행되지 않았다. 다만 개별 후보는 margin에 불리하더라도 두 후보의 집합은 유리할 수 있다는 사례는 집합 단위 선택을 검토할 근거다.

2026-10-09에 직접 SHA256을 비교해 로컬 `tvrval.caption.txt`와 `tvrtest.caption.txt`의 동일성을 확인했다. 둘 다 `ae9a60f6fc7581a3295e366d9a91c21b481d4411fb94b68422c09dc1751f3ef1`이다. 개발용 비교로 보고하고 독립 test 성능으로 표시하지 않는다. tuning을 한다면 비디오 단위 held-out 분할 또는 별도의 독립 test가 필요하다.

**권장 실험 순서와 판단 기준**

| 단계 | 비교 | 해결할 질문 |
| --- | --- | --- |
| 1 | 원문 / 원문+검증 rewrite 1개 / 원문+2개: score mean, rank fusion | 표현 다양성이 같은 feature 조건에서도 이득을 주는가? |
| 2 | 개별 temporal max 후 평균 / 같은 위치 또는 window에서 평균 후 max | 이득이 같은 사건에 대한 증거에서 오는가? |
| 3 | DST / 중복 병합+normalized evidence 평균 / disagreement discount | 과신을 늘리지 않고 결합할 수 있는가? |
| 4 | 고정 결합 / GT 없는 gate | 원문 성공 손상을 줄이고 호출 비용을 절약하는가? |
| 5 | 동일 수의 무작위 증강 / 선택 증강 / 사건별 QDL·FOT 수정 | 추가 학습이 single-query held-out 성능을 개선하는가? |

각 단계에서 R@1/5/10/100, SumR, 원문 Top1 성공의 손상률, 원문 실패의 복구율, 생성/채택 수, latency와 LLM 비용을 함께 기록한다. uncertainty에는 error detection AUROC 또는 risk-coverage를 사용하고, 확률을 보고하는 경우 calibration을 추가한다. 동일 query 반복 대조군은 중복 정보를 독립 증거로 세는 오류를 확인하는 데 필수다.

학습 비교는 초기 checkpoint, 총 optimizer step, 원래 사건의 sampling 빈도, 추가 query 수, seed를 맞춘다. 분할 seed와 생성 모델/prompt/tokenizer/feature/checkpoint 식별 정보를 고정하고 가능한 경우 3개 seed 및 비디오 단위 paired bootstrap으로 개선의 안정성을 확인한다. 생성 실패를 제외한 subset만으로 headline recall을 계산하지 않는다.

성능을 먼저 확인할 우선순위는 **원문 포함 3-query late fusion → temporal agreement → normalized evidence 및 gate → 사건별 묶음을 반영한 학습**이다. 가장 PRVR다운 연구 가설은 “동일 사건의 여러 표현이 같은 시간 구간을 지지할 때 evidence를 신뢰할 수 있다”는 것이다.
