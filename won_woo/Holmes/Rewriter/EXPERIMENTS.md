# 전체 MCA 검색 실험

기존 Holmes 체크포인트를 고정하고 **전체 validation 10,895개 쿼리 × 2,179개
비디오**를 평가합니다. 재학습은 하지 않습니다. 기본 재작성 모델은
Qwen2.5-7B-Instruct, 피처 모델은 로컬 TVR RoBERTa입니다.

## 바로 실행

실행 중 진행률과 상세 출력이 터미널에 바로 표시되며, 같은 내용이 `logs/`에도
저장됩니다. 학습에서는 Holmes의 배치별 loss, epoch별 R@K·Rsum·Best를 확인할 수 있습니다.


```bash
cd /home/viglab2/won_woo/Holmes

# 전체 실험: 샘플 수 제한 없음
bash scripts/run_mca_experiment.sh --device cuda:1
```

기본 출력은 `/home/viglab2/won_woo/experiments/mca_full/`입니다.
GPU 번호는 선택할 수 있습니다. 스크립트는 현재 설치된 holmes 환경의 Python을
사용하고, 모델은 기본적으로 로컬 캐시에서 읽습니다.

터미널을 닫아도 실행하려면:

```bash
nohup bash scripts/run_mca_experiment.sh --device cuda:1 \
  > ../mca_full.log 2>&1 &
tail -f ../mca_full.log
```

실행 준비만 검사하려면 `--check-only`를 붙입니다. 체크포인트를 바꾸는 예:

```bash
bash scripts/run_mca_experiment.sh \
  --checkpoint /absolute/path/to/best.ckpt --backbone multisrc \
  --device cuda:1 --output ../experiments/mca_another_checkpoint
```

`src` 체크포인트는 `--backbone src`를 사용합니다. 아키텍처와 가중치가 다르면
사전 검사에서 실패합니다. `PYTHON_BIN=/path/to/python`으로 환경을 바꿀 수 있습니다.

## 수행하는 비교

원본 피처를 같은 RoBERTa로 다시 추출한 뒤, 각 문맥 조건에서 모든 쿼리를 재작성하고
피처·검색 점수를 계산합니다. 원본/재작성에 동일한 후보 비디오 순서,
피처 모델, 레이어(-2), 길이(특수 토큰 포함 30)를 사용합니다.

| 문맥 조건 | 입력 |
|---|---|
| target-only | 대상 쿼리 하나 |
| oracle-context | 대상 쿼리 + 정답 비디오 ID로 묶은 나머지 캡션 |

| 평가 방법 | 점수 |
|---|---|
| original | 원본 Holmes clip/frame 가중 합 |
| rewritten | 채택된 재작성문의 점수로 교체 |
| average | 원본과 재작성 점수의 1:1 평균 |
| dst | clip/frame 각각 원본·재작성 증거를 DST 결합한 후 체크포인트 가중 합 |
| adaptive_dst | 불확실성이 임계값 이상인 쿼리에만 dst 적용 |

재작성 실패 또는 변화 없는 쿼리는 **모든 방법에서 원본 점수로 평가**합니다.
통과한 쿼리만 분모에 넣지 않습니다. 기본 결과표는 두 문맥 조건 × 다섯 방법의 10행입니다.
original 행은 두 문맥 조건에서 동일합니다.

전체 비교를 위해 모든 쿼리를 미리 재작성합니다. adaptive_dst는 이 캐시를 이용한
오프라인 선별 비교이며, LLM 호출 비용 절감을 측정하는 구현은 아닙니다.
재작성은 쿼리별 순차 생성이며 초과 길이에 한 번 재시도합니다.

## 데이터 분할과 기준선 해석

현재 로컬 `tvrval.caption.txt`와 `tvrtest.caption.txt`는 바이트 단위로 동일합니다.
기본 실행은 validation 전체를 한 번 평가하며, 독립 test 결과로 표시하지 않습니다.

별도의 test 데이터와 비디오 피처를 준비하면:

```bash
bash scripts/run_mca_experiment.sh --splits val test \
  --data-root /path/to/data-parent --output ../experiments/mca_val_test
```

분할 간 비디오 중복은 거부합니다. test 평가 시에도 임계값은 validation에서 정한
값을 고정합니다. 기본 임계값은 validation의 `max(u_clip, u_frame)` 상위 25%에
해당하는 분위수이며 `--ratio`로 변경합니다. 동률 때문에 실제 비율은 다를 수 있습니다.
validation 결과는 개발용 비교이며, 해당 분할에서 설정을 선택했다면 별도 held-out
평가가 필요합니다.

oracle-context는 정답 비디오에 속한 추가 캡션을 제공하는 조건입니다.
일반 단일 쿼리 검색 결과와 동등한 입력 조건으로 해석하지 마세요.
원본 피처도 재추출하므로 제공된 체크포인트의 논문 수치를 그대로 재현한다고
가정하지 않고, 이 실행에서 계산한 original 행을 기준선으로 사용합니다.

## 저장되는 결과

- `summary.csv`: 문맥/방법별 R@1, R@5, R@10, R@100, Rsum, 채택·융합 건수.
- `val.target-only.queries.csv`, `val.oracle-context.queries.csv`: 모든 쿼리의 방법별
  정답 순위, Top-1 비디오, 원본 불확실성, 적용 여부.
- `val.<문맥>/audit.jsonl`: 원문, 형제 문맥, 생성문, 거절 사유, 토큰 수.
- `*.npz`: 쿼리 ID·비디오 ID를 포함한 clip/frame 점수 캐시.
- `threshold.json`: validation에서 계산한 임계값.
- `experiment.json`: 실행 옵션, 입력 식별 정보, 코드 해시.
- `logs/`: 각 단계의 상세 진행 로그.
- `stages/`: 완료된 단계 기록.
- `complete.json`: 전체 비교표 생성 완료 표시.

## 중단과 재개

같은 명령을 다시 실행하면 완료된 단계를 건너뜁니다. 재작성은 audit에 기록된
쿼리를 재생성하지 않고 이어서 실행하며, 마지막 불완전한 기록은 제거하고 재시도합니다.
피처 추출·점수 계산은 중단된 단계부터 다시 시작합니다. 하나의 출력 디렉터리에서
동시에 두 실험을 실행하지 않도록 잠금을 사용합니다.

코드·캡션·모델 옵션 등 실험 설정을 변경했다면 새 `--output`을 지정해야 합니다.
GPU 번호는 재개할 때 변경할 수 있습니다. 기존 원본 데이터와 체크포인트는
수정하지 않습니다.

## 수식과 검증

증거 변환은 저장소의 `EDL_loss.get_alpha`와 같은
`e = exp(tanh(score) / tau)`, `alpha = e + 1`입니다.
`tau`와 clip/frame 가중치는 체크포인트에서 가져옵니다.
후보 수 `K`는 실제 평가 비디오 수입니다.

동일한 uniform base rate의 두 의견을 결합할 때 DST 증거는
`e_fused = e_original + e_rewritten + e_original * e_rewritten / K`로 계산하고,
`(e_fused + 1) / sum(e_fused + 1)`를 branch 점수로 사용합니다.
이는 Dempster 결합식과 동치이며 K×K conflict 텐서를 만들지 않습니다.
원본과 재작성문은 상관된 관측이므로 이 결합의 이득은 실험으로 판단해야 합니다.

```bash
python -m Rewriter.test_rewriter
python -m Rewriter.test_experiment
python -m Rewriter.test_experiment --gpu cuda:1
```

검사에는 DST의 직접 결합식과의 수치 일치, ID 재정렬, 재작성 실패 fallback,
재개, CSV 생성, 실제 체크포인트의 기존 평가기와 블록 처리 평가기 점수 일치,
단일 쿼리 마지막 배치가 포함됩니다.
