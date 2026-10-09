# Evidence Risk-Aware PRVR

고정 A3PRVR/Holmes 검색기에 evidence 위험도 기반 재작성·후보 선택·원문 anchor 융합을
추가한 **inference-only 연구 코드**입니다. 원본 모델, 학습 손실, 데이터, checkpoint를
수정하지 않습니다. `--use-method`를 지정하지 않으면 native 원문 검색만 실행합니다.

## 구현과 실제 A3PRVR 코드 대응

공식 소스: [chuanshen-chen/A3PRVR](https://github.com/chuanshen-chen/A3PRVR),
commit `ae02a70f7d6d5320531df0113ee48e47712a072d`.
원본은 workspace의 `third_party/A3PRVR/`에 변경 없이 보관했습니다.

공식 TVR checkpoint는 `checkpoints/a3prvr/tvr.ckpt`입니다.
출처: https://huggingface.co/chadcs/A3PRVR_AAAI2026/resolve/main/tvr.ckpt
SHA256: `ac2f20338dc203b0c0882d9c198a34f1bf148a531d88dd42ced38cbe5b057665`.
공개 파일에는 `model`과 `epoch`만 있으며 `model_cfg`가 없습니다.
`tvr_config.json`은 **공식 parser 기본값 + do_tvr.sh의 모델 계산 인자**로 재구성했으며,
모든 state key/shape가 strict loading으로 일치하는 것을 확인했습니다.
별도 학습 checkpoint를 사용할 때는 그 실험의 실제 config를 `--a3-config`로 지정하세요.

A3PRVR 실제 흐름:

```
I3D/ResNet [V,T,3072] → normalize/sample → encoder [V,T,384]
CLIP       [V,T,512]  → normalize/sample → encoder [V,T,384]
→ 공식 cross-branch Q-DTA
query token [B,L,768] → 두 query encoder → [B,384]
→ 분기별 projected cosine, valid temporal max → [B,V]
→ fusion_weight * I3D_score + (1-fusion_weight) * CLIP_score
```

`phrase_action_branch=True`가 코드에서는 CLIP 경로입니다.
`topk`를 inference pooling에 적용하지 않습니다. native inference는 max입니다.
adapter의 explicit `einsum`은 native `squeeze/bmm`와 같은 수식이며 Q=1/V=1도 유지합니다.
Q-DTA의 temporal length 1은 native normalize_grid에 문제가 있어 임의 padding을 하지 않고
명확히 거부합니다. 단일 **비디오 수**와 단일 **temporal position**은 다른 조건입니다.

로컬 I3D/CLIP 파일에는 현재 train/val의 모든 video key와 기대 차원이 있습니다.
두 개의 로컬 HDF5 또는 upstream joint HDF5의 `i3d_feat`/`clip_feat` group을 지원합니다.
로컬 피처가 공식 28GB joint 파일과 값 단위로 동일한지는 검증하지 않았습니다.
따라서 제공된 논문 수치 재현을 주장하지 않고, 같은 입력으로 계산한 original을 baseline으로 씁니다.

## 구성 파일

- `core.py`: G/D/OOD calibration, 위험도, controller, candidate utility, selection/fusion.
- `language.py`: 고정 LLM 분해·생성·judge prompt와 source span/graph/lexical 검증.
- `backbones.py`: native A3PRVR/Holmes adapter와 RoBERTa token/contextual span feature.
- `run.py`: train-only bank 생성, method ON/OFF 검색, audit·scores·metrics 저장.
- `test_method.py`: 수식·fallback·보존 검사 및 실제 checkpoint baseline 비교.

수식과 실험 계획: `../../experiments/evidence_risk_rewriting_design.md`.
`example_config.json` 수치는 **초기 실험을 위한 예시이며 학습/최적화된 값이 아닙니다**.
모든 meaningful method parameter를 JSON으로 고정하며 train/dev에서 결정해야 합니다.

## 실행

workspace에 설치된 `holmes` 환경과 로컬 모델 캐시를 사용합니다.

```bash
cd /home/viglab2/won_woo/Holmes

# 1. train에서 reference video/evidence bank 생성. held-out video 중복을 거부합니다.
bash scripts/run_evidence_retrieval.sh build-bank \
  --config EvidenceRetrieval/example_config.json \
  --reference-videos 64 --bank-queries 128 \
  --output ../experiments/evidence_a3_bank

# 2. 같은 native token feature로 baseline 실행. LLM/bank를 로드하지 않습니다.
bash scripts/run_evidence_retrieval.sh run \
  --captions ../tvr/TextData/tvrval.caption.txt \
  --features ../tvr/TextData/roberta_tvr_query_feat_extract.hdf5 \
  --output ../experiments/evidence_a3_original

# 3. 새 방법: 원문 점수와 최종 점수를 같은 실행에서 모두 저장합니다.
bash scripts/run_evidence_retrieval.sh run --use-method \
  --config EvidenceRetrieval/example_config.json \
  --bank ../experiments/evidence_a3_bank/bank.npz \
  --captions ../tvr/TextData/tvrval.caption.txt \
  --features ../tvr/TextData/roberta_tvr_query_feat_extract.hdf5 \
  --output ../experiments/evidence_a3_method
```

기본 backbone은 `a3prvr`, device는 `cuda:1`, LLM은 Qwen2.5-7B입니다.
`--llm-device cuda:2`로 LLM을 분리할 수 있습니다.
bank와 run에서 같은 checkpoint, feature extractor, LLM, context batch를 사용해야 합니다.
공식 upstream joint feature를 사용하려면 i3d/clip 옵션에 같은 HDF5 경로를 지정하세요.
original token feature 모델과 새 feature extractor의 일치도 사용자 실험 설정에서 확인하세요.
기존 HDF5에 충분한 provenance가 없다면 동일 모델이라고 자동으로 판정할 수 없습니다.

Holmes 비교는 두 명령 모두 `--backbone holmes-multisrc`로 실행하여 별도 bank를 만듭니다.
`holmes-src`는 `--checkpoint`를 명시해야 합니다. 서로 다른 Holmes model 패키지는
기존 import 구조상 하나의 Python process에 혼합하지 말고 CLI를 각각 실행하세요.
두 검색기의 앙상블은 이 버전에 포함하지 않았으며 각 백본에서 method 효과를 비교합니다.

`--mode selection`으로 선택식을 비교할 수 있습니다.
`--limit`은 진단용 query 수만 제한하며 video 후보 집합은 유지합니다.
전체 candidate pool이 다른 캡션 파일이면 `--video-captions`로 명시하세요.
출력 폴더는 매번 새 경로여야 하며 기존 실험을 덮어쓰거나 이어 쓰지 않습니다.

## 실패·baseline 보존·비용

원문 score vector를 항상 보존합니다. method OFF, gate OFF, 잘못된 분해,
해당 role의 train calibration 부족, 모든 후보 거절, 개선 utility 없음에서는 원문으로 복귀합니다.
해당 query는 평가 분모에 남으며 사유를 audit에 기록합니다.
모델 로딩·CUDA·누락된 피처·NaN·잘못된 tensor shape 등의 실행 오류는 전파합니다.
묵시적 샘플 삭제나 무조건적 예외 무시는 사용하지 않습니다.

LLM decomposition은 원문의 모든 단어를 비중첩 atomic span으로 포함해야 합니다.
관계 graph를 함께 전달하고 별도 호출의 judge가 evidence 보존·관계·신규 사실을 검사합니다.
같은 LLM의 다른 prompt 호출이므로 의미 보존을 완전히 보장하지 않습니다.
judge 결과를 confidence probability나 사람의 annotation으로 해석하지 마세요.

gate는 재작성 **생성 전** 작동합니다. 분해와 evidence 검색은 gate OFF에도 실행되므로
전체 비용을 계산할 때 제외하면 안 됩니다. prompt generation은 deterministic이며
bank audit, sampled IDs, seeds와 feature/checkpoint 식별 정보를 보관합니다.
reference bank의 role별 이웃 수가 부족하면 값을 만들어 넣지 않습니다.
bank가 작으면 spatial/temporal 등의 calibration이 없어 원문 복귀가 많을 수 있습니다.

## 결과와 검증

- bank: `manifest.json`, `bank.npz`, `audit.jsonl`, `complete.json`.
- run: `manifest.json`, `scores.npz`, `audit.jsonl`, `queries.csv`, `summary.json`, `complete.json`.
- R@1/5/10/100, SumR, mAP, 개선/악화/동일 건수, 원문 Top-1 상실, gate 건수, LLM 호출/token.
- 현재 TVR protocol의 정답 비디오는 query당 하나이므로 mAP는 MRR과 같습니다.
- tie는 기존 Holmes 공유 평가기와 같이 고정 video 순서의 stable ranking을 따릅니다.
- `complete.json`은 실행 완료를 뜻하며 연구 가설이나 성능 향상 검증을 뜻하지 않습니다.

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python -m EvidenceRetrieval.test_method
/home/viglab2/miniconda3/envs/holmes/bin/python -m EvidenceRetrieval.test_method --gpu cuda:1
```

소규모 진단 입력/출력은 `../experiments/evidence_retrieval_smoke_20260929/`에 보관합니다.
여기의 forced gate, 작은 reference/후보 집합과 3B LLM은 실행 경로 검사용이며 benchmark가 아닙니다.
정식 평가는 전체 후보를 유지하고 dev에서 설정을 고정한 뒤 해야 합니다.

현재 로컬 val/test 캡션은 byte 단위로 동일합니다. 이를 독립 test 결과로 보고하지 마세요.
정답 비디오의 sibling caption은 method 입력에 제공하지 않습니다.
여러 seed·ablation·독립 test의 성능 향상과 통계적 유의성은 별도 실험이 필요합니다.
