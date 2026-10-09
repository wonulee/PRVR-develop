> 이 기록은 최초 적용 이력입니다. 사전 범위를 위반한 8개 치환은 applied_dictionary_scope_repair_20261007에서 제거했으며, 현재 결과는 1,031개 캡션·1,132개 쌍입니다.

# TVR Train 유의어 치환 — 2026-10-07

`diagram_synonym_pairs/object_B_train.csv`와 `action_B_train.csv`의 실제 Train→Val 후보에서 규칙 기반으로 선별했습니다. 최종 선별과 적용에는 LLM을 사용하지 않았습니다.

- 전체 캡션: 87,174개
- 변경 캡션: 1,037개 / 유지 캡션: 86,137개
- 적용 Object 쌍: 633개 / Action 쌍: 507개 / 실제 치환: 1,158회
- 변경 문장 모두 14–20단어, 시작·종료 토큰 포함 30 미만 RoBERTa 토큰
- 캡션 ID·video ID·enc·순서·구분 공백·줄바꿈·파일 권한 유지, Val SHA-256 동일

통상적인 유의어 계열에 포함되고, 원문 및 실제 Val 근거에서 관측된 표현의 품사가 일치하는 후보만 사용했습니다. 모호한 대명사, 부정, 내적 상태 표현이 있는 원문과 대화 내용, 합성어·구동사·논항·관사 오류 가능성은 보존했습니다. `synonym_rank`는 사전순이므로 적합도 점수로 사용하지 않았습니다. 각 표현의 규칙 통과 후보 중 원래 순위가 가장 낮은 하나를 선택하고, 선택된 표현을 동시에 치환했습니다. 원래 순위는 `original_synonym_rank`에 남아 있습니다.

예외 규칙은 `lift`의 차량 동승 의미, `drink`의 마시는 행위 의미, `bike`의 모터사이클/운동기구 가능성 등을 제외합니다. 허용 계열 밖의 다른 적합한 유의어도 있을 수 있습니다. 규칙 검사는 문맥 적합성의 완전한 보장이 아니며, 영상을 직접 확인하지 않았습니다. 입력 CSV의 `requires_review`는 유지했습니다.

| 파일 | 내용 |
| --- | --- |
| `tvrtrain.caption.original.txt` | 적용 직전 전체 원본 백업 |
| `tvrtrain.caption.rewritten.txt` | 실제 적용 결과와 같은 전체 사본 |
| `caption_changes.csv` | 변경 전후 문장, 치환 목록, 단어/토큰 수 |
| `approved_pairs.csv` | 최종 선별 후보와 원본 CSV/행 번호/Val 근거 |
| `rule_pair_audit.csv` | 검토한 후보별 채택/제외 사유 |
| `rule_selection_summary.json` | 선별 규칙, 수치, 입력 해시, seed 2026 |
| `summary.json` | 적용 수치, 해시, 백업 경로 |
| `verification.json` | 적용 결과의 독립 검증 |
| `source_hashes.json` | 원본 diagram 파일의 SHA-256 |

최종 도구는 `rewriting/select_diagram_synonyms.py`와 `rewriting/apply_val_synonyms.py`입니다. 원본을 다시 선별할 때는 select 도구에 `--captions tvrtrain.caption.original.txt`를 지정하세요. 적용에는 `--enforce-length-budget`을 사용합니다. `rewriting/test_apply_val_synonyms.py`는 문맥 제외, 길이 제한, ID 및 백업 보존을 검사합니다.

초기 LLM 점검은 대조 사례를 통과하지 못했고, 사용자 지시로 중단했습니다. 실패 기록과 중단한 코드는 재현 이력을 위해 별도로 보존했으며 최종 선별에는 사용하지 않았습니다.

Holmes의 Dirichlet 손실·불확실성 계산·데이터 split은 변경하지 않았습니다. 객체와 동작의 의미를 보존하는 어휘 치환으로 Train/Val 표현 정합성을 높이는 작업입니다. 기존 RoBERTa HDF5와 Object/Action 추출 사전은 갱신하지 않았습니다. 변경 문장을 모델의 유사도·evidence aggregation에 반영하려면 텍스트 특징을 다시 추출해야 합니다.
