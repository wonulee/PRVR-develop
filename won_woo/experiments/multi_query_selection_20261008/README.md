# 원문별 Object/Action multi-query 선택 결과

원문 87,174개를 보존하고 추가 1,371개를 선택했습니다. 최종 TXT는 88,545개입니다. 원문당 추가 후보는 최대 2개입니다.

| 원문당 추가 쿼리 수 | 원문 개수 |
| --- | ---: |
| 0개 | 85,805 |
| 1개 | 1,365 |
| 2개 | 4 |

선택된 추가 쿼리는 Object 1,364개, Action 7개입니다. 원문과 추가 쿼리의 연결은 1,373행이며, 같은 추가 쿼리가 여러 원문에 연결되는 경우를 보존했습니다. 최종 TXT에서는 같은 영상·문장의 추가 쿼리를 중복 저장하지 않습니다.

- `selection_audit.csv`: 모든 원문의 후보 수, 선택, margin, uncertainty와 사유.
- `candidate_validation.csv`: 모든 생성 연결의 원문·Val 대응 및 텍스트 검사 상태.
- `selected_query_links.csv`: 선택된 원문→추가 쿼리→Val 표현 연결.
- `term_selection_summary.csv`: Train→Val 표현별 입력·통과·보류·선택 개수.
- `subset_evaluations.csv`: 원문·모든 단독 후보·모든 후보 쌍 비교.
- `tvrtrain.caption.selected.txt`: 학습용 원문+선택 추가 쿼리.
- `roberta_tvr_query_feat.selected.hdf5`: 별도 추출 특징과 기존 Val 특징의 external links.
- `dataset/`: 기존 loader를 사용할 수 있는 분리 데이터 경로.

선택은 동결 teacher와 전체 Train gallery의 평균 쿼리 점수로 평가했습니다. 원문 대비 개선, 후보 삭제, 동일 후보 반복 대조를 확인했습니다. 수치 허용오차 epsilon=1e-06과 lambda=0.0이며 성능에 맞춰 튜닝하지 않았습니다.

전체 Train 영상 17,435개를 비교 대상으로 사용했습니다. 텍스트 검사를 통과한 후보가 있는 원문 3,224개에 대해 모든 단독 후보 3,297개와 모든 후보 쌍 77개를 평가했습니다. 통과한 후보가 없는 원문 83,950개는 원문만 보존하고 검색 점수는 빈칸으로 기록했습니다. Action 후보의 선택이 적은 것은 텍스트 검사에서 보류·제외된 후보가 많기 때문이며, Action 변경 자체가 불필요하다는 결론은 아닙니다.

실제 쌍 선택 사례는 `house_s01e11_seg02_clip_13#enc#2`입니다. 원문은 `Cameron walks into the elevator without the other doctors.`입니다. `elevator→lift` 후보 단독의 margin 변화는 −0.000030, `doctors→physicians` 후보 단독은 −0.001589이지만, 원문과 두 후보를 함께 사용하면 +0.001783으로 개선됩니다. 두 후보는 각각 별도 쿼리이며, 두 용어를 한 문장에서 동시에 바꾼 결과가 아닙니다.

LLM을 사용하지 않았고 기존 용어 대응의 1순위 후보만 사용했습니다. 이전 설정에 따라 14–20단어·30 미만 토큰의 길이 필터는 적용하지 않았습니다. 길이와 truncation 여부는 CSV에 기록했습니다. 특징 인코더의 입력 상한은 special token을 포함해 30토큰이므로 긴 캡션의 특징에는 잘린 입력이 반영될 수 있습니다.

텍스트 검사 통과는 임시 연구 후보 상태입니다. 영상·구간의 의미 보존을 확정한 결과가 아닙니다. 정답을 이용한 Train 내부 선택이며, 독립 평가나 추가학습 성능을 증명하지 않습니다. Val 어휘를 사용했고 현재 Val/Test 캡션이 동일합니다.

최종 TXT와 선택된 특징의 ID·연결·shape, 원문 보존, 입력 파일 해시 검증은 `verification.json`의 `PASS_STRUCTURAL_AND_SCORING`으로 기록했습니다. 기존 학습 loader에는 이 폴더 아래 `dataset/`을 데이터 루트로 사용할 수 있습니다. 학습은 아직 실행하지 않았습니다.

HDF5 external links와 dataset symlink가 있어 전체 출력 폴더를 유지해야 합니다. 현재 원본 파일은 변경하지 않았습니다.
