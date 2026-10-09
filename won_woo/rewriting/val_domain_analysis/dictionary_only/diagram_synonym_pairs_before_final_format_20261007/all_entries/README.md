# Object/Action 전체 CSV

모든 원본 항목과 WordNet 조회 정보를 함께 담았습니다. 유의어가 없는 행도 보존합니다.

- all_objects.csv / all_actions.csv: 모든 항목·품사·caption ID·원문. EMPTY와 UNCLASSIFIED도 포함
- all_captions.csv: 전체 캡션별 object/action 원본 사전 목록
- file_counts.csv: 종류·split별 전체 항목 및 행 수

WordNet 등록 여부·공식 사전 내 유의어 후보 수/목록·반대 split 후보 수/목록·synset·선택 synset·의미·검토 여부를 포함했습니다. 후보 표현은 같은 aspect 공식 사전 항목으로 제한합니다. 기본형이 없거나 지원하지 않는 품사는 조회 여부가 빈칸이고 후보 수는 0입니다.

Train은 공식 ID 사전의 모든 항목을 그대로 보존했습니다. 사전 기준 원본 캡션과 현재 캡션을 모두 넣었습니다. Val은 공식 사전 표면형·POS 허용 범위의 전체 추출입니다. 기본형은 정규화 표시이며 사전항목은 원본 표면형입니다. CSV를 다시 읽어 모든 ID의 원본 사전 매핑과 완전히 일치하는지 검증했습니다. UTF-8 BOM입니다.
