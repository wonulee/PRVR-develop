# A/B/C별 전체 항목 CSV

Object와 Action 각각 A_Train, B_Train, B_Val, C_Val로 나눴습니다. 모든 원본 항목을 포함하며 유의어 후보 유무로 필터링하지 않았습니다.

WordNet 등록 여부·synset·의미·공식 사전 내 후보·반대 split 후보·검토 여부 열을 포함했습니다. 후보는 공식 Object/Action 사전에 있는 표현만 담습니다. 같은 품사 WordNet 의미 전체를 조회하며 문맥 검증 전 정보입니다.

EMPTY와 UNCLASSIFIED는 별도 파일에 보존했습니다. C_Val은 현재 사전 제한 범위에서 0행이며 헤더를 유지했습니다. Val은 공식 사전 표면형·POS 허용 범위의 전체 추출입니다.

file_counts.csv에 각 파일의 전체 항목 행 수와 고유 기본형 수를 기록했습니다. 모든 분할 파일을 다시 읽어 원본의 모든 열·값·파일 내 순서와 일치하는지 검증했습니다. UTF-8 BOM입니다.
