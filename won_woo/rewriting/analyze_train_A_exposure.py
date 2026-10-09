"""Audit Train A exposure and literal absence from Val without changing training."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from analyze_val_domain import read_captions, write_csv
from build_object_action_regions import sha256


def main():
    workspace = Path(__file__).resolve().parent.parent
    base = workspace / 'rewriting/val_domain_analysis'
    out = base / 'train_A_exposure'
    out.mkdir(parents=True, exist_ok=True)
    membership_path = base / 'diagram_synonym_pairs/query_membership.csv'
    reference_path = base / 'diagram_region_audit/context_review/final_term_regions.csv'
    audit_path = base / 'diagram_region_audit/context_review/final_audit/summary.json'
    audit = json.loads(audit_path.read_text())
    assert sha256(membership_path) == audit['checked_csv_sha256'][str(membership_path)]
    train_path = workspace / 'tvr/TextData/tvrtrain.caption.txt'
    val_path = workspace / 'tvr/TextData/tvrval.caption.txt'
    train = {r['caption_id']: r for r in read_captions(train_path)}
    val = {r['caption_id']: r for r in read_captions(val_path)}
    queries = {cid: {'A':set(),'B':set()} for cid in train}
    term_ids = defaultdict(set)
    with membership_path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'train' or r['region'] not in {'A','B'}:
                continue
            assert train[r['caption_id']]['caption'] == r['caption']
            for lemma in json.loads(r['source_lemmas']):
                key = r['aspect'], lemma
                queries[r['caption_id']][r['region']].add(key)
                term_ids[key].add(r['caption_id'])
    with reference_path.open(encoding='utf-8-sig',newline='') as f:
        a_reference = {(r['aspect'],r['lemma']):r for r in csv.DictReader(f) if r['split']=='train' and r['term_region']=='A'}
    expressions = defaultdict(set)
    source_paths = []
    for aspect in ['object','action']:
        p = base / f'diagram_synonym_pairs/{aspect}_A_train.csv'
        source_paths.append(p)
        assert sha256(p) == audit['checked_csv_sha256'][str(p)]
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                assert r['synonym_rank']=='0'
                expressions[(aspect,r['source_lemma'])].add(r['source_expression'])
    assert set(expressions)==set(a_reference)
    # Literal presence is a separate diagnostic: it does not certify matching POS/sense.
    val_lower = {cid:r['caption'].casefold() for cid,r in val.items()}
    token_index = defaultdict(set)
    for cid,text in val_lower.items():
        for token in re.findall(r'\w+', text):
            token_index[token].add(cid)
    word_report = []
    literal_seen = set()
    for key,info in a_reference.items():
        aspect,lemma = key
        variants = expressions[key] | {lemma}
        seen = set()
        for text in variants:
            token_texts = re.findall(r'\w+', text.casefold())
            if not token_texts:
                continue
            candidates = token_index[token_texts[0]]
            pattern = re.compile(r"(?<!\w)" + re.escape(text.casefold()) + r"(?!\w)")
            seen.update(cid for cid in candidates if pattern.search(val_lower[cid]))
        if seen:
            literal_seen.add(key)
        ids = term_ids[key]
        word_report.append({'aspect':aspect,'source_lemma':lemma,'train_queries':len(ids),
                            'source_expressions':' | '.join(sorted(expressions[key])),
                            'val_query_text_match_count':len(seen),'literal_form_absent_from_val':not bool(seen),
                            'has_wordnet_entry':bool(info['wordnet_synsets']),
                            'wordnet_candidates':info['wordnet_candidates'],'selected_wordnet_synsets':info['selected_wordnet_synsets'],
                            'train_example_caption_ids':' | '.join(sorted(ids)[:3]),
                            'train_example_captions':json.dumps([train[cid]['caption'] for cid in sorted(ids)[:3]],ensure_ascii=False),
                            'val_example_caption_ids':' | '.join(sorted(seen)[:3]),
                            'val_example_captions':json.dumps([val[cid]['caption'] for cid in sorted(seen)[:3]],ensure_ascii=False),
                            'interpretation':'Same lemma/surface string in Val text; verify POS and meaning before calling it Train-exclusive' if seen else
                                             'No observed lemma/source-form string in Val text; not proof of harmful learning'})
    word_report.sort(key=lambda r:(not r['literal_form_absent_from_val'],-r['train_queries'],r['aspect'],r['source_lemma']))
    write_csv(out/'A_word_exposure.csv',list(word_report[0]),word_report)
    query_report = []
    for cid,state in queries.items():
        if not state['A']:
            continue
        a,b = state['A'],state['B']
        query_report.append({'caption_id':cid,'video_id':train[cid]['video_id'],'caption':train[cid]['caption'],
                             'object_A_lemmas':' | '.join(sorted(t for kind,t in a if kind=='object')),
                             'action_A_lemmas':' | '.join(sorted(t for kind,t in a if kind=='action')),
                             'A_valid_lemma_count':len(a),'B_valid_lemma_count':len(b),
                             'A_fraction_of_extracted_valid_lemmas':len(a)/(len(a)+len(b)),
                             'all_valid_terms_A':not bool(b),
                             'A_lemmas_with_literal_val_match':' | '.join(f'{kind}:{term}' for kind,term in sorted(a & literal_seen)),
                             'A_lemmas_without_literal_val_match':' | '.join(f'{kind}:{term}' for kind,term in sorted(a - literal_seen))})
    query_report.sort(key=lambda r:(-r['A_fraction_of_extracted_valid_lemmas'],-r['A_valid_lemma_count'],r['caption_id']))
    write_csv(out/'A_train_queries.csv',list(query_report[0]),query_report)
    clean_words = [r for r in word_report if r['literal_form_absent_from_val']]
    write_csv(out/'A_words_without_literal_val_match.csv',list(word_report[0]),clean_words)
    a_exposure=sum(len(r['A']) for r in queries.values())
    b_exposure=sum(len(r['B']) for r in queries.values())
    singleton=sum(len(term_ids[key])==1 for key in a_reference)
    report = {'train_queries':len(train),'A_containing_queries':len(query_report),
              'A_containing_query_percent':100*len(query_report)/len(train),
              'A_only_valid_term_queries':sum(r['all_valid_terms_A'] for r in query_report),
              'A_and_B_queries':sum(not r['all_valid_terms_A'] for r in query_report),
              'A_unique_lemmas':len(a_reference),'A_query_lemma_exposures':a_exposure,'B_query_lemma_exposures':b_exposure,
              'A_share_of_extracted_query_lemma_exposures_percent':100*a_exposure/(a_exposure+b_exposure),
              'A_singleton_lemmas':singleton,'A_singleton_percent':100*singleton/len(a_reference),
              'A_lemmas_with_literal_Val_text_match':len(literal_seen),'A_lemmas_without_literal_Val_text_match':len(clean_words),
              'A_lemmas_without_WordNet_entry':sum(not r['has_wordnet_entry'] for r in word_report),
              'training_effect_status':'Not measured. Neither literal absence nor frequency proves a term decreases validation recall.',
              'exposure_unit':'Distinct (caption_id, aspect, lemma) entries, not encoder tokens, attention weights, gradient mass, or training repetitions',
              'presence_unit':'Case-insensitive standalone source-expression or lemma string in a Val caption; does not establish matching POS/meaning',
              'source_sha256':{str(p):sha256(p) for p in [membership_path,reference_path,train_path,val_path]+source_paths},
              'script_sha256':sha256(Path(__file__))}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text(f'''# A 영역의 학습 노출과 Val 원문 출현 진단

검증할 가설은 “Train 전용 A 표현에 대한 학습이 Val 검색에 악영향을 주는가”입니다. 앞선 Val B/C 성능 비교는 이 질문을 직접 검증하지 않습니다.
Val에 없다는 이유만으로 유해한 학습 표현이라고 확정하지 않습니다. 이 보고서는 검토 후보와 분류 문제를 찾는 데이터 진단입니다.

- A 포함 Train 쿼리: {len(query_report):,}/{len(train):,} ({report['A_containing_query_percent']:.2f}%).
- A만 있는 쿼리: {report['A_only_valid_term_queries']:,}. A와 B가 함께 있는 쿼리: {report['A_and_B_queries']:,}.
- 고유 A 기본형: {len(a_reference):,}. 한 Train 쿼리에만 나타나는 A 기본형: {singleton:,} ({report['A_singleton_percent']:.2f}%).
- 추출 쿼리×유형(Object/Action)×기본형 출현 중 A 비중: {report['A_share_of_extracted_query_lemma_exposures_percent']:.2f}%. 실제 학습 손실·토큰 비중은 아닙니다.
- A 중 Val 원문에 같은 기본형/관측 표현 문자열이 있는 항목: {len(literal_seen):,}. 없는 항목: {len(clean_words):,}.

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

- [A 단어별 학습 빈도·Val 원문 출현·WordNet 후보]({out/'A_word_exposure.csv'})
- [A 포함 Train 쿼리 8,024개와 A/B 구성]({out/'A_train_queries.csv'})
- [Val 원문에서 같은 관측 표현/기본형 문자열을 찾지 못한 A 단어]({out/'A_words_without_literal_val_match.csv'})
- [수치·정의·입력 해시]({out/'summary.json'})

검토 후보이며 “이 단어를 학습하면 성능이 낮아진다”는 판정 목록이 아닙니다. 실제 Train 손실·gradient나 A 개입 후 모델의 Val 결과는 현재 CSV에 없습니다.
최종 A 분류의 657개 문맥 검토 밖 항목은 자동 POS/WordNet 조회 기준입니다. 실제 문자열 부재 검사는 어휘 진단이며 영상 의미 검증은 아닙니다.
''',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2),flush=True)
    print('A strings found in Val:',[(r['aspect'],r['source_lemma'],r['train_queries'],r['val_query_text_match_count']) for r in sorted(word_report,key=lambda r:-r['train_queries']) if not r['literal_form_absent_from_val']][:12],flush=True)


if __name__ == '__main__':
    main()
