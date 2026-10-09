"""Exact audit condition counts and per-query checks, without inferred type labels."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT=Path('/home/viglab2/won_woo')
sys.path.insert(0,str(ROOT/'Holmes'))
from Rewriter.six_strategy import candidate_check, verdict_check, normalized, TEMPORAL
from transformers import RobertaTokenizer
from Rewriter.linguistic_rules import PATTERNS

OUT=Path(__file__).resolve().parent
FLAGS=('pronoun','negation','mental_state')
LABELS={'pronoun':'대명사','negation':'부정어','mental_state':'감정·심리'}
COMBINATIONS=[(),('pronoun',),('negation',),('mental_state',),('pronoun','negation'),('pronoun','mental_state'),('negation','mental_state'),FLAGS]


def label(combo):
    return ' + '.join(LABELS[k] for k in combo) if combo else '세 flag 모두 없음'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    tokenizer=RobertaTokenizer.from_pretrained(str(ROOT/'tvr/query_only/roberta-base_tuned_model'),local_files_only=True)
    summaries={}
    for split,b in [('train',64),('val',128)]:
        run=ROOT/f'experiments/rewrite_{split}_b{b}_20260930'
        cfg=json.loads((run/'run.json').read_text())
        with (run/'audit.jsonl').open() as f:records=[json.loads(line) for line in f]
        assert len(records)==cfg['total']
        assert len({r['id'] for r in records})==len(records)
        by_video=defaultdict(dict)
        for r in records:by_video[r['context_video_id']][r['id']]=r['original']
        total_counts=Counter();accepted_counts=Counter();gates=Counter();per_query=[];seen=defaultdict(set)
        previously_reviewed=set()
        known_file=run/'adoption_analysis_20261002/reviewed_error_ids.txt'
        if known_file.exists():previously_reviewed=set(known_file.read_text().splitlines())
        for line_number,r in enumerate(records,1):
            combo=tuple(f for f in FLAGS if r['linguistic_flags'][f])
            total_counts[combo]+=1
            if not r['accepted']:continue
            accepted_counts[combo]+=1
            candidate=candidate_check(r['original'],r['payload'],tokenizer,by_video[r['context_video_id']],cfg['max_tokens'],cfg['word_min'],cfg['word_max'],cfg['strict_word_budget'])
            assert candidate==r['candidate']
            verdict_check(r['verdict'])
            key=normalized(candidate)
            assert key not in seen[r['context_video_id']]
            seen[r['context_video_id']].add(key)
            tokens=len(tokenizer.encode(candidate,add_special_tokens=True))
            words=len(candidate.split())
            checks={
                'changed_and_text_schema_pass':True,
                'max_22_words_pass':words<=cfg['word_max'],
                'max_29_roberta_tokens_pass':tokens<=cfg['max_tokens'],
                'temporal_marker_check_pass':TEMPORAL.findall(r['original'].lower())==TEMPORAL.findall(candidate.lower()),
                'literal_grounding_check_pass':True,
                'target_event_preserved_verdict':r['verdict']['target_event_preserved'],
                'participant_roles_preserved_verdict':r['verdict']['participant_roles_preserved'],
                'negation_equivalent_verdict':r['verdict']['negation_equivalent'],
                'temporal_relationships_preserved_verdict':r['verdict']['temporal_relationships_preserved'],
                'no_sibling_event_import_verdict':not r['verdict']['sibling_event_imported'],
                'no_unsupported_visual_details_verdict':not r['verdict']['unsupported_visual_details'],
                'duplicate_guard_pass':True,
            }
            assert all(checks.values())
            gates.update(k for k,v in checks.items() if v)
            grounding=r['payload']['grounding']
            uses_sibling=any(g['caption_id']!=r['id'] for g in grounding)
            per_query.append({
                'id':r['id'],'audit_line':line_number,'original':r['original'].rstrip(),'rewrite':candidate,
                'source_flag_combination':label(combo),'source_pronoun_flag':r['linguistic_flags']['pronoun'],
                'source_negation_flag':r['linguistic_flags']['negation'],'source_mental_state_flag':r['linguistic_flags']['mental_state'],
                'object_flag':'not_recorded','actual_transformation_type':'not_recorded',
                'causal_acceptance_route':'not_recorded','accepted':True,
                **checks,'rewrite_words':words,'rewrite_roberta_tokens_with_specials':tokens,
                'strict_min_words_enforced':cfg['strict_word_budget'],'grounding_items':len(grounding),
                'sibling_grounding_present':uses_sibling,'verifier_reason':r['verdict']['reason'],
                'previously_reviewed_semantic_issue':r['id'] in previously_reviewed,
            })
        accepted=len(per_query)
        assert accepted==cfg['accepted']
        combination_rows=[{'source_flag_combination':label(c),'total_queries_with_combination':total_counts[c],
            'accepted_queries':accepted_counts[c],'percent_of_accepted':100*accepted_counts[c]/accepted,
            'acceptance_rate_within_combination':100*accepted_counts[c]/total_counts[c] if total_counts[c] else None} for c in COMBINATIONS]
        assert sum(r['accepted_queries'] for r in combination_rows)==accepted
        assert sum(r['total_queries_with_combination'] for r in combination_rows)==len(records)
        overlapping=[]
        for flag in FLAGS:
            all_n=sum(n for c,n in total_counts.items() if flag in c)
            accepted_n=sum(n for c,n in accepted_counts.items() if flag in c)
            overlapping.append({'source_flag':flag,'accepted':accepted_n,'percent_of_accepted':100*accepted_n/accepted,'total_with_flag':all_n,'acceptance_rate_with_flag':100*accepted_n/all_n})
        summary={'total':len(records),'accepted':accepted,'overall_acceptance_percent':100*accepted/len(records),
            'source_audit':str(run/'audit.jsonl'),'source_audit_sha256':sha(run/'audit.jsonl'),
            'source_run':str(run/'run.json'),'recorded_flag_patterns':PATTERNS,'exclusive_source_flag_combinations':combination_rows,
            'overlapping_source_flag_rates':overlapping,'validation_gates':{k:{'count':n,'percent_of_accepted':100*n/accepted} for k,n in gates.items()},
            'word_budget':{'soft_min':cfg['word_min'],'max':cfg['word_max'],'strict_min':cfg['strict_word_budget'],'below_soft_min':sum(r['rewrite_words']<cfg['word_min'] for r in per_query)},
            'grounding':{'empty':sum(r['grounding_items']==0 for r in per_query),'sibling_used':sum(r['sibling_grounding_present'] for r in per_query)},
            'limitations':['source flags describe ORIGINAL wording, not the transformation performed or causal acceptance reason','object condition and applied transformation labels are absent in audit','LLM verification booleans are logged judgments, not proof of semantic correctness','literal grounding permits empty evidence and does not establish new candidate-detail correctness','word_min was soft in these historical runs; current 14-20-word requirement is not retroactively applied']}
        write_csv(OUT/f'{split}_accepted_conditions.csv',per_query)
        write_csv(OUT/f'{split}_condition_combinations.csv',combination_rows)
        summaries[split]=summary
        print(split,'total',len(records),'accepted',accepted,'all stored gates pass',flush=True)
    (OUT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    lines=['# Train / Val 채택 조건 전수 분석','',
        '이 보고서는 audit.jsonl에 실제 기록된 원문 flag와 채택 검사를 집계합니다. 원문 flag는 실제 수행된 변환 유형이나 채택 원인을 의미하지 않습니다. 각 채택 쿼리의 ID·원문·rewrite·flag 조합·통과 검사·LLM 설명은 split별 CSV에서 확인할 수 있습니다.','',
        '| 구분 | 전체 쿼리 | 채택 | 전체 대비 채택률 |','|---|---:|---:|---:|']
    for split,s in summaries.items():lines.append(f"| {split} | {s['total']:,} | {s['accepted']:,} | {s['overall_acceptance_percent']:.2f}% |")
    lines+=['','## 채택된 쿼리의 원문 flag 조합','',
        '**분모: train 11,064개, val 1,290개. 아래 조합은 서로 배타적이며 split별 합계 100%입니다.**','',
        '| 원문 flag 조합 | Train 개수 | Train 비율 | Val 개수 | Val 비율 |','|---|---:|---:|---:|---:|']
    for i,c in enumerate(COMBINATIONS):
        tr=summaries['train']['exclusive_source_flag_combinations'][i];va=summaries['val']['exclusive_source_flag_combinations'][i]
        lines.append(f"| {label(c)} | {tr['accepted_queries']:,} | {tr['percent_of_accepted']:.2f}% | {va['accepted_queries']:,} | {va['percent_of_accepted']:.2f}% |")
    lines+=['','세 flag가 없다는 것은 원문에 코드의 대명사·부정어·감정 keyword가 검출되지 않았다는 뜻입니다. Object flag가 없으므로 4가지 전략이 모두 미적용됐다는 의미로 해석할 수 없습니다.','',
        '## 개별 flag 포함 비율: 중복 허용','',
        '| 원문 flag | Train | Val |','|---|---:|---:|']
    for i,f in enumerate(FLAGS):
        tr=summaries['train']['overlapping_source_flag_rates'][i];va=summaries['val']['overlapping_source_flag_rates'][i]
        lines.append(f"| {LABELS[f]} | {tr['accepted']:,} ({tr['percent_of_accepted']:.2f}%) | {va['accepted']:,} ({va['percent_of_accepted']:.2f}%) |")
    lines+=['','flag는 코드의 keyword 정규식입니다. 감정·심리 목록은 upset/angry/happy/sad/afraid/worried/think/believe/want/feel 및 일부 활용형으로 제한되어 confused/shocked 등을 놓칩니다. 부정어 정규식도 cannot 같은 표현을 놓칩니다. 따라서 이는 모든 자연언어 감정·부정 표현의 의미 기반 분포가 아닙니다.','',
        '## 실제 공통 채택 검사','',
        '채택된 모든 쿼리는 비어 있지 않은 변경 문장, 최대 22단어, 특수 토큰 포함 최대 29 RoBERTa 토큰, temporal marker 검사, grounding schema/문자 포함 검사, 영상 내 채택 후보 중복 방지 검사를 통과했습니다. LLM verifier는 event/participant/negation/temporal 보존 true, sibling_event_imported false, unsupported_visual_details []로 판정했습니다. 모든 검사 통과 비율은 train과 val 모두 100%입니다.','',
        '최소 16단어는 강제하지 않았습니다. 16단어 미만 채택은 train 10,241개, val 1,203개입니다. 빈 grounding은 train 12개, val 0개이며 기존 코드가 허용한 결과입니다.','',
        '**통과 비율 100%는 코드상 판정 통과를 뜻하며, 의미 보존의 정확도 100%를 뜻하지 않습니다.** 이전에 검토한 val 의미 변경 사례는 CSV의 previously_reviewed_semantic_issue에 표시했고, 전체 오류율로 확대 해석하지 않습니다.','',
        '## 실제 4가지 변환별 비율의 해석 한계','',
        '현재 audit에는 applied_strategy / transformation_type / causal_acceptance_route가 없습니다. 따라서 “대명사 변환으로 채택된 비율”이나 “Object 강조로 채택된 비율”을 로그만으로 확정할 수 없습니다. 모든 쿼리에 rewriting을 시도하고 공통 검사를 통과하면 채택하는 구조입니다.','',
        '예: he를 생략한 문법 축약은 pronoun flag 감소여도 이름으로 대체한 해소가 아닙니다. upset → sad도 감정 flag가 있어도 구체 표정/제스처 변환이 아닙니다. 4가지 실제 유형을 보고하려면 원문/rewrite 변경 구간과 근거를 별도로 분류·검토해야 하며 그 추정 결과를 이 확정 집계와 구분해야 합니다.','',
        '## 재현','',
        '```bash','/home/viglab2/miniconda3/envs/holmes/bin/python /home/viglab2/won_woo/experiments/acceptance_condition_analysis_20261003/analyze.py','```','',
        '원본 caption, audit, rewriting 결과, HDF5, checkpoint를 변경하지 않습니다. CSV는 Excel에서 열 수 있도록 UTF-8-SIG로 저장했습니다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
