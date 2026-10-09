"""Compare query-level retrieval with the audited Object/Action coverage regions."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_val_domain import read_captions, write_csv
from build_object_action_regions import sha256

KS = (1, 5, 10, 100)


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def query_region(regions):
    if 'C' in regions:
        return 'C_PRESENT'
    if 'B' in regions:
        return 'B_ONLY'
    return 'EMPTY' if regions == {'EMPTY'} else 'REVIEW_ONLY'


def metrics(records):
    result = {'evaluated_queries': len(records)}
    for k in KS:
        correct = sum(row[f'R@{k}_correct'] == 'True' for row in records)
        result.update({f'R@{k}_successes': correct, f'R@{k}_failures': len(records) - correct,
                       f'R@{k}': 100 * correct / len(records) if records else '',
                       f'failure@{k}': 100 * (len(records) - correct) / len(records) if records else ''})
    result['Rsum'] = sum(result[f'R@{k}'] for k in KS) if records else ''
    result['mean_query_words'] = float(np.mean([row['query_word_count'] for row in records])) if records else ''
    return result


def main():
    workspace = Path(__file__).resolve().parent.parent
    base = workspace / 'rewriting/val_domain_analysis'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=workspace / 'tvr_test_results.csv')
    parser.add_argument('--output-dir', type=Path, default=base / 'retrieval_region_performance')
    parser.add_argument('--bootstrap-iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    assert args.bootstrap_iterations > 0
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    membership_path = base / 'diagram_synonym_pairs/query_membership.csv'
    term_path = base / 'diagram_region_audit/context_review/final_term_regions.csv'
    audit_path = base / 'diagram_region_audit/context_review/final_audit/summary.json'
    audit = json.loads(audit_path.read_text())
    assert not audit['findings']
    assert sha256(membership_path) == audit['checked_csv_sha256'][str(membership_path)]

    memberships = defaultdict(lambda: defaultdict(set))
    vocabulary = defaultdict(set)
    train_a_queries = defaultdict(set)
    for row in load_csv(membership_path):
        key = row['split'], row['caption_id'], row['aspect']
        memberships[key][row['region']].update(json.loads(row['source_lemmas']))
        # Keep a region with an empty lemma list, including EMPTY.
        memberships[key][row['region']]
        for term in json.loads(row['source_lemmas']):
            vocabulary[(row['aspect'], row['split'], term)].add(row['caption_id'])
        if row['split'] == 'train' and row['region'] == 'A':
            train_a_queries[row['aspect']].add(row['caption_id'])
    terms = {(r['aspect'], r['split'], r['lemma']): r for r in load_csv(term_path)}
    captions_path = workspace / 'tvr/TextData/tvrval.caption.txt'
    test_path = workspace / 'tvr/TextData/tvrtest.caption.txt'
    train_path = workspace / 'tvr/TextData/tvrtrain.caption.txt'
    captions = {r['caption_id']: r for r in read_captions(captions_path)}
    train_ids = {r['caption_id'] for r in read_captions(train_path)}
    records = load_csv(args.results)
    assert len({r['caption_id'] for r in records}) == len(records)
    assert {r['caption_id'] for r in records} == set(captions)
    assert not {r['caption_id'] for r in records} & train_ids
    whitespace_differences = 0
    for row in records:
        cid = row['caption_id']
        original = captions[cid]
        assert row['query'].split() == original['caption'].split(), ('Query text changed', cid)
        whitespace_differences += row['query'] != original['caption']
        assert row['gt_video_ids'] == original['video_id'] == row['video_id_from_caption']
        rank = int(row['gt_rank'])
        assert rank >= 1
        for k in KS:
            assert row[f'R@{k}_correct'] in {'True', 'False'}
            assert (row[f'R@{k}_correct'] == 'True') == (rank <= k)
        assert (row['R@1_correct'] == 'True') == (row['predicted_video_id'] == original['video_id'])
        assert row['top5_predictions'].split(', ')[0] == row['predicted_video_id']
        for aspect in ['object', 'action']:
            state = memberships[('validation', cid, aspect)]
            assert state and set(state) <= {'B', 'C', 'EMPTY', 'EXCLUDED', 'UNCLASSIFIED'}
            row[f'{aspect}_region'] = query_region(set(state))
            row[f'{aspect}_term_regions'] = ' | '.join(sorted(state))
            for region in ['B', 'C']:
                row[f'{aspect}_{region}_lemmas'] = ' | '.join(sorted(state.get(region, set())))
            for field in ['wordnet_candidates', 'context_wordnet_candidates', 'selected_wordnet_synsets']:
                row[f'{aspect}_C_{field}'] = json.dumps(
                    {term: terms[(aspect, 'validation', term)][field] for term in sorted(state.get('C', set()))},
                    ensure_ascii=False)
        labels = {row['object_region'], row['action_region']}
        row['combined_region'] = ('C_PRESENT' if 'C_PRESENT' in labels else
                                  'B_ONLY' if 'B_ONLY' in labels else
                                  'EMPTY' if labels == {'EMPTY'} else 'REVIEW_ONLY')
        row['has_unclassified'] = any('UNCLASSIFIED' in memberships[('validation', cid, aspect)] for aspect in ['object', 'action'])
        row['has_excluded_originals'] = any('EXCLUDED' in memberships[('validation', cid, aspect)] for aspect in ['object', 'action'])
        row['query_word_count'] = len(re.findall(r"\b\w+(?:['’]\w+)?\b", row['query']))
        prefix = cid.split('_')[0]
        row['series'] = {'friends': 'Friends', 'house': 'House', 'castle': 'Castle', 'met': 'HIMYM'}.get(prefix, 'BBT' if re.fullmatch(r's\d+e\d+', prefix) else 'OTHER')
    by_id = {r['caption_id']: r for r in records}
    cohorts = {(aspect, region): [r for r in records if r[f'{aspect}_region'] == region]
               for aspect in ['combined', 'object', 'action'] for region in ['B_ONLY', 'C_PRESENT', 'EMPTY', 'REVIEW_ONLY']}
    summaries = []
    for aspect in ['combined', 'object', 'action']:
        assert sum(len(cohorts[(aspect, region)]) for region in ['B_ONLY', 'C_PRESENT', 'EMPTY', 'REVIEW_ONLY']) == len(records)
        a_count = len(set().union(*train_a_queries.values())) if aspect == 'combined' else len(train_a_queries[aspect])
        summaries.append({'aspect': aspect, 'region': 'A_TRAIN_ONLY', 'train_queries_containing_A': a_count, **metrics([])})
        summaries.append({'aspect': aspect, 'region': 'ALL', 'train_queries_containing_A': '', **metrics(records)})
        for region in ['B_ONLY', 'C_PRESENT', 'EMPTY', 'REVIEW_ONLY']:
            summaries.append({'aspect': aspect, 'region': region, 'train_queries_containing_A': '', **metrics(cohorts[(aspect, region)])})
        for k in KS:
            assert sum(s[f'R@{k}_successes'] for s in summaries if s['aspect'] == aspect and s['region'] in {'B_ONLY', 'C_PRESENT', 'EMPTY', 'REVIEW_ONLY'}) == metrics(records)[f'R@{k}_successes']
    write_csv(out / 'region_recall.csv', list(summaries[0]), summaries)
    clean = [r for r in records if not r['has_unclassified']]
    clean_summaries = [{'aspect': aspect, 'region': region, **metrics([r for r in clean if r[f'{aspect}_region'] == region])}
                       for aspect in ['combined', 'object', 'action'] for region in ['B_ONLY', 'C_PRESENT', 'EMPTY', 'REVIEW_ONLY']]
    write_csv(out / 'region_recall_without_unclassified.csv', list(clean_summaries[0]), clean_summaries)
    matrix = defaultdict(list)
    for row in records:
        matrix[(row['object_region'], row['action_region'])].append(row)
    cross = [{'object_region': obj, 'action_region': act, **metrics(rows)} for (obj, act), rows in sorted(matrix.items())]
    write_csv(out / 'object_action_matrix.csv', list(cross[0]), cross)

    # Resample whole videos, retaining dependence between captions of a clip.
    videos = sorted({r['video_id_from_caption'] for r in records})
    video_index = {vid: i for i, vid in enumerate(videos)}
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=args.bootstrap_iterations)
    comparisons = []
    for aspect in ['combined', 'object', 'action']:
        rates = {}
        for region in ['B_ONLY', 'C_PRESENT']:
            counts = np.zeros((len(videos), 5), dtype=np.int64)
            for row in cohorts[(aspect, region)]:
                counts[video_index[row['video_id_from_caption']]] += [1] + [row[f'R@{k}_correct'] == 'True' for k in KS]
            boot = weights @ counts
            with np.errstate(divide='ignore', invalid='ignore'):
                rates[region] = boot[:, 1:] / boot[:, :1]
        b, c = metrics(cohorts[(aspect, 'B_ONLY')]), metrics(cohorts[(aspect, 'C_PRESENT')])
        for i, k in enumerate(KS):
            delta = 100 * (rates['C_PRESENT'][:, i] - rates['B_ONLY'][:, i])
            valid = np.isfinite(delta)
            assert valid.sum() >= 0.95 * args.bootstrap_iterations
            lower, upper = np.quantile(delta[valid], [0.025, 0.975])
            comparisons.append({'aspect': aspect, 'metric': f'R@{k}', 'B_queries': b['evaluated_queries'], 'C_queries': c['evaluated_queries'],
                                'B_recall': b[f'R@{k}'], 'C_recall': c[f'R@{k}'], 'C_minus_B_percentage_points': c[f'R@{k}'] - b[f'R@{k}'],
                                'B_failure_rate': b[f'failure@{k}'], 'C_failure_rate': c[f'failure@{k}'],
                                'C_failure_risk_ratio_vs_B': c[f'failure@{k}'] / b[f'failure@{k}'],
                                'delta_ci95_lower': float(lower), 'delta_ci95_upper': float(upper),
                                'ci_includes_zero': bool(lower <= 0 <= upper), 'bootstrap_valid_samples': int(valid.sum())})
    write_csv(out / 'C_vs_B_comparison.csv', list(comparisons[0]), comparisons)
    print('Computed disjoint query cohorts and video-cluster bootstrap intervals', flush=True)

    references = [{'source': str(args.results), 'variant': 'requested_csv', 'checkpoint': 'Not recorded in CSV', **{k: v for k, v in metrics(records).items() if k.startswith('R@') or k in {'evaluated_queries','Rsum'}}}]
    secondary_path = workspace / 'tvr/holmes_tvr_test_results_1.csv'
    inputs = [args.results, membership_path, term_path, audit_path, captions_path, test_path, train_path]
    if secondary_path.exists():
        secondary = load_csv(secondary_path)
        assert {r['Caption ID'] for r in secondary} == set(by_id) and len(secondary) == len(records)
        for old in secondary:
            row = by_id[old['Caption ID']]
            assert old['Query (자연어 검색 쿼리)'].split() == row['query'].split()
            assert old['정답 비디오 ID (Ground Truth)'] == row['gt_video_ids']
            assert old['R@1 결과'] in {'정답 (Success)', '오답 (Failure)'}
            for k in KS:
                if k > 1:
                    assert old[f'R@{k}'] in {'O', 'X'}
                row[f'reference_R@{k}_correct'] = (old['R@1 결과'] == '정답 (Success)') if k == 1 else old[f'R@{k}'] == 'O'
            assert [row[f'reference_R@{k}_correct'] for k in KS] == sorted(row[f'reference_R@{k}_correct'] for k in KS)
        other_stats = {f'R@{k}': 100 * sum(r[f'reference_R@{k}_correct'] for r in records) / len(records) for k in KS}
        references.append({'source': str(secondary_path), 'variant': 'additional_query_result_csv', 'checkpoint': 'Not recorded in CSV', 'evaluated_queries': len(records), **other_stats, 'Rsum':sum(other_stats.values())})
        paired = []
        for aspect in ['combined', 'object', 'action']:
            for region in ['ALL', 'B_ONLY', 'C_PRESENT']:
                cohort = records if region == 'ALL' else cohorts[(aspect, region)]
                for k in KS:
                    new = sum(r[f'R@{k}_correct'] == 'True' for r in cohort)
                    old = sum(r[f'reference_R@{k}_correct'] for r in cohort)
                    paired.append({'aspect': aspect, 'region': region, 'metric': f'R@{k}', 'queries':len(cohort),
                                   'requested_csv_recall':100*new/len(cohort), 'reference_csv_recall':100*old/len(cohort), 'difference_percentage_points':100*(new-old)/len(cohort),
                                   'fail_to_success':sum(r[f'R@{k}_correct']=='True' and not r[f'reference_R@{k}_correct'] for r in cohort),
                                   'success_to_fail':sum(r[f'R@{k}_correct']=='False' and r[f'reference_R@{k}_correct'] for r in cohort)})
        write_csv(out / 'paired_result_comparison.csv', list(paired[0]), paired)
        inputs.append(secondary_path)
    for path in [workspace / 'experiments/eval_train_effective_best_20261002/metrics.json', workspace / 'tvr/TextData/effective_rewritten1.holmes_eval/metrics.json']:
        if path.exists():
            metadata = json.loads(path.read_text())
            for variant in ['single','original','original_plus_rewrite']:
                if variant in metadata:
                    value = metadata[variant]
                    references.append({'source': str(path), 'variant':variant,'checkpoint':metadata['checkpoint'], 'evaluated_queries':value['queries'], **{f'R@{k}':value[f'R@{k}'] for k in KS}, 'Rsum':value['Rsum']})
            inputs.append(path)
    reference_fields = ['source','variant','checkpoint','evaluated_queries']+[f'R@{k}' for k in KS]+['Rsum']
    write_csv(out / 'recorded_performance_comparison.csv', reference_fields, [{k:r.get(k,'') for k in reference_fields} for r in references])

    term_stats = []
    for (aspect, split, lemma), info in sorted(terms.items()):
        if split == 'train' and info['term_region'] != 'A':
            continue
        ids = vocabulary[(aspect, split, lemma)]
        assert ids
        evaluated = [by_id[cid] for cid in sorted(ids)] if split == 'validation' else []
        term_stats.append({'aspect':aspect,'split':split,'source_lemma':lemma,'term_region':info['term_region'],'source_queries':len(ids),
                           **metrics(evaluated), 'wordnet_candidates':info['wordnet_candidates'], 'context_wordnet_candidates':info['context_wordnet_candidates'],
                           'selected_wordnet_synsets':info['selected_wordnet_synsets'],'other_split_candidates':info['other_split_candidates']})
    write_csv(out / 'term_recall.csv', list(term_stats[0]), term_stats)
    write_csv(out / 'query_region_results.csv', list(records[0]), records)
    for k in [1,100]:
        failed = sorted([r for r in records if r['combined_region']=='C_PRESENT' and r[f'R@{k}_correct']=='False'], key=lambda r:(-int(r['gt_rank']), r['caption_id']))
        write_csv(out / f'C_failure_queries_R{k}.csv', list(records[0]), failed)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    x = np.arange(len(KS))
    for ax, aspect in zip(axes, ['combined','object','action']):
        b, c = metrics(cohorts[(aspect,'B_ONLY')]), metrics(cohorts[(aspect,'C_PRESENT')])
        ax.bar(x-0.18,[b[f'R@{k}'] for k in KS],width=.36,color='#153b56',label=f"B only (n={b['evaluated_queries']:,})")
        ax.bar(x+0.18,[c[f'R@{k}'] for k in KS],width=.36,color='#e84839',label=f"C present (n={c['evaluated_queries']:,})")
        ax.plot(x,[metrics(records)[f'R@{k}'] for k in KS],color='#64748b',linestyle='--',marker='o',label='All queries')
        ax.set_xticks(x,[f'R@{k}' for k in KS])
        ax.set_title(aspect.title(),fontsize=15)
        ax.set_ylim(0,105)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left',fontsize=9)
    axes[0].set_ylabel('Recall (%)')
    fig.suptitle('TVR retrieval performance by Object / Action coverage',fontsize=18,y=.98)
    fig.text(.5,.055,'C present = at least one uncovered term; B only = observed valid terms covered, with no C term.',ha='center',fontsize=10)
    fig.text(.5,.02,'A is Train-only and has no evaluation in this CSV. WordNet coverage does not establish causal failure.',ha='center',fontsize=10,color='#64748b')
    fig.tight_layout(rect=(0,.1,1,.94))
    for extension in ['png','svg','pdf']:
        fig.savefig(out / f'region_recall_comparison.{extension}',dpi=180,facecolor='white')
    plt.close(fig)

    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    fig.suptitle('A / B / C regions and query-level retrieval performance',fontsize=22,y=.97)
    fig.text(.5,.92,'Evaluation cohort: C if any C term; otherwise B when at least one valid B term is present.',ha='center',fontsize=12)
    for ax, aspect in zip(axes, ['object','action']):
        ax.set_xlim(-2,102)
        ax.set_ylim(-9,105)
        ax.axis('off')
        ax.set_title(aspect.title(),fontsize=20,weight='bold',pad=6)
        for center,color,fill in [(35,'#153b56','#eaf1f7'),(65,'#e84839','#fff0ed')]:
            ax.add_patch(Ellipse((center,47),67,75,facecolor=fill,edgecolor=color,linewidth=2,alpha=.64))
        ax.text(24,89,'Train-derived coverage',ha='center',fontsize=11,color='#153b56')
        ax.text(78,89,'Evaluation queries',ha='center',fontsize=11,color='#e84839')
        ax.text(16,63,'A | Train',ha='center',fontsize=16,weight='bold',color='#153b56')
        ax.text(16,49,f"{len(train_a_queries[aspect]):,} queries\ncontain A terms",ha='center',fontsize=12,linespacing=1.6)
        ax.text(16,26,'R@1 / 5 / 10 / 100\nN/A: no Train inference\nin the supplied CSV',ha='center',fontsize=10,linespacing=1.7)
        for x,region,label,color in [(50,'B_ONLY','B | Val, no C','#153b56'),(84,'C_PRESENT','C | Val, has C','#e84839')]:
            values=metrics(cohorts[(aspect,region)])
            ax.text(x,65,label,ha='center',fontsize=14,weight='bold',color=color)
            ax.text(x,56,f"n = {values['evaluated_queries']:,}",ha='center',fontsize=12,color=color)
            detail='\n'.join(f"R@{k}: {values[f'R@{k}']:.2f}%" for k in KS)
            ax.text(x,37,detail,ha='center',va='center',fontsize=12,linespacing=1.7,color=color)
        empty=metrics(cohorts[(aspect,'EMPTY')])
        review_only=metrics(cohorts[(aspect,'REVIEW_ONLY')])
        ax.text(50,3,f"Separate: EMPTY {empty['evaluated_queries']:,} | REVIEW_ONLY {review_only['evaluated_queries']:,}",ha='center',fontsize=11,
                bbox={'boxstyle':'round,pad=.5','fc':'#f1f5f9','ec':'#cbd5e1'})
    fig.text(.5,.067,'Counts above are queries, not unique lemmas or synonym-pair rows. B and C evaluation cohorts do not overlap.',ha='center',fontsize=11)
    fig.text(.5,.032,'C queries can also contain B words. Ellipse areas are schematic. A recall is unavailable, not zero.',ha='center',fontsize=10,color='#64748b')
    fig.subplots_adjust(top=.79,bottom=.14,left=.035,right=.965,wspace=.08)
    for extension in ['png','svg','pdf']:
        fig.savefig(out / f'A_B_C_performance_diagram.{extension}',dpi=180,facecolor='white')
    plt.close(fig)

    report = {'queries':len(records),'videos':len(videos),'joined_queries':len(by_id),'train_query_matches':0,
              'whitespace_only_text_differences':whitespace_differences,'val_and_test_caption_files_identical':captions_path.read_bytes()==test_path.read_bytes(),
              'overall':metrics(records),'cohorts':summaries,'C_vs_B':comparisons,
              'unclassified_queries':sum(r['has_unclassified'] for r in records),
              'bootstrap':{'unit':'ground-truth video/clip','iterations':args.bootstrap_iterations,'seed':args.seed,'interval':'percentile 95% for C minus B recall, exploratory'},
              'classification':'Per query/aspect: C if any final C term; otherwise B if any B term; otherwise EMPTY only for a raw-empty aspect, else REVIEW_ONLY. Combined C if either aspect is C. Valid terms only; UNCLASSIFIED and EXCLUDED are separately flagged.',
              'term_metric_scope':'Per-term queries can overlap; do not average term metrics to recover overall recall. A has zero evaluated queries and blank recall, not 0% performance.',
              'limitations':'Final lexical labels include 657 context reviews; other senses/POS remain automatic. No direct video review, length/show adjustment, or evidence of augmentation causality. CSV checkpoint identities are not recorded. Archived metrics compare recorded scores, not controlled interventions.',
              'source_sha256':{str(p):sha256(p) for p in inputs},'script_sha256':sha256(Path(__file__))}
    (out / 'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    overall_table = '\n'.join('| '+label+' | '+f"{len(rows):,}"+' | '+' | '.join(f"{metrics(rows)[f'R@{k}']:.2f}" for k in KS)+' |'
                              for label,rows in [('전체',records),('B: C 단어 없음',cohorts[('combined','B_ONLY')]),('C: C 단어 포함',cohorts[('combined','C_PRESENT')])])
    aspect_table = '\n'.join('| '+aspect.title()+' | '+region+' | '+f"{len(cohorts[(aspect,region)]):,}"+' | '+' | '.join(f"{metrics(cohorts[(aspect,region)])[f'R@{k}']:.2f}" for k in KS)+' |'
                             for aspect in ['object','action'] for region in ['B_ONLY','C_PRESENT','EMPTY','REVIEW_ONLY'])
    delta_table = '\n'.join(f"| {r['metric']} | {r['C_minus_B_percentage_points']:+.2f} | [{r['delta_ci95_lower']:+.2f}, {r['delta_ci95_upper']:+.2f}] |" for r in comparisons if r['aspect']=='combined')
    c=metrics(cohorts[('combined','C_PRESENT')])
    b=metrics(cohorts[('combined','B_ONLY')])
    both=sum(r['object_region']=='C_PRESENT' and r['action_region']=='C_PRESENT' for r in records)
    zero_count=sum(r['ci_includes_zero'] for r in comparisons if r['aspect']=='combined')
    interpretation = ('이 데이터에서는 C의 실패가 일관되게 더 높다는 근거가 뚜렷하지 않습니다.' if zero_count == 4 else
                      '위 차이와 구간을 지표별로 해석해야 합니다. C의 WordNet 연결 여부만으로 실패 원인을 확정할 수 없습니다.')
    (out / 'README.md').write_text(f'''# TVR: A/B/C 영역과 검색 성능 비교

입력은 `{args.results}`입니다. 요청한 reslts 철자 파일은 없고 `tvr_test_results.csv`가 확인됐습니다.
{len(records):,}개 쿼리를 최종 문맥 검토 후 영역 CSV와 caption_id로 연결했습니다. Train ID와 겹치는 평가 쿼리는 0개입니다.
정답 순위는 1부터 시작하며 R@K는 정답의 순위가 K 이하인 쿼리 비율입니다. 원본의 네 성공 표시와 gt_rank가 모두 일치합니다.
원본 캡션과 다른 {whitespace_differences}건은 공백 차이뿐입니다. 현재 tvrval.caption.txt와 tvrtest.caption.txt의 내용 동일 여부는 {report['val_and_test_caption_files_identical']}입니다.

## 비교 단위와 분류

- **A:** Train 단어 영역. 이 결과 CSV에는 Train inference가 없어 Recall은 미측정입니다. 0%로 해석하지 않습니다.
- **C_PRESENT:** 해당 쿼리의 Object/Action에서 최종 C 단어가 하나라도 등장하는 경우. 다른 B 단어가 함께 있어도 C로 분류합니다.
- **B_ONLY:** 해당 기준에서 C 단어가 없고 유효한 B 단어가 있는 경우.
- **EMPTY:** 그 항목의 추출 결과가 비어 있는 쿼리.
- **REVIEW_ONLY:** 추출 기록은 있으나 남은 표현이 EXCLUDED/UNCLASSIFIED뿐인 쿼리. EMPTY나 C로 넣지 않습니다.

Combined는 두 항목 중 하나라도 C이면 C_PRESENT입니다. 각 평가 쿼리는 각 비교 기준에서 한 번만 계산합니다.
Object C 701개와 Action C 85개 중 {both}개가 겹쳐 Combined C는 {c['evaluated_queries']:,}개입니다.
기존 단어 영역 CSV의 B 쿼리 수에는 C 단어가 함께 있는 문장이 포함됐으므로 아래 B_ONLY 수와 다릅니다.

## 전체와 Combined 비교

모든 Recall 값은 %입니다.

| 평가 쿼리 구분 | 쿼리 수 | R@1 | R@5 | R@10 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
{overall_table}

A는 Train 쿼리 중 {len(set().union(*train_a_queries.values())):,}개에 A 단어가 있으나 이 CSV의 평가 쿼리는 0개입니다. Object A 포함 Train 쿼리는 {len(train_a_queries['object']):,}개, Action은 {len(train_a_queries['action']):,}개입니다.
이 숫자는 고유 기본형 단어 수(기존 다이어그램의 Object A 2,875 / Action A 882)와 구별해야 합니다.

## Object / Action 별 비교

| 기준 | 쿼리 영역 | 쿼리 수 | R@1 | R@5 | R@10 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{aspect_table}

Object와 Action은 서로 다른 기준으로 같은 쿼리를 분석하므로 두 표의 쿼리 수를 합산하지 않습니다.

## C의 실패가 더 많은가

Combined C의 R@1 실패는 {c['R@1_failures']:,}/{c['evaluated_queries']:,}개({c['failure@1']:.2f}%), R@100 실패는 {c['R@100_failures']:,}/{c['evaluated_queries']:,}개({c['failure@100']:.2f}%)입니다.
B의 R@100 실패는 {b['R@100_failures']:,}/{b['evaluated_queries']:,}개({b['failure@100']:.2f}%)입니다.
지표별 Recall 차이의 방향과 크기는 다음 표에서 확인할 수 있습니다.

| 지표 | C−B Recall 차이(%p) | 클립 재표집 95% 구간(%p) |
| --- | ---: | --- |
{delta_table}

동일 클립의 여러 캡션을 함께 재표집하는 video-cluster bootstrap {args.bootstrap_iterations:,}회(seed {args.seed})를 사용했습니다.
Combined 네 지표 중 {zero_count}개의 차이 구간이 0을 포함합니다. {interpretation}
C 쿼리 {c['R@100_successes']:,}개는 실제로 R@100에서 정답을 찾았습니다. C의 WordNet 미연결 자체로 기존 학습에서 극복 불가능한 샘플이라고 결론 내릴 수 없습니다.
추가 학습으로 개선되는지는 동일 평가 쿼리에서 증강 전후를 비교하는 실험이 필요합니다.

## 해석 범위

최종 영역은 657개 문맥 검토를 반영했으며 나머지 품사와 WordNet 의미는 자동 조회 기준입니다. 영상을 직접 확인한 분류는 아닙니다.
UNCLASSIFIED가 함께 있는 쿼리는 {report['unclassified_queries']}개이며 해당 쿼리를 제외한 민감도 표도 별도 저장했습니다.
Combined 평균 단어 수는 B {b['mean_query_words']:.2f}, C {c['mean_query_words']:.2f}입니다. 길이·TV 시리즈·동작 난이도를 통제한 인과 분석은 아닙니다.
재표집 구간은 기술적·탐색적 비교입니다. 구간에 0이 포함된다고 두 영역의 성능이 같다고 증명한 것은 아닙니다.
두 결과 CSV의 체크포인트 정보는 파일에 기록돼 있지 않습니다. 추가 CSV의 성능 차이를 특정 학습·증강 전략의 효과라고 해석하지 않습니다.
기록된 metrics.json의 비교는 저장된 점수 대조이며 이 파일들에는 영역별 쿼리 결과가 없어 영역별 개선을 계산하지 않았습니다.

## 파일

- [A/B/C 성능 다이어그램 PNG]({out/'A_B_C_performance_diagram.png'}) · [SVG]({out/'A_B_C_performance_diagram.svg'}) · [PDF]({out/'A_B_C_performance_diagram.pdf'})
- [Recall 비교 그래프]({out/'region_recall_comparison.png'})
- [영역별 전체 성능 CSV]({out/'region_recall.csv'})
- [10,895개 쿼리의 영역·정답 순위·성과·C 단어·WordNet 후보]({out/'query_region_results.csv'})
- [A/B/C 단어별 성능 CSV]({out/'term_recall.csv'}) — A는 평가 수 0 / Recall 빈칸. 한 쿼리가 여러 단어에 기여하므로 단어별 점수를 평균내 전체 점수를 계산하지 않습니다.
- [Object×Action 영역 교차표]({out/'object_action_matrix.csv'})
- [C의 R@1 실패 {c['R@1_failures']}개]({out/'C_failure_queries_R1.csv'})
- [C의 R@100 실패 {c['R@100_failures']}개]({out/'C_failure_queries_R100.csv'})
- [C−B 차이·실패율·재표집 구간]({out/'C_vs_B_comparison.csv'})
- [UNCLASSIFIED 동반 쿼리 제외 성능]({out/'region_recall_without_unclassified.csv'})
- [다른 결과 CSV와 쿼리별 성능 대조]({out/'paired_result_comparison.csv'})
- [저장된 전체 실험 성능 대조]({out/'recorded_performance_comparison.csv'})
- [검증·방법·입력 해시]({out/'summary.json'})

쿼리 CSV의 `*_C_wordnet_candidates`는 전체 사전 의미 후보, `*_C_context_wordnet_candidates`와 `*_C_selected_wordnet_synsets`는 해당 기본형의 출현별 선택 의미를 합친 값입니다.
검토되지 않은 출현의 후보는 문맥에서 검증됐다는 의미가 아닙니다. C의 반대 split 후보는 없지만 자체 사전 후보는 있을 수 있습니다.
모든 출력은 압축하지 않은 CSV/그림입니다. 원본 평가·추출 데이터, 모델, 특징 파일, 학습 설정은 변경하지 않았습니다.

## 재현

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/viglab2/miniconda3/envs/holmes/bin/python rewriting/analyze_region_retrieval.py
```
''',encoding='utf-8')
    print(json.dumps({'overall':report['overall'],'C_vs_B':comparisons,'output_dir':str(out)},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
