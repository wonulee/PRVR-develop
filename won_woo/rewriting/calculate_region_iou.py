"""Calculate Train/Val IoU using common lemma and selected WordNet-synset units."""

import csv
import json
from pathlib import Path

from analyze_val_domain import write_csv
from build_object_action_regions import sha256


def main():
    base = Path(__file__).resolve().parent / 'val_domain_analysis'
    source = base / 'diagram_region_audit/context_review/final_term_regions.csv'
    original_summary = base / 'diagram_region_audit/context_review/final_region_summary.csv'
    out = base / 'region_iou'
    out.mkdir(parents=True, exist_ok=True)
    with source.open(encoding='utf-8-sig', newline='') as handle:
        terms = list(csv.DictReader(handle))
    with original_summary.open(encoding='utf-8-sig', newline='') as handle:
        original_counts = {(r['aspect'], r['split'], r['region']): int(r['unique_source_lemmas'])
                           for r in csv.DictReader(handle) if r['region'] in {'A', 'B', 'C'}}
    assert len({(r['aspect'], r['split'], r['lemma']) for r in terms}) == len(terms)
    results = []
    sets_for_check = []
    selected_synset_missing = {}
    for aspect in ('object', 'action'):
        valid = {split: [r for r in terms if r['aspect'] == aspect and r['split'] == split
                         and r['term_region'] in {'A', 'B', 'C'}]
                 for split in ('train', 'validation')}
        for split, rows in valid.items():
            assert len(rows) == sum(n for (a, s, _), n in original_counts.items()
                                   if a == aspect and s == split)
            selected_synset_missing[f'{aspect}_{split}'] = sum(not r['selected_wordnet_synsets'] for r in rows)
        for basis in ('lemma', 'selected_wordnet_synset'):
            vocabulary = {split: {unit for r in rows for unit in
                                 ([r['lemma']] if basis == 'lemma' else r['selected_wordnet_synsets'].split(' | '))
                                 if unit} for split, rows in valid.items()}
            train, val = vocabulary['train'], vocabulary['validation']
            a, b, c = train - val, train & val, val - train
            union = train | val
            assert not (a & b or a & c or b & c)
            assert a | b == train and b | c == val and a | b | c == union
            assert len(union) == len(train) + len(val) - len(b)
            results.append({'aspect': aspect, 'basis': basis,
                            'train_units': len(train), 'val_units': len(val),
                            'A_train_only': len(a), 'B_intersection': len(b), 'C_val_only': len(c),
                            'union_units': len(union), 'iou': len(b) / len(union),
                            'iou_percent': 100 * len(b) / len(union),
                            'train_coverage_percent': 100 * len(b) / len(train),
                            'val_coverage_percent': 100 * len(b) / len(val)})
            sets_for_check.append((aspect, basis, train, val))
    write_csv(out / 'object_action_ABC_iou.csv', list(results[0]), results)
    # Check exported counts against an independent union identity and set-membership count.
    with (out / 'object_action_ABC_iou.csv').open(encoding='utf-8-sig', newline='') as handle:
        exported = list(csv.DictReader(handle))
        assert len(exported) == len(sets_for_check)
        for row, (aspect, basis, train, val) in zip(exported, sets_for_check):
            intersection = sum(unit in val for unit in train)
            union = len(train) + len(val) - intersection
            assert row['aspect'] == aspect and row['basis'] == basis
            assert int(row['B_intersection']) == intersection
            assert int(row['union_units']) == union
            assert abs(float(row['iou']) - intersection / union) < 1e-12
    metadata = {'formula': '|Train intersection Val| / |Train union Val| = B / (A+B+C)',
                'unit': {'lemma': 'Distinct extracted valid lemma, separately for Object and Action',
                         'selected_wordnet_synset': 'Distinct selected same-POS WordNet synset; not lemma count'},
                'selected_synset_missing_lemma_counts': selected_synset_missing,
                'original_WordNet_coverage_regions': {aspect: {
                    f'{split}_{region}': count for (a, split, region), count in original_counts.items() if a == aspect}
                    for aspect in ('object', 'action')},
                'sources_sha256': {str(p): sha256(p) for p in (source, original_summary)},
                'script_sha256': sha256(Path(__file__)), 'csv_sha256': sha256(out / 'object_action_ABC_iou.csv'),
                'verification': 'Disjoint A/B/C partitions, input summary reconciliation, exported intersection and union recomputation passed',
                'scope': 'Vocabulary overlap; not query-ID overlap, literal full-caption overlap, frequency weighting, or retrieval performance'}
    (out / 'methodology.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Object / Action Train–Val IoU', '',
             'IoU = B / (A+B+C). A=Train 전용, B=공통 교집합, C=Val 전용이며, 각 기준의 동일 단위 집합으로 다시 계산했습니다.', '',
             '| 기준 | 유형 | A | B | C | 합집합 | IoU |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['basis']} | {r['aspect']} | {r['A_train_only']:,} | {r['B_intersection']:,} | {r['C_val_only']:,} | {r['union_units']:,} | {r['iou_percent']:.2f}% |")
    lines += ['', '## 기존 A/B/C와 다른 이유', '',
              '기존 WordNet 연결 기준 B는 Train에서 연결되는 기본형 수와 Val에서 연결되는 기본형 수를 각각 셉니다. 여러 표현이 같은 의미에 연결되므로 두 B 수가 다릅니다.',
              '따라서 기존 A/B/C 기본형 수 중 Train B 또는 Val B 하나를 골라 B/(A+B+C)에 넣으면 집합 IoU가 아닙니다. 기존 영역 파일은 변경하지 않았습니다.',
              'lemma 기준은 동일 기본형끼리만 교집합으로 셉니다. 기존 WordNet 기준 B에 있던 유의어의 서로 다른 기본형은 이 기준에서는 A 또는 C에 들어갈 수 있습니다.',
              'selected_wordnet_synset 기준은 단어가 아니라 선택 의미 ID 집합의 겹침입니다. 한 기본형에 여러 의미가 있어 단어 수보다 많을 수 있습니다.',
              '문맥 미검토 항목은 전체 같은 품사 의미를 유지하므로 synset IoU가 문장 문맥에서 검증된 의미 일치율을 뜻하지 않습니다.',
              '선택 synset이 없는 기본형은 synset 집합에 포함되지 않습니다. 그 수는 methodology.json에 있습니다. lemma 기준에는 포함됩니다.',
              'EMPTY / EXCLUDED / UNCLASSIFIED는 단어 집합에서 제외합니다. 고유 기본형 집합이므로 출현 빈도는 가중하지 않습니다.',
              '단어 IoU를 보기 위해 Train/Val의 서로 다른 caption_id를 직접 교집합으로 세지 않았습니다. 이 수치는 검색 성능이나 A 학습의 유해성을 측정하지 않습니다.', '',
              '[결과 CSV](object_action_ABC_iou.csv) · [계산 기준·입력 해시·검증](methodology.json)', '',
              '재계산: `python rewriting/calculate_region_iou.py`']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
