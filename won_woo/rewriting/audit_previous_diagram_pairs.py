"""Record errors in the previous export using the completed independent audit."""

import csv
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, make_normalizer, write_csv
from build_object_action_regions import sha256
from suggest_val_replacements import load_wordnet


def main():
    import spacy

    workspace = Path(__file__).resolve().parent.parent
    base = workspace / 'rewriting/val_domain_analysis'
    previous = base / 'diagram_synonym_pairs_before_independent_audit'
    if not previous.exists():
        previous = base / 'diagram_synonym_pairs'
    audit = base / 'diagram_region_audit'
    after = audit / 'after'
    assert not json.loads((after / 'summary.json').read_text())['findings']
    sys.path.insert(0, str(workspace / 'rewriting/nlp_resources/python'))
    from lemminflect import getAllLemmas

    nlp = spacy.load('en_core_web_md', exclude=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'senter'])
    normalize = make_normalizer(nlp)
    wn = load_wordnet(workspace / 'rewriting/nlp_resources')
    changes, rejected, corrections, witnesses = {}, {}, {}, {}
    for filename, result in [('region_changes.csv', changes), ('rejected_source_terms.csv', rejected), ('normalization_corrections.csv', corrections)]:
        with (after / filename).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                result[(row['aspect'], row['split'], row['caption_id'], row['source_expression'])] = row
    for split, filename in [('train', 'tvrtrain_domain_analysis.csv'), ('validation', 'tvrval_action_object_analysis.csv')]:
        with (base / filename).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                for kind, tags in [('object', OBJECT_TAGS), ('action', ACTION_TAGS)]:
                    for surface, tag in json.loads(row[kind + 's_raw']).items():
                        witnesses[(kind, split, row['caption_id'], surface)] = normalize(surface, tag) if tag in tags else ''

    @lru_cache(None)
    def senses(kind, word):
        return {sense.name() for sense in wn.synsets(word.replace(' ', '_'), pos=wn.NOUN if kind == 'object' else wn.VERB)}

    known_forms = lru_cache(None)(getAllLemmas)
    findings, categories, bad_rows = [], Counter(), set()
    invalid_sources, changed_sources, corrected_sources = set(), set(), set()
    total = pairs = 0
    previous_hashes = {}
    for path in sorted(previous.glob('*.csv')):
        if not path.name.startswith(('object_', 'action_')):
            continue
        kind, region, suffix = path.stem.split('_')
        split = 'train' if suffix == 'train' else 'validation'
        other, target = ('validation', 'val') if split == 'train' else ('train', 'train')
        previous_hashes[path.name] = sha256(path)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            assert 'original_source_lemma' not in reader.fieldnames, 'Expected the historical export, not the corrected CSVs'
            for number, row in enumerate(reader, 1):
                total += 1
                key = kind, split, row['caption_id'], row['source_expression']
                errors = []
                expected = changes.get(key, {}).get('corrected_region', region)
                if expected != region:
                    errors.append(('REGION_MISMATCH', f'{region} -> {expected}'))
                    changed_sources.add(key)
                if key in rejected and region not in {'EMPTY', 'UNCLASSIFIED'}:
                    errors.append(('INVALID_SOURCE_IN_REGION', rejected[key]['reason']))
                    invalid_sources.add(key)
                if key in corrections:
                    errors.append(('SOURCE_LEMMA_CORRECTION', corrections[key]['original_source_lemma'] + ' -> ' + corrections[key]['source_lemma']))
                    corrected_sources.add(key)
                if int(row['synonym_rank']):
                    pairs += 1
                    witness = kind, other, row[f'{target}_caption_id'], row[f'{target}_expression']
                    word = witnesses[witness]
                    if witness in rejected:
                        errors.append(('INVALID_OPPOSITE_WITNESS', rejected[witness]['reason']))
                    expression = row['synonym_expression'].casefold()
                    dictionary_match = word in known_forms(expression).get('NOUN' if kind == 'object' else 'VERB', ()) and expression not in {'could', 'would', 'should', 'might'}
                    if not senses(kind, expression) & senses(kind, word) and not dictionary_match:
                        errors.append(('INVALID_GENERATED_INFLECTION', row['synonym_expression'] + ' from ' + word))
                for category, details in errors:
                    categories[category] += 1
                    bad_rows.add((path.name, number))
                    findings.append({'category': category, 'csv_file': path.name, 'data_row_number': number,
                                     'aspect': kind, 'split': split, 'caption_id': row['caption_id'],
                                     'source_expression': row['source_expression'], 'source_lemma': row['source_lemma'],
                                     'synonym_expression': row['synonym_expression'], 'actual_region': region,
                                     'expected_region': expected, 'details': details, 'caption': row['caption']})
        print(f'Compared previous {path.name}', flush=True)
    fields = ['category', 'csv_file', 'data_row_number', 'aspect', 'split', 'caption_id', 'source_expression',
              'source_lemma', 'synonym_expression', 'actual_region', 'expected_region', 'details', 'caption']
    write_csv(audit / 'previous_csv_findings.csv', fields, findings)
    report = {'rows_checked': total, 'pairs_checked': pairs, 'rows_with_findings': len(bad_rows),
              'findings_by_category': dict(categories), 'changed_region_source_groups': len(changed_sources),
              'invalid_source_groups_previously_in_ABC': len(invalid_sources), 'normalized_source_groups': len(corrected_sources),
              'row_numbers': 'Header-excluded 1-based row numbers in the previous export; multiple categories may describe the same row',
              'previous_csv_sha256': previous_hashes,
              'comparison_audit_sha256': sha256(after / 'summary.json'),
              'script_sha256': sha256(Path(__file__)),
              'previous_directory_after_promotion': str(base / 'diagram_synonym_pairs_before_independent_audit')}
    (audit / 'previous_csv_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
