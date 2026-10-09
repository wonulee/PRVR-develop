"""Independently audit the region CSVs against source extractions and WordNet."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, make_normalizer, read_captions, write_csv
from build_object_action_regions import distinct_expression, sha256
from export_B_synonym_pairs import expression_pattern
from suggest_val_replacements import load_wordnet


def main():
    import spacy

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=workspace / 'rewriting/val_domain_analysis/diagram_synonym_pairs')
    parser.add_argument('--output-dir', type=Path, default=workspace / 'rewriting/val_domain_analysis/diagram_region_audit')
    parser.add_argument('--review-file', type=Path, default=workspace / 'rewriting/val_domain_analysis/diagram_region_audit/context_review/decisions.csv')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extraction = args.input_dir.parent
    spacy.util.fix_random_seed(2026)
    nlp = spacy.load('en_core_web_md', exclude=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'senter'])
    normalize = make_normalizer(nlp)
    wn = load_wordnet(workspace / 'rewriting/nlp_resources')
    sys.path.insert(0, str(workspace / 'rewriting/nlp_resources/python'))
    from lemminflect import getAllLemmas, getInflection
    dictionary_forms = lru_cache(None)(getAllLemmas)
    inflections = lru_cache(None)(getInflection)
    allowed = {'object': OBJECT_TAGS, 'action': ACTION_TAGS}
    original, source, pools, grounded, source_paths = {}, {}, defaultdict(set), defaultdict(set), []
    rejected, corrections = [], []
    original_lemmas = {}
    original_tags, original_aspects = {}, {}
    unresolved_morphology = set()
    counts = Counter()
    dictionaries = {}
    for kind in allowed:
        path = workspace / 'tvr/TextData' / f'tvr_id_{kind}_dict.json'
        source_paths.append(path)
        dictionaries[kind] = json.loads(path.read_text())

    @lru_cache(None)
    def tokens(caption):
        return {token.text for token in nlp.make_doc(caption)}

    for split, prefix in [('train', 'tvrtrain'), ('validation', 'tvrval')]:
        caption_path = workspace / 'tvr/TextData' / f'{prefix}.caption.txt'
        originals = read_captions(caption_path)
        original[split] = {row['caption_id']: row for row in originals}
        source_paths.append(caption_path)
        path = extraction / ('tvrtrain_domain_analysis.csv' if split == 'train' else 'tvrval_action_object_analysis.csv')
        source_paths.append(path)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                assert row['caption'] == original[split][row['caption_id']]['caption']
                for kind in allowed:
                    raw = json.loads(row[f'{kind}s_raw'])
                    if split == 'train':
                        assert raw == dictionaries[kind][row['caption_id']]
                    mapped = {}
                    legacy_terms = set()
                    for surface, tag in raw.items():
                        old_term = normalize(surface, tag) if tag in allowed[kind] else ''
                        term = old_term
                        correction_rule = ''
                        # Reconstruct the canonical rule directly from the corpus,
                        # independently of the exporter's optional normalizer.
                        if tag in allowed[kind]:
                            pos = wn.NOUN if kind == 'object' else wn.VERB
                            roots = wn._morphy(surface.strip().casefold().replace(' ', '_'), pos)
                            matching = [root for root in roots if surface.strip().casefold() in
                                        {form.casefold() for form in inflections(root.replace('_', ' '), tag)}] if len(roots) > 1 else []
                            if len(roots) == 1:
                                term = roots[0].replace('_', ' ')
                                correction_rule = 'UNIQUE_WORDNET_ROOT'
                            elif len(matching) == 1:
                                term = matching[0].replace('_', ' ')
                                correction_rule = 'UNIQUE_POS_INFLECTION_ROOT'
                            else:
                                term = term.replace('_', ' ')
                                correction_rule = 'CANONICAL_DISPLAY' if term != old_term else ''
                                if len(roots) > 1:
                                    unresolved_morphology.add((kind, split, row['caption_id'], surface))
                        lemma_roots = wn._morphy(term.replace(' ', '_'), wn.NOUN if kind == 'object' else wn.VERB) if term else []
                        original_lemmas[(kind, split, row['caption_id'], surface)] = old_term
                        original_tags[(kind, split, row['caption_id'], surface)] = tag
                        original_aspects[(kind, split, row['caption_id'], surface)] = kind
                        if old_term:
                            legacy_terms.add(old_term)
                            pools[(kind, split)].add(old_term)
                        if term != old_term:
                            corrections.append({'aspect': kind, 'split': split, 'caption_id': row['caption_id'],
                                                'source_expression': surface, 'source_pos': tag,
                                                'original_source_lemma': old_term, 'source_lemma': term,
                                                'correction_rule': correction_rule,
                                                'caption': row['caption']})
                        reason = ('UNSUPPORTED_SOURCE_POS' if tag not in allowed[kind] else
                                  'NONLEXICAL_LEMMA' if not any(c.isalnum() for c in term) else
                                  'UNRESOLVED_CONTRACTION' if surface.casefold().replace('’', "'") in {"'s", "'re", "'m", "'ve", "'d", "'ll"} and term not in {'be', 'have', 'will', 'would'} else
                                  'TRUNCATED_AUXILIARY' if kind == 'action' and term == 's' else
                                  'NONCANONICAL_LEMMA' if lemma_roots and term.replace(' ', '_') not in lemma_roots else
                                  'SOURCE_NOT_OBSERVED' if not expression_pattern(surface).search(row['caption']) and surface not in tokens(row['caption']) else '')
                        mapped[surface] = (term, tag, reason)
                        if not reason:
                            grounded[(kind, split)].add(term)
                        else:
                            rejected.append({'aspect': kind, 'split': split, 'caption_id': row['caption_id'],
                                             'caption': row['caption'], 'source_expression': surface,
                                             'source_lemma': term, 'source_pos': tag, 'reason': reason})
                    assert legacy_terms == set(json.loads(row[f'{kind}_lemmas']))
                    source[(kind, split, row['caption_id'])] = mapped
        print(f'Reconstructed {split} original captions and extraction terms', flush=True)

    # Apply the reviewed CSV independently of the exporter's source adapter.
    review_rows, context_names, review_meta = [], {}, {}
    source_before_context = {key: dict(raw) for key, raw in source.items()}
    automatic_corrections = { (r['aspect'], r['split'], r['caption_id'], r['source_expression']): r['correction_rule'] for r in corrections }
    if args.review_file.exists():
        source_paths.append(args.review_file)
        review_rows = list(csv.DictReader(args.review_file.open(encoding='utf-8-sig', newline='')))
        queue = list(csv.DictReader((args.review_file.parent / 'review_queue.csv').open(encoding='utf-8-sig', newline='')))
        assert len(review_rows) == len(queue) == 657
        assert {(r['aspect'],r['split'],r['caption_id'],r['source_expression']) for r in review_rows} == {(r['aspect'],r['split'],r['caption_id'],r['source_expression']) for r in queue}
        assert len({r['review_id'] for r in review_rows}) == 657
        for row in review_rows:
            kind, split, cid, surface = row['aspect'], row['split'], row['caption_id'], row['source_expression']
            qkey, group = (kind, split, cid), (kind, split, cid, surface)
            term, tag, old_reason = source[qkey][surface]
            assert tag == row['source_pos'] and row['caption'] == original[split][cid]['caption']
            review_meta[group] = row
            unresolved_morphology.discard(group)
            if row['decision'] in {'EXCLUDE', 'MOVE'}:
                source[qkey][surface] = term, tag, 'MOVED_TO_ACTION' if row['decision'] == 'MOVE' else 'CONTEXT_EXCLUDED'
                context_names[group] = set()
                if row['decision'] == 'EXCLUDE':
                    continue
                target_key = row['corrected_aspect'], split, cid
                target_group = target_key + (surface,)
                assert row['corrected_aspect'] == 'action'
                if surface not in source[target_key]:
                    original_lemmas[target_group] = original_lemmas[group]
                    original_tags[target_group] = original_tags[group]
                    original_aspects[target_group] = kind
                review_meta[target_group] = row
                qkey, group = target_key, target_group
            source[qkey][surface] = row['corrected_lemma'], row['corrected_pos'], ''
            context_names[group] = set(row['context_synsets'].split(' | ')) - {''}
            assert context_names[group] <= {s.name() for s in wn.synsets(row['corrected_lemma'].replace(' ','_'), pos='n' if qkey[0]=='object' else 'v')}
    grounded, pooled_senses, rejected, corrections = defaultdict(set), defaultdict(set), [], []
    for key, raw in source.items():
        kind, split, cid = key
        for surface, (term, tag, reason) in raw.items():
            group = key + (surface,)
            if not reason:
                grounded[(kind, split)].add(term)
                selected = context_names.get(group, {s.name() for s in wn.synsets(term.replace(' ','_'), pos='n' if kind=='object' else 'v')})
                pooled_senses[(kind, split, term)].update(selected)
                if term != original_lemmas[group]:
                    corrections.append({'aspect':kind,'split':split,'caption_id':cid,'source_expression':surface,'source_pos':tag,
                                        'original_source_lemma':original_lemmas[group],'source_lemma':term,
                                        'correction_rule':'CAPTION_CONTEXT_REVIEW' if group in review_meta else automatic_corrections.get(group,''),
                                        'caption':original[split][cid]['caption']})
            else:
                rejected.append({'aspect':kind,'split':split,'caption_id':cid,'caption':original[split][cid]['caption'],
                                 'source_expression':surface,'source_lemma':term,'source_pos':tag,'reason':reason})

    @lru_cache(None)
    def senses(kind, term):
        return {s.name() for s in wn.synsets(term.replace(' ', '_'), pos=wn.NOUN if kind == 'object' else wn.VERB)}

    @lru_cache(None)
    def selected_candidates(kind, term, names):
        pos = wn.NOUN if kind == 'object' else wn.VERB
        return sorted({word.replace('_', ' ').casefold() for name in names for word in wn.synset(name).lemma_names()
                       if distinct_expression(wn, term, word, pos)})

    @lru_cache(None)
    def opposite_candidates(kind, split, term, names):
        other = 'validation' if split == 'train' else 'train'
        pos = wn.NOUN if kind == 'object' else wn.VERB
        return sorted({word for name in names for word in grounded_index[(kind, other)][name]
                       if distinct_expression(wn, term, word, pos)})

    @lru_cache(None)
    def definitions(names):
        return ' | '.join(wn.synset(name).definition() for name in names)

    def build_index(vocabulary):
        index = defaultdict(lambda: defaultdict(set))
        for key, words in vocabulary.items():
            for word in words:
                for name in pooled_senses[key + (word,)] if vocabulary is grounded else senses(key[0], word):
                    index[key][name].add(word)
        return index

    raw_index, grounded_index = build_index(pools), build_index(grounded)

    @lru_cache(None)
    def reference(kind, split, term, clean=False):
        vocabulary, index = (grounded, grounded_index) if clean else (pools, raw_index)
        other = 'validation' if split == 'train' else 'train'
        pos = wn.NOUN if kind == 'object' else wn.VERB
        names = pooled_senses[(kind, split, term)] if clean else senses(kind, term)
        connected = {word for name in names for word in index[(kind, other)][name]}
        direct = term in vocabulary[(kind, other)]
        status = 'DIRECT_LEMMA' if direct else 'WORDNET_CANDIDATE' if connected else 'UNCOVERED'
        region = 'B' if direct or connected else 'A' if split == 'train' else 'C'
        synonyms = sorted(word for word in connected if distinct_expression(wn, term, word, pos))
        wordnet_words = sorted({word.replace('_', ' ').casefold()
                                for name in senses(kind, term) for word in wn.synset(name).lemma_names()
                                if distinct_expression(wn, term, word, pos)})
        return region, status, synonyms, wordnet_words

    issues, region_changes = [], []
    review_results, review_pair_counts = {}, Counter()
    fields = ['category', 'csv_file', 'data_row_number', 'aspect', 'split', 'caption_id', 'source_expression',
              'source_lemma', 'actual_region', 'expected_region', 'details', 'caption']
    seen_sources, query_keys = set(), set()
    member_stats = defaultdict(Counter)
    member_lemmas = defaultdict(set)
    empty_annotations = {}
    file_stats = []
    morphological = {}
    old_catalog = {}
    for row in csv.DictReader((extraction / 'object_action_regions/term_coverage.csv').open(encoding='utf-8-sig', newline='')):
        old_catalog[(row['aspect'], row['split'], row['lemma'])] = row
    catalog_errors = []
    for (kind, split), words in pools.items():
        for word in sorted(words):
            expected, status, _, _ = reference(kind, split, word)
            current = old_catalog[(kind, split, word)]
            if (expected, status, senses(kind, word)) != (current['term_region'], current['coverage_status'], set(json.loads(current['source_synsets']))):
                catalog_errors.append({'aspect': kind, 'split': split, 'lemma': word, 'expected_region': expected})

    for path in sorted(args.input_dir.glob('*.csv')):
        if not path.name.startswith(('object_', 'action_')):
            continue
        kind, region, suffix = path.stem.split('_')
        split = 'train' if suffix == 'train' else 'validation'
        target = 'val' if split == 'train' else 'train'
        other = 'validation' if split == 'train' else 'train'
        n = pairs = 0
        previous_group, group_ranks, group_expressions = None, [], []
        local_issues = Counter()

        def check_group():
            if group_ranks and group_ranks != [0]:
                assert group_ranks == list(range(1, len(group_ranks) + 1)), (path.name, previous_group)
                assert group_expressions == sorted(set(group_expressions)), (path.name, previous_group)

        with path.open(encoding='utf-8-sig', newline='') as handle:
            for n, row in enumerate(csv.DictReader(handle), 1):
                qkey = kind, split, row['caption_id']
                assert row['caption'] == original[split][row['caption_id']]['caption']
                assert row['video_id'] == original[split][row['caption_id']]['video_id']
                assert int(row['enc']) == original[split][row['caption_id']]['enc']
                query_keys.add(qkey)
                surface = row['source_expression']
                group = qkey + (surface,)
                if group != previous_group:
                    check_group()
                    assert group not in seen_sources, ('Duplicated source group', group)
                    seen_sources.add(group)
                    previous_group, group_ranks, group_expressions = group, [], []
                rank = int(row['synonym_rank'])
                mkey = path.name, row['caption_id']
                member_stats[mkey]['rows'] += 1
                member_stats[mkey]['pairs' if rank else 'no_pair_rows'] += 1
                if row['source_lemma']:
                    member_lemmas[mkey].add(row['source_lemma'])
                if not rank:
                    assert all(not row[field] for field in ['synonym_expression', 'meaning', f'{target}_expression', f'{target}_caption_id', f'{target}_caption'])
                    empty_annotations[(path.name, n)] = (row['caption_id'], surface, row['source_lemma'], region)
                group_ranks.append(rank)
                group_expressions.append(row['synonym_expression'])
                errors = []
                if surface:
                    term, tag, reason = source[qkey][surface]
                    assert row['source_lemma'] == term
                    assert row['original_source_lemma'] == original_lemmas[group]
                    assert row['source_pos'] == tag and row['original_source_pos'] == original_tags[group]
                    assert row['original_aspect'] == original_aspects[group]
                    selected = context_names.get(group, senses(kind, term))
                    assert set(row['selected_wordnet_synsets'].split(' | ')) - {''} == selected
                    selected_key = tuple(sorted(selected))
                    context_words = selected_candidates(kind, term, selected_key)
                    assert (row['context_wordnet_candidates'].split(' | ') if row['context_wordnet_candidates'] else []) == context_words
                    reviewed = review_meta.get(group)
                    assert row['review_id'] == (reviewed['review_id'] if reviewed else '')
                    assert row['review_decision'] == (reviewed['decision'] if reviewed else 'AUTO')
                    expected, status, synonyms, wn_words = reference(kind, split, term, True) if term else ('UNCLASSIFIED', 'UNCLASSIFIED', [], [])
                    synonyms = opposite_candidates(kind, split, term, selected_key)
                    if reason:
                        expected = 'EXCLUDED' if reason in {'MOVED_TO_ACTION','CONTEXT_EXCLUDED'} else 'UNCLASSIFIED'
                        status = expected
                    if reviewed and kind == (reviewed['corrected_aspect'] if reviewed['decision']=='MOVE' else reviewed['aspect']):
                        rid = reviewed['review_id']
                        review_results.setdefault(rid, {**reviewed, 'final_aspect':kind,'final_lemma':term,'final_region':region,
                                                       'csv_file':path.name,'data_row_number':n,
                                                       'wordnet_candidates':row['wordnet_candidates'],'context_wordnet_candidates':row['context_wordnet_candidates'],
                                                       'other_split_candidates':row['other_split_candidates']})
                        review_pair_counts[rid] += int(rank > 0)
                    if len(group_ranks) == 1:
                        old_term = original_lemmas[group]
                        previous_region = reference(kind, split, old_term)[0] if old_term else 'UNCLASSIFIED'
                        if previous_region != expected:
                            region_changes.append({'aspect': kind, 'split': split, 'caption_id': row['caption_id'],
                                                   'source_expression': surface, 'original_source_lemma': old_term,
                                                   'source_lemma': term, 'previous_region': previous_region,
                                                   'corrected_region': expected, 'reason': reason or 'RECOMPUTED_WORDNET_COVERAGE',
                                                   'caption': row['caption']})
                    if row['term_region'] != region or expected != region:
                        errors.append(('REGION_MISMATCH', expected, f'Independent WordNet/direct-lemma status={status}'))
                    if reason and region not in {'UNCLASSIFIED','EXCLUDED'}:
                        errors.append(('INVALID_SOURCE_IN_REGION', expected, reason))
                    if row['term_coverage_status'] != status:
                        errors.append(('COVERAGE_STATUS_MISMATCH', expected, f'Expected {status}'))
                    if term:
                        listed = row['other_split_candidates'].split(' | ') if row['other_split_candidates'] else []
                        if listed != synonyms:
                            errors.append(('OPPOSITE_CANDIDATES_MISMATCH', expected, 'Observed-token vocabulary candidates differ'))
                        assert int(row['other_split_candidate_count']) == len(listed)
                        assert (row['wordnet_candidates'].split(' | ') if row['wordnet_candidates'] else []) == wn_words
                        assert int(row['wordnet_candidate_count']) == len(wn_words)
                        assert set(row['wordnet_synsets'].split(' | ')) - {''} == senses(kind, term)
                        assert row['has_wordnet_entry'] == str(bool(senses(kind, term)))
                        surface_senses = senses(kind, surface.casefold())
                        if group in unresolved_morphology and surface_senses - senses(kind, term):
                            key = kind, surface, tag, term
                            info = morphological.setdefault(key, {'aspect': kind, 'surface': surface, 'source_pos': tag,
                                                                 'source_lemma': term, 'source_count': 0, 'example_caption': row['caption'],
                                                                 'extra_surface_synsets': ' | '.join(sorted(surface_senses - senses(kind, term)))})
                            if len(group_ranks) == 1:
                                info['source_count'] += 1
                    if rank:
                        assert region == 'B' and not reason, (path.name, n, reason)
                        witness_id = row[f'{target}_caption_id']
                        witness_surface = row[f'{target}_expression']
                        assert row[f'{target}_caption'] == original[other][witness_id]['caption']
                        witness_group = kind, other, witness_id, witness_surface
                        witness_term, _, witness_reason = source[(kind, other, witness_id)][witness_surface]
                        assert not witness_reason, (path.name, n, witness_reason)
                        witness_names = context_names.get(witness_group, senses(kind, witness_term))
                        shared_names = selected & witness_names
                        assert witness_term in synonyms and shared_names
                        expression = row['synonym_expression'].casefold()
                        dictionary_roots = dictionary_forms(expression).get('NOUN' if kind == 'object' else 'VERB', ())
                        dictionary_match = witness_term in dictionary_roots and expression not in {'could', 'would', 'should', 'might'}
                        if not senses(kind, expression) & shared_names and not dictionary_match:
                            errors.append(('INVALID_GENERATED_INFLECTION', expected, f"Generated '{row['synonym_expression']}' does not resolve to witnessed '{witness_term}' WordNet senses"))
                        expected_meaning = definitions(tuple(sorted(shared_names)))
                        assert row['meaning'] == expected_meaning
                        pairs += 1
                else:
                    assert region == 'EMPTY' and not source[qkey]
                    assert rank == 0 and not row['source_lemma']
                if region in {'A', 'C'}:
                    assert rank == 0 and row['other_split_candidate_count'] == '0'
                for category, expected, details in errors:
                    counts[category] += 1
                    local_issues[category] += 1
                    issues.append({'category': category, 'csv_file': path.name, 'data_row_number': n, 'aspect': kind,
                                   'split': split, 'caption_id': row['caption_id'], 'source_expression': surface,
                                   'source_lemma': row['source_lemma'], 'actual_region': region, 'expected_region': expected,
                                   'details': details, 'caption': row['caption']})
        check_group()
        file_stats.append({'csv_file': path.name, 'rows': n, 'pairs': pairs, 'issues': dict(local_issues)})
        print(f'Audited {path.name}: {n:,} rows; issues={dict(local_issues)}', flush=True)

    assert query_keys == set(source), 'Some original query/aspect records are missing'
    expected_sources = {key + (surface,) for key, raw in source.items() for surface in (raw or [''])}
    assert seen_sources == expected_sources, 'Some original source expressions are missing'
    assert not catalog_errors, catalog_errors[:10]
    assert len(review_results) == len(review_rows)
    for rid, row in review_results.items():
        row['exported_pair_count'] = review_pair_counts[rid]
    if review_results:
        write_csv(args.output_dir / 'review_resolution.csv', list(next(iter(review_results.values()))), sorted(review_results.values(), key=lambda r:int(r['review_id'])))
    # Cross-check the membership and rank-0 sidecars against the streamed rows.
    for row in csv.DictReader((args.input_dir / 'query_membership.csv').open(encoding='utf-8-sig', newline='')):
        key = row['csv_file'], row['caption_id']
        actual = member_stats.pop(key)
        assert (int(row['exported_rows']), int(row['pair_count']), int(row['no_pair_rows'])) == (actual['rows'], actual['pairs'], actual['no_pair_rows'])
        assert set(json.loads(row['source_lemmas'])) == member_lemmas.pop(key, set())
    assert not member_stats and not member_lemmas
    for row in csv.DictReader((args.input_dir / 'no_pair_annotations.csv').open(encoding='utf-8-sig', newline='')):
        key = row['csv_file'], int(row['data_row_number'])
        assert empty_annotations.pop(key) == (row['caption_id'], row['source_expression'], row['source_lemma'], row['region'])
    assert not empty_annotations
    exported_summary = json.loads((args.input_dir / 'summary.json').read_text())
    for row in exported_summary['statistics']:
        actual = next(record for record in file_stats if record['csv_file'] == row['csv_file'])
        assert (row['rows'], row['pairs'], row['no_pair_rows']) == (actual['rows'], actual['pairs'], actual['rows'] - actual['pairs'])
    for path, digest in exported_summary['inputs'].items():
        assert sha256(Path(path)) == digest, ('Changed input', path)
    for path, digest in exported_summary['implementation_sha256'].items():
        assert sha256(Path(path)) == digest, ('Changed implementation', path)
    assert sha256(workspace / 'rewriting/export_diagram_synonym_pairs.py') == exported_summary['script_sha256']
    for row in corrections:
        row['previous_region'] = reference(row['aspect'], row['split'], row['original_source_lemma'])[0]
        row['corrected_region'] = reference(row['aspect'], row['split'], row['source_lemma'], True)[0]
    references = []
    for key, words in grounded.items():
        for term in sorted(words):
            region, status, synonyms, _ = reference(*key, term, True)
            references.append({'aspect': key[0], 'split': key[1], 'lemma': term, 'term_region': region,
                               'coverage_status': status, 'other_split_candidates': ' | '.join(synonyms)})
    write_csv(args.output_dir / 'findings.csv', fields, issues)
    write_csv(args.output_dir / 'rejected_source_terms.csv', ['aspect', 'split', 'caption_id', 'caption', 'source_expression', 'source_lemma', 'source_pos', 'reason'], rejected)
    write_csv(args.output_dir / 'morphology_requires_review.csv', ['aspect', 'surface', 'source_pos', 'source_lemma', 'source_count', 'example_caption', 'extra_surface_synsets'], morphological.values())
    write_csv(args.output_dir / 'independent_term_reference.csv', ['aspect', 'split', 'lemma', 'term_region', 'coverage_status', 'other_split_candidates'], references)
    term_reports = []
    for row in references:
        kind, split, term = row['aspect'], row['split'], row['lemma']
        pos = wn.NOUN if kind == 'object' else wn.VERB
        selected = pooled_senses[(kind, split, term)]
        candidates = sorted({word.replace('_', ' ').casefold() for name in selected for word in wn.synset(name).lemma_names()
                             if distinct_expression(wn, term, word, pos)})
        term_reports.append({**row, 'wordnet_candidates': ' | '.join(reference(kind, split, term, True)[3]),
                             'context_wordnet_candidates': ' | '.join(candidates),
                             'wordnet_synsets': ' | '.join(sorted(senses(kind, term))),
                             'selected_wordnet_synsets': ' | '.join(sorted(selected)),
                             'sense_scope': 'Union of selected occurrence senses; unreviewed occurrences use all same-POS senses'})
    write_csv(args.output_dir / 'final_term_regions.csv', list(term_reports[0]), term_reports)
    unknown = [row for row in references if not senses(row['aspect'], row['lemma'])]
    write_csv(args.output_dir / 'terms_without_wordnet_entry.csv', ['aspect', 'split', 'lemma', 'term_region', 'coverage_status', 'other_split_candidates'], unknown)
    write_csv(args.output_dir / 'normalization_corrections.csv', ['aspect', 'split', 'caption_id', 'source_expression', 'source_pos', 'original_source_lemma', 'source_lemma', 'correction_rule', 'previous_region', 'corrected_region', 'caption'], corrections)
    write_csv(args.output_dir / 'region_changes.csv', ['aspect', 'split', 'caption_id', 'source_expression', 'original_source_lemma', 'source_lemma', 'previous_region', 'corrected_region', 'reason', 'caption'], region_changes)
    report = {'files': file_stats, 'rows_checked': sum(r['rows'] for r in file_stats), 'pairs_checked': sum(r['pairs'] for r in file_stats),
              'query_aspect_records_checked': len(query_keys), 'source_expression_groups_checked': len(seen_sources),
              'findings': dict(counts), 'rejected_source_terms': len(rejected),
              'context_reviewed_occurrences':len(review_rows),'context_review_decisions':dict(Counter(r['decision'] for r in review_rows)),
              'new_action_expressions':sum(surface not in source_before_context[key] for key,raw in source.items() if key[0]=='action' for surface in raw),
              'rejected_reasons': dict(Counter(r['reason'] for r in rejected)),
              'morphology_review_groups': len(morphological), 'catalog_classification_errors': len(catalog_errors),
              'normalization_corrected_source_occurrences': len(corrections),
              'normalization_corrected_groups': len({(r['aspect'], r['source_expression'], r['source_pos'], r['original_source_lemma'], r['source_lemma']) for r in corrections}),
              'normalization_correction_rules': dict(Counter(r['correction_rule'] for r in corrections)),
              'normalization_changed_region_occurrences': sum(r['previous_region'] != r['corrected_region'] for r in corrections),
              'changed_region_source_occurrences': len(region_changes),
              'region_changes': dict(Counter(r['previous_region'] + '->' + r['corrected_region'] for r in region_changes)),
              'independent_vocabulary_sizes': {kind + ':' + split: len(words) for (kind, split), words in grounded.items()},
              'unknown_wordnet_term_records': len(unknown),
              'method': 'Original extraction raw/POS -> unique same-POS WordNet root or unique POS-inflection-matching root or legacy spaCy lemma -> same-POS WordNet synsets/direct lemma; also check observed tokens, all candidate witnesses, generated inflections, membership and rank-0 sidecars',
              'limitations': '657 caption-specific reviews correct POS/lemmas and select source senses; other occurrences retain automatic lexical lookup. Reviews use caption text and same-clip caption corroboration, not direct video inspection. All replacement candidates still require review.',
              'wordnet_version': wn.get_version(), 'seed': 2026,
              'source_inputs_sha256': {str(p.resolve()): sha256(p) for p in source_paths},
              'checked_csv_sha256': {str(p.resolve()): sha256(p) for p in args.input_dir.glob('*.csv')},
              'audit_script_sha256': sha256(Path(__file__))}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'files', 'source_inputs_sha256'}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
