"""Export Object/Action words into their own coverage regions as plain CSVs."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, make_normalizer, dictionary_sources, validate_dictionary_terms, write_csv
from build_object_action_regions import distinct_expression, plot_regions, sha256
from export_B_synonym_pairs import expression_observed, lemma_issue
from context_review import index_sources, load_reviews, reviewed_terms
from suggest_val_replacements import load_wordnet


def pair_fields(split):
    target = "val" if split == "train" else "train"
    return ["caption_id", "video_id", "enc", "caption", "term_type", "source_expression",
            "synonym_expression", "synonym_rank", "meaning", f"{target}_expression",
            f"{target}_caption_id", f"{target}_caption", "requires_review"]


CANDIDATE_FIELDS = ["source_lemma", "has_wordnet_entry", "wordnet_candidate_count",
                    "wordnet_candidates", "other_split_candidate_count",
                    "other_split_candidates", "wordnet_synsets"]
REGION_FIELDS = ["term_region", "term_coverage_status", "query_region", "original_source_lemma"]
REVIEW_FIELDS = ["source_pos", "original_aspect", "original_source_pos", "selected_wordnet_synsets",
                "context_wordnet_candidates", "review_id", "review_decision", "review_reason"]
DICTIONARY_FIELDS = ["dictionary_file", "source_dictionary_caption_id", "synonym_dictionary_caption_id",
                     "target_dictionary_caption_id", "target_source_pos"]


def csv_fields(split):
    return pair_fields(split) + CANDIDATE_FIELDS + REGION_FIELDS + REVIEW_FIELDS + DICTIONARY_FIELDS


def main():
    import spacy

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "rewriting/val_domain_analysis/dictionary_only/object_action_regions")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-file", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir or args.input_dir.parent / "diagram_synonym_pairs"
    out.mkdir(parents=True, exist_ok=True)
    inputs = [args.input_dir / "methodology.json", args.input_dir / "term_coverage.csv", args.input_dir / "query_regions.csv"]
    metadata = json.loads(inputs[0].read_text())
    for info in metadata["inputs"].values():
        if sha256(Path(info["path"])) != info["sha256"]:
            raise ValueError(f"Diagram input has changed: {info['path']}; rebuild regions first")
    sys.path.insert(0, str(workspace / "rewriting/nlp_resources/python"))
    from lemminflect import getInflection

    spacy.util.fix_random_seed(2026)
    nlp = spacy.load("en_core_web_md", exclude=["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "senter"])
    wn = load_wordnet(workspace / "rewriting/nlp_resources")
    legacy_normalize = make_normalizer(nlp)
    normalize = make_normalizer(nlp, wn, getInflection)
    dictionaries, vocabularies = dictionary_sources(args.input_dir.parent)
    extraction_summary = json.loads((args.input_dir.parent / "summary.json").read_text())
    dictionary_paths = {kind: Path(extraction_summary["inputs"][f"tvr_id_{kind}_dict.json"]["path"])
                        for kind in dictionaries}
    inputs.extend(dictionary_paths.values())
    dictionary_forms = defaultdict(set)
    dictionary_surfaces = defaultdict(set)
    dictionary_by_sense = defaultdict(set)
    dictionary_rows = []
    for kind, vocabulary in vocabularies.items():
        tags = OBJECT_TAGS if kind == "object" else ACTION_TAGS
        pos = wn.NOUN if kind == "object" else wn.VERB
        for surface, evidence in sorted(vocabulary.items()):
            for tag, caption_id in sorted(evidence.items()):
                term = normalize(surface, tag) if tag in tags else ""
                dictionary_rows.append({"aspect": kind, "surface": surface, "source_pos": tag,
                                        "lemma": term, "dictionary_caption_id": caption_id,
                                        "dictionary_file": str(dictionary_paths[kind])})
                if tag not in tags or lemma_issue(kind, surface, term, wn):
                    continue
                dictionary_forms[(kind, term, tag)].add(surface)
                dictionary_surfaces[(kind, term)].add(surface)
                for sense in wn.synsets(term.replace(" ", "_"), pos=pos):
                    dictionary_by_sense[(kind, sense.name())].add(term)
    write_csv(out / "dictionary_terms.csv", list(dictionary_rows[0]), dictionary_rows)
    assert normalize("walks", "VBZ") == "walk"
    assert normalize("miss", "VB") == "miss"
    assert normalize("access", "VB") == "access"
    assert normalize("fling", "VB") == "fling"
    assert normalize("singing", "VBG") == "sing"
    assert normalize("found", "VBD") == "find"
    assert normalize("found", "VB") == "found"
    assert normalize("lay", "VBD") == "lie"
    assert getInflection("buy", "VBD")[0] == "bought"
    reviews = load_reviews(args.review_file)
    if reviews:
        inputs.append(args.review_file)
    representatives, senses, witness_by_sense, source_paths = index_sources(args.input_dir.parent, normalize, nlp.make_doc, wn, reviews)
    inputs.extend(source_paths)
    assert wn.get_version() == metadata["wordnet_version"]
    allowed = {"object": OBJECT_TAGS, "action": ACTION_TAGS}
    coverage, by_sense = {}, defaultdict(lambda: defaultdict(set))
    # Derive the vocabulary from original observed tokens, not the historical
    # term catalogue: its isolated spaCy lemmatization can truncate base verbs.
    for key in representatives:
        kind, split, term = key
        coverage[key] = {}
        for name in senses[key]:
            by_sense[key[:2]][name].add(term)
    for (kind, split, term), info in coverage.items():
        other = "validation" if split == "train" else "train"
        direct = (kind, other, term) in representatives
        connected = any(by_sense[(kind, other)][name] for name in senses[(kind, split, term)])
        info["coverage_status"] = "DIRECT_LEMMA" if direct else "WORDNET_CANDIDATE" if connected else "UNCOVERED"
        info["term_region"] = "B" if direct or connected else "A" if split == "train" else "C"

    @lru_cache(None)
    def options(kind, split, source, selected_names):
        other = "validation" if split == "train" else "train"
        pos = wn.NOUN if kind == "object" else wn.VERB
        names = set(selected_names)
        targets = {word for name in names for word in by_sense[(kind, other)][name]
                   if distinct_expression(wn, source, word, pos)}
        return [(word, tuple(sorted(names & senses[(kind, other, word)])))
                for word in sorted(targets)]

    @lru_cache(None)
    def candidate_columns(kind, split, source, selected_names):
        if not source:
            return dict(zip(CANDIDATE_FIELDS, ["", "", 0, "", 0, "", ""]))
        pos = wn.NOUN if kind == "object" else wn.VERB
        source_senses = wn.synsets(source.replace(" ", "_"), pos=pos)
        names = sorted({sense.name() for sense in source_senses})
        words = sorted({surface for name in names for term in dictionary_by_sense[(kind, name)]
                        if distinct_expression(wn, source, term, pos)
                        for surface in dictionary_surfaces[(kind, term)]})
        other = "validation" if split == "train" else "train"
        observed = sorted({representatives[(kind, other, word)]["expression"]
                           for name in selected_names for word in by_sense[(kind, other)][name]
                           if distinct_expression(wn, source, word, pos)})
        return {"source_lemma": source, "has_wordnet_entry": bool(names),
                "wordnet_candidate_count": len(words), "wordnet_candidates": " | ".join(words),
                "other_split_candidate_count": len(observed), "other_split_candidates": " | ".join(observed),
                "wordnet_synsets": " | ".join(names)}

    @lru_cache(None)
    def verified_forms(kind, target, tag, selected_names=None):
        # Select existing supplied surface/POS pairs; never generate inflections.
        return tuple(sorted(dictionary_forms[(kind, target, tag)]))

    assert lemma_issue("action", "beings", "being", wn) == "NONCANONICAL_LEMMA"
    assert not lemma_issue("object", "comics", "comic strip", wn)
    assert not verified_forms("action", "being", "VBD")
    assert all("VBD" in vocabularies["action"][form] for form in verified_forms("action", "ready", "VBD"))
    cases = [(kind, split, region) for kind in allowed for region, split in [
        ("A", "train"), ("B", "train"), ("B", "validation"), ("C", "validation"),
        ("EMPTY", "train"), ("EMPTY", "validation"),
        ("UNCLASSIFIED", "train"), ("UNCLASSIFIED", "validation")]]
    filenames = {case: f"{case[0]}_{case[2]}_{'train' if case[1] == 'train' else 'val'}.csv" for case in cases}
    counts, skipped_candidates = Counter(), Counter()
    original_query_counts = Counter()
    exported_lemmas = defaultdict(set)
    membership_fields = ["region", "query_region", "aspect", "split", "caption_id", "video_id", "enc", "caption",
                         "source_lemmas", "term_count", "query_coverage_status", "query_uncovered_lemmas", "query_terms_raw",
                         "csv_file", "exported_rows", "pair_count", "no_pair_rows"]
    annotation_fields = ["csv_file", "data_row_number", "region", "query_region", "aspect", "split", "caption_id",
                         "source_expression", "source_lemma", "status", "reason"]
    with ExitStack() as stack:
        writers = {}
        for case, filename in filenames.items():
            handle = stack.enter_context((out / filename).open("w", encoding="utf-8-sig", newline=""))
            writers[case] = csv.DictWriter(handle, fieldnames=csv_fields(case[1]))
            writers[case].writeheader()
        handle = stack.enter_context((out / "query_membership.csv").open("w", encoding="utf-8-sig", newline=""))
        membership_writer = csv.DictWriter(handle, fieldnames=membership_fields)
        membership_writer.writeheader()
        handle = stack.enter_context((out / "no_pair_annotations.csv").open("w", encoding="utf-8-sig", newline=""))
        annotation_writer = csv.DictWriter(handle, fieldnames=annotation_fields)
        annotation_writer.writeheader()
        handle = stack.enter_context(inputs[2].open(encoding="utf-8-sig", newline=""))
        for query_index, query in enumerate(csv.DictReader(handle), 1):
            kind, split, region = query["aspect"], query["split"], query["region"]
            original_query_counts[(kind, split, region)] += 1
            target_prefix = "val" if split == "train" else "train"
            other = "validation" if split == "train" else "train"
            original_raw = json.loads(query["original_terms_raw"])
            validate_dictionary_terms(original_raw, kind, split, query["caption_id"], dictionaries[kind], vocabularies[kind])
            reviewed = reviewed_terms(original_raw, kind, split, query["caption_id"], query["caption"], reviews)
            raw = {surface: info["tag"] for surface, info in reviewed.items()}
            q_rows = q_pairs = q_empty = 0
            q_case_rows, q_case_pairs, q_case_empty = Counter(), Counter(), Counter()
            q_case_lemmas = defaultdict(set)
            for surface in (raw or [""]):
                term, original_term, reason, pairs = "", "", "", []
                selected_names, review = (), None
                source_info = reviewed.get(surface, {"original_aspect": kind, "original_tag": "", "excluded": ""})
                if not surface:
                    reason = "NO_EXTRACTED_TERMS"
                elif raw[surface] not in allowed[kind]:
                    reason = "UNSUPPORTED_SOURCE_POS"
                else:
                    review = source_info.get("review")
                    term = review["corrected_lemma"] if review and not source_info["excluded"] else normalize(surface, raw[surface])
                    original_term = legacy_normalize(surface, source_info["original_tag"])
                    pos = wn.NOUN if kind == "object" else wn.VERB
                    full_names = {sense.name() for sense in wn.synsets(term.replace(" ", "_"), pos=pos)}
                    selected_names = tuple(sorted(set(review["context_synsets"].split(" | ")) - {""} if review and not source_info["excluded"] else full_names if not source_info["excluded"] else set()))
                    if source_info["excluded"]:
                        reason = source_info["excluded"]
                    elif lemma_issue(kind, surface, term, wn):
                        reason = lemma_issue(kind, surface, term, wn)
                    elif not expression_observed(surface, query["caption"], nlp.make_doc):
                        reason = "SOURCE_NOT_OBSERVED"
                    else:
                        alternatives = {}
                        for target, shared_names in options(kind, split, term, selected_names):
                            witness = next((witness_by_sense[(kind, other, target, name)] for name in shared_names
                                            if (kind, other, target, name) in witness_by_sense), None)
                            if not witness:
                                skipped_candidates["no_actual_opposite_expression"] += 1
                                continue
                            if " " in target:
                                skipped_candidates["multiword_inflection_requires_head_review"] += 1
                                continue
                            actual_names = tuple(sorted(set(selected_names) & witness["synsets"]))
                            meaning = " | ".join(wn.synset(name).definition() for name in actual_names)
                            forms = verified_forms(kind, target, raw[surface], actual_names)
                            if not forms:
                                skipped_candidates["no_verified_source_pos_inflection"] += 1
                                continue
                            capitalized = surface[:1].isupper() and surface[1:].islower()
                            matching_forms = [form for form in forms if form[:1].isupper() == capitalized]
                            if not matching_forms:
                                skipped_candidates["matching_case_form_outside_supplied_dictionary"] += 1
                                continue
                            expression = witness["expression"] if witness["expression"] in matching_forms else matching_forms[0]
                            if expression.casefold() == surface.casefold():
                                skipped_candidates["no_surface_change"] += 1
                                continue
                            alternatives.setdefault(expression, (meaning, witness))
                        for rank, (expression, (meaning, witness)) in enumerate(sorted(alternatives.items()), 1):
                            pair = {key: query[key] for key in ["caption_id", "video_id", "enc", "caption"]}
                            pair.update({"term_type": kind, "source_expression": surface, "synonym_expression": expression,
                                         "synonym_rank": rank, "meaning": meaning, f"{target_prefix}_expression": witness["expression"],
                                         f"{target_prefix}_caption_id": witness["caption_id"], f"{target_prefix}_caption": witness["caption"],
                                         "requires_review": True,
                                         "synonym_dictionary_caption_id": vocabularies[kind][expression][raw[surface]],
                                         "target_dictionary_caption_id": (witness["caption_id"] if other == "train" else
                                             vocabularies[kind][witness["expression"]][witness["tag"]]),
                                         "target_source_pos": witness["tag"]})
                            pairs.append(pair)
                    if not pairs and not reason:
                        reason = "NO_SELECTED_WORDNET_SENSE" if not selected_names else (
                            "NO_DISTINCT_OTHER_SPLIT_SYNONYM" if not options(kind, split, term, selected_names) else "NO_EXPORTABLE_OTHER_SPLIT_PAIR")
                info = coverage.get((kind, split, term))
                invalid = reason in {"NONLEXICAL_LEMMA", "NONCANONICAL_LEMMA", "UNRESOLVED_CONTRACTION", "TRUNCATED_AUXILIARY", "UNSUPPORTED_SOURCE_POS", "SOURCE_NOT_OBSERVED", "NO_RAW_EXPRESSION"}
                term_region = "EXCLUDED" if source_info["excluded"] else "EMPTY" if not surface and not raw else info["term_region"] if info and not invalid else "UNCLASSIFIED"
                case = kind, split, term_region
                if case not in writers:
                    cases.append(case)
                    filenames[case] = f"{kind}_{term_region}_{'train' if split == 'train' else 'val'}.csv"
                    handle = stack.enter_context((out / filenames[case]).open("w", encoding="utf-8-sig", newline=""))
                    writers[case] = csv.DictWriter(handle, fieldnames=csv_fields(split))
                    writers[case].writeheader()
                if term_region in {"A", "C"}:
                    assert info["coverage_status"] == "UNCOVERED" and not pairs
                if not pairs:
                    pair = {key: "" for key in pair_fields(split)}
                    pair.update({key: query[key] for key in ["caption_id", "video_id", "enc", "caption"]})
                    pair.update({"term_type": kind, "source_expression": surface, "synonym_rank": 0, "requires_review": True})
                    annotation_writer.writerow({"csv_file": filenames[case], "data_row_number": counts[(case, "rows")] + 1,
                        "region": term_region, "query_region": region, "aspect": kind, "split": split, "caption_id": query["caption_id"],
                        "source_expression": surface, "source_lemma": term, "status": "NO_PAIR", "reason": reason})
                    counts[(case, "no_pair_rows")] += 1
                    q_empty += 1
                    q_case_empty[case] += 1
                    pairs = [pair]
                else:
                    counts[(case, "pairs")] += len(pairs)
                    q_pairs += len(pairs)
                    q_case_pairs[case] += len(pairs)
                extra = candidate_columns(kind, split, term, selected_names)
                extra.update({"term_region": term_region,
                              "term_coverage_status": info["coverage_status"] if info and term_region not in {"UNCLASSIFIED", "EXCLUDED"} else term_region,
                              "query_region": region, "original_source_lemma": original_term})
                context_words = sorted({form for name in selected_names for word in dictionary_by_sense[(kind, name)]
                                        if distinct_expression(wn, term, word, wn.NOUN if kind == "object" else wn.VERB)
                                        for form in dictionary_surfaces[(kind, word)]})
                extra.update({"source_pos": raw.get(surface, ""), "original_aspect": source_info["original_aspect"],
                              "original_source_pos": source_info["original_tag"], "selected_wordnet_synsets": " | ".join(selected_names),
                              "context_wordnet_candidates": " | ".join(context_words), "review_id": review["review_id"] if review else "",
                              "review_decision": review["decision"] if review else "AUTO", "review_reason": review["reason"] if review else ""})
                extra.update({"dictionary_file": str(dictionary_paths[kind]),
                              "source_dictionary_caption_id": (query["caption_id"] if split == "train" else
                                  vocabularies[kind][surface][raw[surface]]) if surface else ""})
                assert term_region not in {"A", "C"} or extra["other_split_candidate_count"] == 0
                if term:
                    exported_lemmas[case].add(term)
                    q_case_lemmas[case].add(term)
                for pair in pairs:
                    pair.update(extra)
                    writers[case].writerow(pair)
                    counts[(case, "rows")] += 1
                    q_rows += 1
                    q_case_rows[case] += 1
            assert q_rows == q_pairs + q_empty and q_rows > 0
            for case, row_count in q_case_rows.items():
                counts[(case, "queries")] += 1
                counts[(case, "queries_with_pairs")] += int(q_case_pairs[case] > 0)
                member = {key: query[key] for key in ["aspect", "split", "caption_id", "video_id", "enc", "caption"]}
                member.update({"region": case[2], "query_region": region,
                               "source_lemmas": json.dumps(sorted(q_case_lemmas[case])), "term_count": len(q_case_lemmas[case]),
                               "query_coverage_status": query["coverage_status"], "query_uncovered_lemmas": query["uncovered_lemmas"],
                               "query_terms_raw": query["original_terms_raw"], "csv_file": filenames[case],
                               "exported_rows": row_count, "pair_count": q_case_pairs[case], "no_pair_rows": q_case_empty[case]})
                membership_writer.writerow(member)
            if query_index % 40000 == 0:
                print(f"Exported {query_index:,} diagram query/aspect rows", flush=True)
    for row in metadata["regions"]:
        assert original_query_counts[(row["aspect"], row["split"], row["region"])] == row["query_count"]
    statistics = [{"aspect": kind, "split": split, "region": region, "csv_file": filenames[(kind, split, region)],
                   "unique_source_lemmas": len(exported_lemmas[(kind, split, region)]),
                   **{metric: counts[((kind, split, region), metric)] for metric in ["queries", "rows", "pairs", "no_pair_rows", "queries_with_pairs"]}}
                  for kind, split, region in cases]
    write_csv(out / "file_counts.csv", list(statistics[0]), statistics)
    (out / "csv_schema.json").write_text(json.dumps({split: csv_fields(split) for split in ("train", "validation")}, indent=2) + "\n")
    term_rows = []
    for (kind, split, term), info in sorted(coverage.items()):
        names = tuple(sorted(senses[(kind, split, term)]))
        term_rows.append({"aspect": kind, "split": split, "lemma": term, **info,
                          "other_split_candidates": " | ".join(word for word, shared in options(kind, split, term, names)),
                          "wordnet_candidates": candidate_columns(kind, split, term, names)["wordnet_candidates"],
                          "context_wordnet_candidates": candidate_columns(kind, split, term, names)["wordnet_candidates"],
                          "wordnet_synsets": " | ".join(names), "selected_wordnet_synsets": " | ".join(names),
                          "sense_scope": "Dictionary-only, same-POS WordNet candidate senses; no contextual certification"})
    write_csv(out / "final_term_regions.csv", list(term_rows[0]), term_rows)
    summary = {"statistics": statistics, "skipped_new_candidates": dict(skipped_candidates), "method": {
        "dictionary_only": True,
        "llm_used": False,
        "scope": "Rebuild lemmas, coverage and all candidates from observed valid source tokens, preserving original queries",
        "schema": "13 reference pair columns + 7 dictionary-limited WordNet columns + 4 region columns + 8 review columns + 5 dictionary provenance columns",
        "candidate_columns": CANDIDATE_FIELDS,
        "region_columns": REGION_FIELDS,
        "review_columns": REVIEW_FIELDS,
        "context_review_rule": "Disabled by default; reviews cannot add terms or alter the supplied aspect/POS",
        "coverage_senses_rule": "Coverage is counted per lemma using the union of valid occurrence-selected senses in that split; direct opposite lemmas are also covered; pair candidates use the source occurrence's selected senses and a matching opposite witness",
        "wordnet_candidates": "Exact supplied same-aspect dictionary surfaces whose normalized lemmas share a same-POS source synset; no unrestricted WordNet expansion",
        "other_split_candidates": "Exact supplied dictionary surfaces observed in the opposite split and sharing a same-POS synset, before same-tag pair filters",
        "empty_candidate_rule": "Candidate counts are 0 and lists blank; has_wordnet_entry blank means no supported source to look up, False means a lookup found no synset",
        "no_pair_rule": "Retain source/query with blank synonym/target/meaning fields and synonym_rank=0; reasons in no_pair_annotations.csv",
        "region_rule": "A=train UNCOVERED terms; C=validation UNCOVERED terms; B=DIRECT_LEMMA or WORDNET_CANDIDATE terms from either query region",
        "empty_rule": "One blank-source row for queries with no terms of the aspect; not a missing-synonym category",
        "unclassified_rule": "Reject punctuation-only or noncanonical lemmas, unresolved auxiliary contractions, truncated auxiliary 's', unsupported POS or unobserved tokens into UNCLASSIFIED files",
        "inflection_rule": "Select an existing exact supplied dictionary surface with the original Penn POS and target lemma; never generate new inflections or case variants",
        "normalization_rule": "Use the unique same-POS WordNet morphological root; with multiple roots use a unique root whose dictionary inflection matches the original surface/Penn POS; otherwise preserve the legacy spaCy lemma; retain the old value in original_source_lemma",
        "observed_source_rule": "Exact word/phrase boundary match OR exact spaCy tokenizer token; exclude punctuation-only lemmas from both split vocabularies",
        "query_membership_rule": "A query can contribute different source words to multiple region files; query_region preserves the previous whole-query label",
        "candidates": "Distinct same-POS WordNet synonyms observed in the opposite original split; exclude morphology/spelling variants",
        "B_rule": "Recompute every pair from the corrected lemmas and original opposite-split witnesses",
        "ranking": "Alphabetical inflected expressions; no top-k, context scoring or visual/temporal validation",
        "review": "All requires_review=True; rank 0 is a retained non-pair record, not a synonym",
        "train_caption_basis": extraction_summary["inputs"]["tvrtrain.caption.txt"],
        "validation_scope": "These dictionaries contain Train IDs only. Val is restricted to exact same-aspect surface/POS pairs in them; outside-dictionary Val novelty cannot be assessed in this scope.",
        "seed": 2026, "wordnet_version": wn.get_version(), "spacy_version": spacy.__version__,
        "spacy_model_version": nlp.meta["version"]},
        "inputs": {str(path.resolve()): sha256(path) for path in inputs}, "script_sha256": sha256(Path(__file__)),
        "implementation_sha256": {str((workspace / "rewriting" / name).resolve()): sha256(workspace / "rewriting" / name)
                                  for name in ["analyze_val_domain.py", "export_B_synonym_pairs.py", "build_object_action_regions.py", "suggest_val_replacements.py", "context_review.py"]}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text("""# 다이어그램 영역별 synonym-pairs 형식 CSV — 공식 사전만 사용

Object / Action 각각 A_train, B_train, B_val, C_val, EMPTY_train, EMPTY_val 파일로 나눕니다.
UNCLASSIFIED_train / UNCLASSIFIED_val도 같은 형식으로 보존합니다.
모든 영역은 기준 object_B_val.csv와 같은 37개 열을 사용합니다.
기존 13개 쌍 열 + WordNet 7개 열 + 영역 4개 열 + 검토 이력 8개 열 + 공식 사전 근거 5개 열입니다.
Val 파일은 대상 열을 train_expression / train_caption_id / train_caption으로 바꿉니다.
Train 파일의 대상 열은 val_expression / val_caption_id / val_caption이며, 나머지 열과 순서는 동일합니다.
file_counts.csv에 모든 파일의 행·후보·빈 후보·고유 기본형 수를 기록합니다.
csv_schema.json에 두 split의 열 이름과 순서를 기록합니다.

| 추가 열 | 의미 |
| --- | --- |
| source_lemma | 원본 표현을 해당 Object/Action 품사로 정규화한 조회 단어 |
| has_wordnet_entry | 같은 품사 WordNet synset 존재 여부. True/False, 조회할 단어 또는 지원 품사가 없으면 빈칸 |
| wordnet_candidate_count | 동일 aspect 공식 사전에 실제로 존재하는 WordNet 후보 표현 수 |
| wordnet_candidates | 공식 사전의 실제 표면형으로 제한한 후보 목록. ` | `로 구분 |
| other_split_candidate_count | 반대 split에서 관측된 WordNet 유의어 후보 수 |
| other_split_candidates | 반대 split에서 관측되고 같은 aspect 공식 사전에 존재하는 후보 표면형 |
| wordnet_synsets | 원본 단어를 해당 품사로 조회한 모든 synset ID. ` | `로 구분 |
| term_region | 이 행의 원본 단어 영역. CSV 파일의 영역과 일치 |
| term_coverage_status | DIRECT_LEMMA / WORDNET_CANDIDATE / UNCOVERED / EMPTY / UNCLASSIFIED |
| query_region | 원본 문장 전체를 분류했을 때의 영역. 문장에 미겹침 단어가 하나라도 있으면 A/C였던 기존 기준 |
| original_source_lemma | 교정 전 기존 spaCy 기본형. 현재 source_lemma와 비교해 기본형 변경을 확인 가능 |

같은 품사 WordNet 형태 분석 결과 기본형이 하나면 그 기본형을 사용합니다. miss→mis, access→acces처럼 잘못 잘린 기본형을 교정합니다.
기본형 후보가 둘 이상이면 원래 Penn POS로 굴절했을 때 원래 표현으로 돌아오는 후보가 하나인지 확인합니다.
예: sing→singing, singe→singeing이므로 singing/VBG는 sing으로 교정합니다.
이 방법으로도 하나를 정할 수 없거나 WordNet에 없으면 기존 기본형을 유지합니다. 이 경우 문맥에 따른 의미·기본형 검토가 필요할 수 있습니다.

추출과 후보의 유일한 허용 어휘는 tvr_id_object_dict.json / tvr_id_action_dict.json입니다.
Train은 caption ID의 원본 항목·품사를 그대로 보존합니다. 두 사전에는 Val ID가 없으므로
Val은 동일 aspect 사전에 존재하는 정확한 표면형·Penn POS 쌍만 선택합니다.
새 단어·새 굴절형·대소문자 변형을 추가하지 않으며 Object↔Action 이동도 허용하지 않습니다.
후보 목록은 원본 단어의 WordNet 의미와 연결되는 공식 사전 항목으로 제한합니다.
원본 자체·굴절만 다른 표현·철자 변형은 제외합니다. 반대 split 후보는 실제 관측 항목에 한정합니다.
synonym_expression도 원본 POS를 가진 공식 사전의 기존 표면형 중에서 선택하며 생성하지 않습니다.
반대 split 후보 목록은 굴절 등 실제 쌍 내보내기 필터 적용 전이므로 rank=0 행에도 목록이 있을 수 있습니다.
후보가 없으면 수는 0이고 목록은 빈칸입니다. 원본 단어가 없는 EMPTY 행은 조회 여부도 빈칸입니다.

한 행은 원본 표현 × 유의어 후보이며, 후보가 없는 표현도 빈 후보 행으로 보존합니다.
synonym_rank >= 1: 실제 반대 split 관측 근거가 있는 유의어 후보.
synonym_rank = 0: 유의어 쌍 없음. synonym_expression / meaning / 대상 표현·ID·캡션은 모두 빈칸.
EMPTY 쿼리는 source_expression도 빈칸인 한 행으로 보존합니다.
no_pair_annotations.csv의 csv_file + data_row_number(헤더 제외 1부터)로 빈 행의 사유를 찾을 수 있습니다.
query_membership.csv는 쿼리 × aspect × 출력 영역별로 실제 파일·단어·후보 수를 기록합니다.
한 쿼리 안의 단어가 서로 다른 영역에 속하면 해당 쿼리가 여러 행에 나타납니다.

A는 Train 단어 중 반대 split의 동일 lemma 및 같은 품사 WordNet synset 연결이 모두 없는 단어만 포함합니다.
C는 Val 단어 중 위 연결이 없는 단어만 포함합니다. A/C의 반대 split 후보 수는 모두 0이며 실제 유의어 쌍도 없습니다.
B는 반대 split과 동일 lemma 또는 같은 품사 synset으로 연결되는 단어입니다. 기존 A/C 문장의 겹치는 단어도 B로 이동합니다.
예: 기존 A 문장 안의 scuba는 A지만, chair/door/mask는 B입니다. door→doorway 행도 B입니다.
EMPTY는 해당 aspect의 추출 단어가 없는 쿼리만 포함합니다.
유효한 품사/lemma 분류가 없는 행, 문장에 실제 토큰으로 존재하지 않는 표현, 구두점만 남은 lemma는 UNCLASSIFIED에 보존합니다.
예: 's를 잘못 정규화한 ' 또는 명사로 잘못 추출된 -는 C가 아니라 검토 대상입니다. 이는 다이어그램 영역이 아닙니다.
기본형으로 해소되지 않은 're 등의 축약형과 잘린 보조동사 s도 UNCLASSIFIED에 보존합니다.
being/children처럼 아직 굴절형이 남아 기본형으로 확인되지 않는 lemma도 UNCLASSIFIED에 보존합니다. 해당 원본의 철자·품사를 먼저 검토해야 합니다.
didn't 안의 did처럼 spaCy가 실제로 분리한 토큰은 유효한 관측으로 인정합니다. 단순 문자열 일부가 존재하는 것만으로 인정하지 않습니다.
B도 동일 lemma로 커버되면서 다른 유의어가 없는 표현이 있습니다. 따라서 B에서도 rank=0 행이 있을 수 있습니다.
수정된 다이어그램의 A/B/C 수는 품사·split별 고유 source_lemma 수입니다. EMPTY 수는 쿼리 수입니다.
한 쿼리의 서로 다른 단어가 A/B 또는 B/C로 나뉠 수 있으므로 영역별 고유 caption_id 수는 서로 중복될 수 있습니다.

모든 실제 후보 행은 교정된 기본형과 원본 split의 실제 표현·근거 문장에서 다시 계산합니다.
유의어 쌍은 동일 품사 synset 기준에서 반대 split의 다른 표현을 찾습니다. 원본 자체·굴절·철자 변형은 후보에서 제외합니다.
후보 표현은 원본 Penn POS와 일치하는 공식 사전의 기존 표면형에서 선택합니다. 원본 쿼리 문장은 보존하고, 파일의 영역만 단어 기준으로 배정합니다.
후보가 공식 사전의 해당 aspect·POS에 없으면 내보내지 않습니다.
synonym_rank는 사전순 번호이며 문맥 적합도 점수가 아닙니다. 의미 설명은 공유 synset의 모든 뜻을 |로 연결합니다.
문맥·시각적 동작·구동사·문법·시간 구간 검증 전 후보입니다. 모든 requires_review=True입니다.
UTF-8 BOM CSV이며, 빈 후보는 추출 오류/WordNet 누락/상대 split 미등장/굴절 실패 등 여러 원인일 수 있습니다.

dictionary_file과 source_dictionary_caption_id는 원본 항목의 사전 근거입니다.
synonym_dictionary_caption_id는 synonym_expression의 사전 근거,
target_dictionary_caption_id와 target_source_pos는 반대 split 표현의 사전 근거입니다.
dictionary_terms.csv는 실제 사전 표면형·POS·기본형·근거 ID를 연결합니다.
source_lemma는 형태 정규화 값이며 새로운 추출 항목이 아닙니다.
final_term_regions.csv는 이 사전 제한 범위의 A/B/C 기본형 집계입니다.

두 사전은 Train 전용이므로 이 범위에서 C가 0이어도 Val 전체에 새로운 단어가 없다는 뜻은 아닙니다.
사전 밖 Val 단어는 추출하지 않았으므로 그 개수와 유의어 전환 가능성은 이 결과로 판단할 수 없습니다.
이전 자동 POS 확장 및 Object→Action 교정 결과와 그에 따른 집계는 이 결과로 대체합니다. LLM은 사용하지 않았습니다.

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/export_diagram_synonym_pairs.py
```
""", encoding="utf-8")
    diagram_counts = Counter({case: len(exported_lemmas[case]) for case in cases})
    pools = {kind: {split: set().union(*(exported_lemmas[case] for case in cases if case[:2] == (kind, split) and case[2] in {"A", "B", "C"}))
                    for split in ["train", "validation"]} for kind in allowed}
    totals = {kind: {split: len(terms) for split, terms in values.items()} for kind, values in pools.items()}
    effective_empty_counts = Counter({case: counts[(case, "queries")] for case in cases if case[2] == "EMPTY"})
    unclassified_counts = Counter({case: counts[(case, "rows")] for case in cases if case[2] == "UNCLASSIFIED"})
    plot_regions(out, diagram_counts, totals, pools, unit="lemma", empty_counts=effective_empty_counts,
                 unclassified_counts=unclassified_counts)
    print(json.dumps({"statistics": statistics, "output_dir": str(out)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
