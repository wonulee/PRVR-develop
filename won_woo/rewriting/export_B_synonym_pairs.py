"""Flatten B-region, opposite-split WordNet candidates into synonym-pair CSVs."""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, encoded, make_normalizer, dictionary_sources, validate_dictionary_terms
from build_object_action_regions import sha256
from suggest_val_replacements import read_csv


@lru_cache(None)
def expression_pattern(expression):
    return re.compile(r"(?<!\w)" + re.escape(expression) + r"(?!\w)")


def expression_observed(expression, caption, tokenizer):
    return bool(expression_pattern(expression).search(caption)) or any(
        token.text == expression for token in tokenizer(caption))


def lemma_issue(kind, surface, lemma, wordnet=None):
    if not any(c.isalnum() for c in lemma):
        return "NONLEXICAL_LEMMA"
    if surface.casefold().replace("’", "'") in {"'s", "'re", "'m", "'ve", "'d", "'ll"} and lemma not in {"be", "have", "will", "would"}:
        return "UNRESOLVED_CONTRACTION"
    if kind == "action" and lemma == "s":
        return "TRUNCATED_AUXILIARY"
    if wordnet is not None:
        pos = wordnet.NOUN if kind == "object" else wordnet.VERB
        roots = wordnet._morphy(lemma.replace(" ", "_"), pos)
        if roots and lemma.replace(" ", "_") not in roots:
            return "NONCANONICAL_LEMMA"
    return ""


def observed_representatives(extraction, normalize, tokenizer, wordnet=None):
    dictionaries, vocabularies = dictionary_sources(extraction)
    representatives, paths = {}, []
    allowed = {"object": OBJECT_TAGS, "action": ACTION_TAGS}
    for split, filename in [("train", "tvrtrain_domain_analysis.csv"), ("validation", "tvrval_action_object_analysis.csv")]:
        path = extraction / filename
        paths.append(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for kind in allowed:
                    raw = json.loads(row[f"{kind}s_raw"])
                    validate_dictionary_terms(raw, kind, split, row["caption_id"], dictionaries[kind], vocabularies[kind])
                    for surface, tag in raw.items():
                        if tag not in allowed[kind] or not expression_observed(surface, row["caption"], tokenizer):
                            continue
                        term = normalize(surface, tag)
                        if not lemma_issue(kind, surface, term, wordnet):
                            representatives.setdefault((kind, split, term), {"expression": surface,
                                "caption_id": row["caption_id"], "caption": row["caption"]})
        print(f"Indexed actual {split} expression occurrences", flush=True)
    return representatives, paths


def main():
    import spacy

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "rewriting/val_domain_analysis/dictionary_only/object_action_regions")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir or args.input_dir.parent / "B_synonym_pairs"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(workspace / "rewriting/nlp_resources/python"))
    from lemminflect import getInflection

    spacy.util.fix_random_seed(2026)
    nlp = spacy.load("en_core_web_md", exclude=["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "senter"])
    normalize = make_normalizer(nlp)
    assert normalize("walks", "VBZ") == "walk"
    assert getInflection("buy", "VBZ")[0] == "buys"
    assert getInflection("buy", "VBD")[0] == "bought"
    assert getInflection("beverage", "NNS")[0] == "beverages"
    assert expression_pattern("pick").search("They pick up a cup.")
    assert not expression_pattern("pick").search("They picked up a cup.")
    assert expression_observed("did", "They didn't leave.", nlp.make_doc)
    assert not expression_observed("-erase", "A dry-erase board.", nlp.make_doc)
    assert lemma_issue("action", "'re", "'re") == "UNRESOLVED_CONTRACTION"
    assert not lemma_issue("action", "'re", "be")

    metadata = json.loads((args.input_dir / "methodology.json").read_text())
    for info in metadata["inputs"].values():
        if sha256(Path(info["path"])) != info["sha256"]:
            raise ValueError(f"B-region input has changed: {info['path']}; rebuild B regions first")
    extraction = args.input_dir.parent
    dictionaries, vocabularies = dictionary_sources(extraction)
    input_paths = [args.input_dir / "methodology.json", workspace / "rewriting/val_domain_analysis/linguistic_synonyms/tvrtrain_synonym_pairs.csv"]
    allowed = {"object": OBJECT_TAGS, "action": ACTION_TAGS}
    representatives, source_paths = observed_representatives(extraction, normalize, nlp.make_doc)
    input_paths.extend(source_paths)
    catalogue = {}
    for kind in allowed:
        path = args.input_dir / f"{kind}_B_synonym_terms.csv"
        input_paths.append(path)
        for row in read_csv(path):
            meanings = {entry["synset"]: entry["meaning"] for entry in json.loads(row["wordnet_synonyms_by_sense"])}
            options = [{"lemma": evidence["other_lemma"],
                        "meaning": " | ".join(meanings[name] for name in evidence["shared_synsets"])}
                       for evidence in json.loads(row["other_split_evidence"])]
            assert [option["lemma"] for option in options] == json.loads(row["other_split_synonyms"])
            catalogue[(kind, row["split"], row["source_lemma"])] = options

    base_fields = ["caption_id", "video_id", "enc", "caption", "term_type", "source_expression",
                   "synonym_expression", "synonym_rank", "meaning"]
    with input_paths[1].open(encoding="utf-8-sig", newline="") as handle:
        reference_fields = next(csv.reader(handle))
    assert reference_fields == base_fields + ["val_expression", "val_caption_id", "val_caption", "requires_review"]
    coverage_fields = ["caption_id", "video_id", "enc", "split", "term_type", "caption", "pair_count",
                       "source_expressions_with_pairs", "source_expressions_without_pairs", "skipped_source_expressions"]
    counts, skips = Counter(), Counter()
    with ExitStack() as stack:
        writers = {}
        coverage_handle = stack.enter_context((out / "B_query_pair_status.csv").open("w", encoding="utf-8-sig", newline=""))
        coverage_writer = csv.DictWriter(coverage_handle, fieldnames=coverage_fields)
        coverage_writer.writeheader()
        for split, prefix, target_prefix in [("train", "tvrtrain", "val"), ("validation", "tvrval", "train")]:
            fields = base_fields + [f"{target_prefix}_expression", f"{target_prefix}_caption_id", f"{target_prefix}_caption", "requires_review"]
            for kind in ["all", "object", "action"]:
                filename = f"{prefix}_synonym_pairs.csv" if kind == "all" else f"{prefix}_{kind}_synonym_pairs.csv"
                handle = stack.enter_context((out / filename).open("w", encoding="utf-8-sig", newline=""))
                writers[(split, kind)] = csv.DictWriter(handle, fieldnames=fields)
                writers[(split, kind)].writeheader()
        for kind in allowed:
            path = args.input_dir / f"{kind}_B.csv"
            input_paths.append(path)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for query_index, row in enumerate(csv.DictReader(handle), 1):
                    split = row["split"]
                    other = "validation" if split == "train" else "train"
                    target_prefix = "val" if split == "train" else "train"
                    assert row["region"] == "B" and row["aspect"] == kind
                    terms = set(json.loads(row["direct_lemmas"]) + json.loads(row["wordnet_candidate_lemmas"]))
                    query_candidates = json.loads(row["other_split_synonym_candidates"])
                    with_pairs, without_pairs, skipped = [], [], []
                    query_pairs = 0
                    for surface, tag in json.loads(row["original_terms_raw"]).items():
                        if tag not in allowed[kind]:
                            skips["unsupported_source_pos"] += 1
                            skipped.append(surface)
                            continue
                        term = normalize(surface, tag)
                        if term not in terms or lemma_issue(kind, surface, term) or not expression_observed(surface, row["caption"], nlp.make_doc):
                            skips["source_not_observed_or_lemma_mismatch"] += 1
                            skipped.append(surface)
                            continue
                        options = catalogue[(kind, split, term)]
                        assert [entry["lemma"] for entry in options] == query_candidates[term]
                        alternatives = {}
                        for option in options:
                            target = option["lemma"]
                            witness = representatives.get((kind, other, target))
                            if not witness:
                                skips["no_actual_opposite_expression_witness"] += 1
                                continue
                            if " " in target:
                                skips["multiword_inflection_requires_head_review"] += 1
                                continue
                            forms = getInflection(target, tag)
                            if not forms:
                                skips["no_source_pos_inflection"] += 1
                                continue
                            expression = forms[0]
                            if surface[:1].isupper() and surface[1:].islower():
                                expression = expression[:1].upper() + expression[1:]
                            if tag not in vocabularies[kind].get(expression, {}):
                                skips["inflection_outside_supplied_dictionary"] += 1
                                continue
                            if expression.casefold() == surface.casefold():
                                skips["no_surface_change_after_inflection"] += 1
                                continue
                            alternatives.setdefault(expression, (option, witness))
                        if alternatives:
                            with_pairs.append(surface)
                        else:
                            without_pairs.append(surface)
                        for rank, (expression, (option, witness)) in enumerate(sorted(alternatives.items()), 1):
                            pair = {key: row[key] for key in ["caption_id", "video_id", "enc", "caption"]}
                            pair.update({"term_type": kind, "source_expression": surface, "synonym_expression": expression,
                                         "synonym_rank": rank, "meaning": option["meaning"],
                                         f"{target_prefix}_expression": witness["expression"],
                                         f"{target_prefix}_caption_id": witness["caption_id"],
                                         f"{target_prefix}_caption": witness["caption"], "requires_review": True})
                            writers[(split, kind)].writerow(pair)
                            writers[(split, "all")].writerow(pair)
                            query_pairs += 1
                            counts[(split, kind, "pairs")] += 1
                    counts[(split, kind, "B_queries")] += 1
                    counts[(split, kind, "queries_with_pairs")] += int(query_pairs > 0)
                    counts[(split, kind, "queries_without_pairs")] += int(query_pairs == 0)
                    coverage = {key: row[key] for key in ["caption_id", "video_id", "enc", "caption"]}
                    coverage.update({"split": split, "term_type": kind, "pair_count": query_pairs,
                                     "source_expressions_with_pairs": encoded(with_pairs),
                                     "source_expressions_without_pairs": encoded(without_pairs),
                                     "skipped_source_expressions": encoded(skipped)})
                    coverage_writer.writerow(coverage)
                    if query_index % 20000 == 0:
                        print(f"Exported {kind}: {query_index:,} B query/aspect rows", flush=True)
    statistics = []
    for split in ["train", "validation"]:
        for kind in allowed:
            entry = {"split": split, "term_type": kind,
                     **{key: counts[(split, kind, key)] for key in ["B_queries", "pairs", "queries_with_pairs", "queries_without_pairs"]}}
            assert entry["queries_with_pairs"] + entry["queries_without_pairs"] == entry["B_queries"]
            statistics.append(entry)
    summary = {"statistics": statistics, "skips": dict(skips), "method": {
        "scope": "Existing B-region queries, separately for objects and actions; opposite-split observed candidates only",
        "schema": "Train->Val matches tvrtrain_synonym_pairs.csv exactly; Val->Train uses train_ target column names",
        "source_expression": "Original raw expression, supported source POS and matching original lemma; literal token-boundary occurrence required",
        "synonym_expression": "Candidate lemma inflected to original Penn POS using lemminflect; original capitalization retained",
        "target_witness": "First original opposite-split caption with an actual expression matching candidate lemma and POS category",
        "ranking": "Alphabetical order of inflected candidate expressions; all candidates, no top-k or semantic score",
        "meaning": "All shared same-POS synset definitions, joined with |; no context-selected sense",
        "review": "Lexical candidates only; no contextual equivalence, visual grounding, phrasal-verb or temporal validation; requires_review=True",
        "difference_from_reference": "The reference applies contextual sense scoring and top-k; this export flattens existing B lexical candidates",
        "seed": 2026, "wordnet_version": metadata["wordnet_version"], "spacy_version": spacy.__version__,
        "spacy_model_version": nlp.meta["version"], "randomness": "No sampling; fixed seed; original query order, sorted alternatives"},
        "inputs": {str(path.resolve()): sha256(path) for path in input_paths}, "script_sha256": sha256(Path(__file__))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text("""# B 영역 유의어 pair CSV

한 행은 원본 B 쿼리의 원본 Object/Action 표현과 반대 split에서 관측된 유의어 후보 한 쌍입니다.

- tvrtrain_synonym_pairs.csv: Train B → Val. 기존 linguistic_synonyms/tvrtrain_synonym_pairs.csv와 동일한 13개 열.
- tvrtrain_object_synonym_pairs.csv / tvrtrain_action_synonym_pairs.csv: 같은 결과를 Object/Action별 분리.
- tvrval_synonym_pairs.csv: Val B → Train. 대상 열은 train_expression / train_caption_id / train_caption.
- tvrval_object_synonym_pairs.csv / tvrval_action_synonym_pairs.csv: Val 쪽 Object/Action별 분리.
- B_query_pair_status.csv: 모든 B 쿼리 × aspect의 pair 수, 후보가 있는/없는 원본 표현, 건너뛴 추출 표현.
- summary.json: 집계, 건너뛴 사유, 방법, 입력 해시, 버전.

원본 표현과 원문을 보존하며 후보 기본형은 원본 Penn POS에 맞게 굴절시킵니다.
대상 표현과 대상 캡션은 반대 split의 실제 관측 근거입니다. 반드시 그 캡션에서 같은 의미로 사용된다는 보장은 없습니다.
원본 자체·단순 굴절·철자 변형은 이전 B 후보 생성 단계에서 제외했습니다.
synonym_rank는 굴절된 표현의 사전순 번호이며 의미 적합도 순위가 아닙니다. Top-k 제한 없이 후보를 내보냅니다.
meaning에는 공유 synset의 뜻을 모두 |로 연결합니다. 문맥에 맞는 의미를 하나 선택하지 않았습니다.

기존 reference 파일은 문맥 의미 선택과 점수·top-k를 사용합니다. 이번 파일은 같은 열 구조로 기존 B lexical 후보를 펼친 결과입니다.
모든 requires_review=True입니다. 실제 치환 전 문맥·가시적 동작·목적어/구동사·문법·시간 구간을 검토해야 합니다.
원본에 없는 표현, 품사 불일치, 실제 반대 split 근거 없음, 굴절 실패 등은 제외하고 사유를 집계합니다.
Pair가 없는 B 쿼리는 현황 CSV에 남아 있습니다. 기존 원본 파일과 B 분류는 변경하지 않습니다.
CSV는 Excel 호환 UTF-8 BOM입니다.

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/export_B_synonym_pairs.py
```
""", encoding="utf-8")
    archive = out.parent / "B_synonym_pairs.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            if path.is_file():
                bundle.write(path, arcname=f"B_synonym_pairs/{path.name}")
        bundle.write(Path(__file__), arcname="B_synonym_pairs/export_B_synonym_pairs.py")
    print(json.dumps({"statistics": statistics, "skips": dict(skips), "archive": str(archive)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
