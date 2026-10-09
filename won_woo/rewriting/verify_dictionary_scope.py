"""Fail if a diagram source, synonym, candidate or witness lacks supplied-dictionary evidence."""

import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from analyze_val_domain import (ACTION_TAGS, dictionary_vocabulary, extract_dictionary_terms,
                                validate_dictionary_terms)
from build_object_action_regions import sha256
from context_review import reviewed_terms
from export_diagram_synonym_pairs import csv_fields


def self_check():
    source = {"train-id": {"walks": "VBZ"}}
    vocabulary = dictionary_vocabulary(source)
    doc = [SimpleNamespace(text=text, tag_=tag) for text, tag in
           [("walks", "VBZ"), ("walks", "NN"), ("walking", "VBG"), ("Walks", "VBZ"), ("jumps", "VBZ")]]
    assert extract_dictionary_terms(doc, source, vocabulary, "val-id", ACTION_TAGS) == {"walks": "VBZ"}
    assert extract_dictionary_terms(doc, source, vocabulary, "train-id", ACTION_TAGS) == source["train-id"]
    for split, cid, raw in [("validation", "val-id", {"walking": "VBG"}),
                            ("validation", "val-id", {"walks": "NN"}),
                            ("train", "train-id", {"jumps": "VBZ"})]:
        try:
            validate_dictionary_terms(raw, "action", split, cid, source, vocabulary)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-dictionary term or POS was accepted")
    review = {"caption": "walking", "source_pos": "NN", "decision": "MOVE",
              "corrected_aspect": "action", "corrected_pos": "VBG"}
    reviews = {("object", "train", "id"): {"walking": review}}
    assert reviewed_terms({}, "action", "train", "id", "walking", reviews) == {}
    try:
        reviewed_terms({"walking": "NN"}, "object", "train", "id", "walking", reviews)
    except ValueError:
        pass
    else:
        raise AssertionError("Aspect transfer was accepted")


def main():
    self_check()
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path,
                        default=root / "rewriting/val_domain_analysis/dictionary_only/diagram_synonym_pairs")
    args = parser.parse_args()
    metadata = json.loads((args.pairs_dir / "summary.json").read_text())
    assert metadata["method"]["dictionary_only"] and not metadata["method"]["llm_used"]
    paths = {kind: root / f"tvr/TextData/tvr_id_{kind}_dict.json" for kind in ("object", "action")}
    dictionaries = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
    vocabularies = {kind: dictionary_vocabulary(source) for kind, source in dictionaries.items()}
    lemmas = {}
    with (args.pairs_dir / "dictionary_terms.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            kind, surface, tag, cid = (row[key] for key in
                                      ("aspect", "surface", "source_pos", "dictionary_caption_id"))
            assert dictionaries[kind][cid][surface] == tag
            lemmas[kind, surface, tag] = row["lemma"]

    @lru_cache(None)
    def check_candidates(kind, value):
        for surface in value.split(" | ") if value else ():
            assert surface in vocabularies[kind], (kind, surface)

    def check_reference(kind, cid, surface, tag):
        assert dictionaries[kind].get(cid, {}).get(surface) == tag, (kind, cid, surface, tag)

    counts, files, restored = Counter(), {}, {}
    extraction = root / "rewriting/val_domain_analysis/dictionary_only"
    with (extraction / "tvrval_action_object_analysis.csv").open(encoding="utf-8-sig", newline="") as handle:
        validation_terms = {row["caption_id"]: {kind: json.loads(row[f"{kind}s_raw"]) for kind in dictionaries}
                            for row in csv.DictReader(handle)}
    for item in metadata["statistics"]:
        path = args.pairs_dir / item["csv_file"]
        kind, split = item["aspect"], item["split"]
        target = "val" if split == "train" else "train"
        rows = pairs = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            assert len(reader.fieldnames) == 37 and reader.fieldnames == csv_fields(split)
            for row in reader:
                rows += 1
                assert row["term_type"] == row["original_aspect"] == kind
                assert row["source_pos"] == row["original_source_pos"]
                assert row["term_region"] == item["region"]
                assert row["dictionary_file"] == str(paths[kind])
                surface, tag = row["source_expression"], row["source_pos"]
                observed = restored.setdefault((kind, split), {}).setdefault(row["caption_id"], {})
                if surface:
                    assert surface not in observed or observed[surface] == tag
                    observed[surface] = tag
                    check_reference(kind, row["source_dictionary_caption_id"], surface, tag)
                    if split == "train":
                        assert row["source_dictionary_caption_id"] == row["caption_id"]
                    assert row["source_lemma"] == lemmas.get((kind, surface, tag), "")
                    counts["dictionary_backed_source_rows"] += 1
                for field in ("wordnet_candidates", "other_split_candidates", "context_wordnet_candidates"):
                    check_candidates(kind, row[field])
                for prefix in ("wordnet", "other_split"):
                    assert int(row[f"{prefix}_candidate_count"]) == len(row[f"{prefix}_candidates"].split(" | ") if row[f"{prefix}_candidates"] else [])
                if int(row["synonym_rank"]) > 0:
                    pairs += 1
                    assert row["synonym_expression"] and item["region"] == "B"
                    check_reference(kind, row["synonym_dictionary_caption_id"], row["synonym_expression"], tag)
                    check_reference(kind, row["target_dictionary_caption_id"], row[f"{target}_expression"], row["target_source_pos"])
                    if split == "validation":
                        assert row["target_dictionary_caption_id"] == row[f"{target}_caption_id"]
                    assert lemmas[kind, row["synonym_expression"], tag] == lemmas[kind, row[f"{target}_expression"], row["target_source_pos"]]
                else:
                    assert not row["synonym_expression"] and not row[f"{target}_expression"]
        assert rows == item["rows"] and pairs == item["pairs"], (path, rows, pairs)
        files[path.name] = {"rows": rows, "pairs": pairs, "sha256": sha256(path)}
        counts["rows"] += rows
        counts["pairs"] += pairs
        print(f"Verified {path.name}: {rows:,} rows, {pairs:,} pairs", flush=True)
    for kind, source in dictionaries.items():
        assert restored[kind, "train"] == source, f"Missing supplied {kind} source entries"
        assert restored[kind, "validation"] == {cid: terms[kind] for cid, terms in validation_terms.items()}, f"Missing whitelisted Val {kind} entries"
    result = {"status": "PASS", "llm_used": False, "outside_dictionary_terms": 0,
              "csv_columns": 37, "all_original_dictionary_entries_preserved": True,
              "all_whitelisted_validation_entries_preserved": True,
              "aspect_transfers": 0, "generated_surface_forms": 0, "counts": dict(counts),
              "files": files, "dictionary_sha256": {kind: sha256(path) for kind, path in paths.items()}}
    (args.pairs_dir / "dictionary_scope_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
