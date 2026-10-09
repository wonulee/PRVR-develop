"""Make small browsing CSVs from the verified 37-column exports without filtering terms."""

import argparse
import csv
import json
from pathlib import Path

from analyze_val_domain import write_csv


LIST_FIELDS = ["wordnet_candidates", "other_split_candidates", "wordnet_synsets"]
TERM_FIELDS = ["term_type", "split", "term_region", "source_lemma", "source_expressions",
               "source_pos", "source_occurrences", "source_caption_count",
               "source_occurrences_without_pairs", "full_csv_rows", "full_pair_rows",
               "term_coverage_status", "example_caption_id", "source_dictionary_caption_id",
               "requires_review", "has_wordnet_entry", "wordnet_candidate_count",
               "wordnet_candidates", "other_split_candidate_count", "other_split_candidates",
               "wordnet_synsets"]
PAIR_FIELDS = ["term_type", "split", "term_region", "source_lemma", "target_lemma",
               "source_expressions", "synonym_expressions", "target_expressions", "source_pos",
               "target_source_pos", "full_pair_rows", "meaning", "example_source_caption_id",
               "example_source_caption", "example_target_caption_id", "example_target_caption",
               "requires_review"]


def joined(values):
    return " | ".join(sorted(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent.parent /
                        "diagram_synonym_pairs_final_20261007")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base = args.package_dir
    out = args.output_dir or base / "compact"
    summary = json.loads((base / "reports/summary.json").read_text())
    proof = json.loads((base / "reports/dictionary_scope_verification.json").read_text())
    assert proof["status"] == "PASS" and summary["method"]["dictionary_only"]
    with (base / "reports/dictionary_terms.csv").open(encoding="utf-8-sig", newline="") as handle:
        lemmas = {(r["aspect"], r["surface"], r["source_pos"]): r["lemma"]
                  for r in csv.DictReader(handle)}
    (out / "terms").mkdir(parents=True, exist_ok=False)
    (out / "pairs").mkdir()
    reports = []
    for item in summary["statistics"]:
        kind, split, region, name = (item[k] for k in ["aspect", "split", "region", "csv_file"])
        target = "val" if split == "train" else "train"
        terms, pairs, seen = {}, {}, set()
        rows = pair_rows = 0
        with (base / "csv" / name).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                assert row["term_type"] == kind and row["term_region"] == region
                source = row["caption_id"], row["source_expression"], row["source_pos"]
                # Unsupported entries retain their surface/POS separately; EMPTY is one count row.
                key = (row["source_lemma"],) if region != "UNCLASSIFIED" else (
                    row["source_lemma"], row["source_expression"], row["source_pos"])
                if key not in terms:
                    terms[key] = {"lemma": row["source_lemma"], "expressions": set(), "pos": set(),
                                  "captions": set(), "occurrences": 0, "no_pairs": 0,
                                  "rows": 0, "pairs": 0, "status": set(), "entry": set(),
                                  "example": row["caption_id"],
                                  "dictionary_id": row["source_dictionary_caption_id"],
                                  **{field: set() for field in LIST_FIELDS}}
                term = terms[key]
                term["rows"] += 1
                is_pair = int(row["synonym_rank"]) > 0
                term["pairs"] += is_pair
                if source not in seen:
                    seen.add(source)
                    term["occurrences"] += 1
                    term["no_pairs"] += not is_pair
                    term["captions"].add(row["caption_id"])
                    if row["source_expression"]:
                        term["expressions"].add(row["source_expression"])
                    if row["source_pos"]:
                        term["pos"].add(row["source_pos"])
                    term["status"].add(row["term_coverage_status"])
                    term["entry"].add(row["has_wordnet_entry"])
                    for field in LIST_FIELDS:
                        term[field].update(row[field].split(" | ") if row[field] else ())
                if not is_pair:
                    continue
                pair_rows += 1
                target_lemma = lemmas[kind, row[f"{target}_expression"], row["target_source_pos"]]
                assert target_lemma == lemmas[kind, row["synonym_expression"], row["source_pos"]]
                pair_key = row["source_lemma"], target_lemma
                if pair_key not in pairs:
                    pairs[pair_key] = {"source": set(), "synonym": set(), "target": set(),
                                       "pos": set(), "target_pos": set(), "rows": 0, "meaning": set(),
                                       "example_source_caption_id": row["caption_id"],
                                       "example_source_caption": row["caption"],
                                       "example_target_caption_id": row[f"{target}_caption_id"],
                                       "example_target_caption": row[f"{target}_caption"]}
                pair = pairs[pair_key]
                pair["source"].add(row["source_expression"])
                pair["synonym"].add(row["synonym_expression"])
                pair["target"].add(row[f"{target}_expression"])
                pair["pos"].add(row["source_pos"])
                pair["target_pos"].add(row["target_source_pos"])
                pair["meaning"].add(row["meaning"])
                pair["rows"] += 1
        assert rows == item["rows"] and pair_rows == item["pairs"], name
        assert sum(t["rows"] for t in terms.values()) == rows
        assert sum(t["pairs"] for t in terms.values()) == pair_rows
        assert sum(t["no_pairs"] for t in terms.values()) == item["no_pair_rows"]
        assert sum(t["occurrences"] for t in terms.values()) == len(seen)
        if region in {"A", "B", "C"}:
            assert len(terms) == item["unique_source_lemmas"], name
        term_rows = []
        for _, term in sorted(terms.items()):
            term_rows.append({"term_type": kind, "split": split, "term_region": region,
                              "source_lemma": term["lemma"], "source_expressions": joined(term["expressions"]),
                              "source_pos": joined(term["pos"]), "source_occurrences": term["occurrences"],
                              "source_caption_count": len(term["captions"]),
                              "source_occurrences_without_pairs": term["no_pairs"],
                              "full_csv_rows": term["rows"], "full_pair_rows": term["pairs"],
                              "term_coverage_status": joined(term["status"]), "example_caption_id": term["example"],
                              "source_dictionary_caption_id": term["dictionary_id"], "requires_review": True,
                              "has_wordnet_entry": "True" if "True" in term["entry"] else (
                                  "False" if "False" in term["entry"] else ""),
                              "wordnet_candidate_count": len(term["wordnet_candidates"]),
                              "other_split_candidate_count": len(term["other_split_candidates"]),
                              **{field: joined(term[field]) for field in LIST_FIELDS}})
        term_path = out / "terms" / name
        write_csv(term_path, TERM_FIELDS, term_rows)
        pair_path = out / "pairs" / name if region == "B" else None
        if pair_path:
            write_csv(pair_path, PAIR_FIELDS, (
                {"term_type": kind, "split": split, "term_region": region,
                 "source_lemma": source_lemma, "target_lemma": target_lemma,
                 "source_expressions": joined(pair["source"]), "synonym_expressions": joined(pair["synonym"]),
                 "target_expressions": joined(pair["target"]), "source_pos": joined(pair["pos"]),
                 "target_source_pos": joined(pair["target_pos"]), "full_pair_rows": pair["rows"],
                 "meaning": " || ".join(sorted(pair["meaning"])), "requires_review": True,
                 **{field: pair[field] for field in PAIR_FIELDS if field.startswith("example_")}}
                for (source_lemma, target_lemma), pair in sorted(pairs.items())))
        assert sum(pair["rows"] for pair in pairs.values()) == pair_rows
        report = {"aspect": kind, "split": split, "region": region, "full_csv": name,
                  "full_rows": rows, "source_occurrences": len(seen),
                  "terms_csv": str(term_path.relative_to(out)), "term_rows": len(term_rows),
                  "pairs_csv": str(pair_path.relative_to(out)) if pair_path else "", "unique_pair_rows": len(pairs),
                  "full_bytes": (base / "csv" / name).stat().st_size,
                  "terms_bytes": term_path.stat().st_size, "pairs_bytes": pair_path.stat().st_size if pair_path else 0}
        reports.append(report)
        print(f"{name}: {rows:,} full rows -> {len(term_rows):,} term rows, {len(pairs):,} unique pairs", flush=True)
    write_csv(out / "file_counts.csv", list(reports[0]), reports)
    result = {"status": "PASS", "llm_used": False, "dictionary_scope_changed": False,
              "all_source_occurrences_accounted_for": True, "all_pair_rows_accounted_for": True,
              "wordnet_information_retained": True,
              "input_full_rows": sum(r["full_rows"] for r in reports),
              "source_occurrences": sum(r["source_occurrences"] for r in reports),
              "term_rows": sum(r["term_rows"] for r in reports),
              "unique_pair_rows": sum(r["unique_pair_rows"] for r in reports),
              "full_bytes": sum(r["full_bytes"] for r in reports),
              "compact_csv_bytes": sum(r["terms_bytes"] + r["pairs_bytes"] for r in reports), "files": reports}
    assert result["input_full_rows"] == proof["counts"]["rows"]
    # Independent raw entry total, including one placeholder for each empty caption/aspect.
    extraction = Path(summary["method"]["train_caption_basis"]["path"]).parents[1] / "dictionary_only"
    dictionaries = {kind: json.loads((Path(summary["method"]["train_caption_basis"]["path"]).parents[3] /
                                     f"tvr/TextData/tvr_id_{kind}_dict.json").read_text())
                    for kind in ["object", "action"]}
    expected = sum(max(1, len(raw)) for source in dictionaries.values() for raw in source.values())
    with (extraction / "tvrval_action_object_analysis.csv").open(encoding="utf-8-sig", newline="") as handle:
        expected += sum(max(1, len(json.loads(row[f"{kind}s_raw"])))
                        for row in csv.DictReader(handle) for kind in dictionaries)
    assert result["source_occurrences"] == expected
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
