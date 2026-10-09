"""Count and plot the existing exact-expression/POS mapping files by A/B/C region."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analyze_val_domain import write_csv
from build_object_action_regions import plot_regions, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent.parent /
                        "diagram_synonym_pairs_final_20261007")
    parser.add_argument("--diagram-dir", type=Path)
    args = parser.parse_args()
    base = args.package_dir
    mapping = base / "mapping"
    out = args.diagram_dir or base / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    proof = json.loads((mapping / "verification.json").read_text())
    roundtrip = json.loads((mapping / "roundtrip_verification.json").read_text())
    assert proof["status"] == roundtrip["status"] == "PASS"
    counts, empty, invalid = Counter(), Counter(), Counter()
    pools = {kind: {split: set() for split in ["train", "validation"]}
             for kind in ["object", "action"]}
    source_sets = {key: set() for kind in pools for key in [(kind, "train"), (kind, "validation")]}
    reports, fingerprints = [], {}
    global_ids = set()
    for item in proof["files"]:
        kind, split, region, filename = (item[k] for k in ["aspect", "split", "region", "csv_file"])
        side = "train" if split == "train" else "val"
        path = mapping / "main" / filename
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        identifiers = {row["mapping_id"] for row in rows}
        expressions = {(row[side + "_expression"], row[side + "_pos"]) for row in rows if row[side + "_expression"]}
        assert len(rows) == len(identifiers) == item["mapping_rows"]
        assert not (identifiers & global_ids)
        global_ids.update(identifiers)
        assert sum(int(r["full_rows"]) for r in rows) == item["full_rows"]
        assert all(r["term_type"] == kind and r["term_region"] == region for r in rows)
        for row in rows:
            opposite = "val" if side == "train" else "train"
            if region in {"A", "C"}:
                assert row["mapping_type"] == "NO_MAPPING" and not row[opposite + "_expression"]
            elif region == "B":
                assert row[side + "_expression"] and row[opposite + "_expression"]
        counts[kind, split, region] = len(rows)
        if region in {"A", "B", "C"}:
            pools[kind][split].update(identifiers)
            source_sets[kind, split].update(expressions)
        elif region == "EMPTY":
            empty[kind, split, region] = item["full_rows"]
        elif region == "UNCLASSIFIED":
            invalid[kind, split, region] = item["full_rows"]
        reports.append({"aspect": kind, "split": split, "region": region,
                        "mapping_rows": len(rows), "source_expression_pos_count": len(expressions),
                        "source_lemma_count": len({r[side + "_lemma"] for r in rows if r[side + "_lemma"]}),
                        "diagram_count": len(rows) if region in {"A", "B", "C"} else item["full_rows"],
                        "diagram_count_unit": "mapping rows" if region in {"A", "B", "C"} else
                            "queries" if region == "EMPTY" else "source occurrences",
                        "original_rows": item["full_rows"], "original_synonym_pair_rows": item["pairs"],
                        "mapping_csv": f"mapping/main/{filename}", "links_csv": f"mapping/links/{filename}"})
        fingerprints[filename] = sha256(path)
    totals = {kind: {split: len(ids) for split, ids in values.items()} for kind, values in pools.items()}
    assert sum(counts.values()) == proof["mapping_rows"]
    assert sum(r["original_rows"] for r in reports) == proof["original_rows"]
    write_csv(base / "reports/ABC_mapping_counts.csv", list(reports[0]), reports)
    plot_regions(out, counts, totals, pools, unit="mapping", empty_counts=empty, unclassified_counts=invalid)
    result = {"status": "PASS", "count_unit": "deduplicated directed expression/POS mapping row",
              "coverage_rule": "Identical normalized lemma or shared same-aspect WordNet synset; unchanged",
              "dictionary_scope_changed": False, "llm_used": False, "mapping_rows": proof["mapping_rows"],
              "original_rows": proof["original_rows"], "original_pair_rows": proof["original_pair_rows"],
              "diagram_totals": totals,
              "distinct_source_expression_pos_totals": {kind: {split: len(source_sets[kind, split]) for split in pools[kind]} for kind in pools},
              "files": reports, "mapping_main_sha256": fingerprints,
              "script_sha256": sha256(Path(__file__)),
              "plot_script_sha256": sha256(Path(__file__).with_name("build_object_action_regions.py")),
              "diagram_sha256": {ext: sha256(out / f"object_action_regions.{ext}") for ext in ["png", "svg", "pdf"]}}
    (base / "reports/ABC_mapping_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"files", "mapping_main_sha256", "diagram_sha256"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
