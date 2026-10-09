"""Keep exact Train/Val expression, POS and caption links in small normalized CSVs."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_val_domain import write_csv


WORDNET_FIELDS = ["has_wordnet_entry", "wordnet_candidate_count", "wordnet_candidates",
                  "other_split_candidate_count", "other_split_candidates", "wordnet_synsets",
                  "selected_wordnet_synsets", "context_wordnet_candidates"]
MAIN_FIELDS = ["mapping_id", "train_expression", "val_expression", "train_pos", "val_pos",
               "train_lemma", "val_lemma", "replacement_expression", "term_type",
               "source_split", "term_region", "mapping_type", "source_coverage_status",
               "source_wordnet_id", "meaning_id", "full_rows", "source_caption_count",
               "example_train_caption_id", "example_val_caption_id", "requires_review"]
LINK_FIELDS = ["mapping_id", "train_caption_key", "val_caption_key", "source_data_row_number", "synonym_rank"]


def main():
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=workspace / "diagram_synonym_pairs_final_20261007")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base, out = args.package_dir, args.output_dir or args.package_dir / "mapping"
    summary = json.loads((base / "reports/summary.json").read_text())
    proof = json.loads((base / "reports/dictionary_scope_verification.json").read_text())
    assert proof["status"] == "PASS" and summary["method"]["dictionary_only"]
    with (base / "reports/dictionary_terms.csv").open(encoding="utf-8-sig", newline="") as handle:
        lemmas = {(r["aspect"], r["surface"], r["source_pos"]): r["lemma"]
                  for r in csv.DictReader(handle)}
    with (base / "reports/final_term_regions.csv").open(encoding="utf-8-sig", newline="") as handle:
        catalog = {(r["aspect"], r["split"], r["lemma"]): r for r in csv.DictReader(handle)}
    with (base / "reports/no_pair_annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
        invalid = {(r["aspect"], r["split"], r["caption_id"], r["source_expression"])
                   for r in csv.DictReader(handle) if r["region"] == "UNCLASSIFIED"}
    captions, observed, queries = {}, defaultdict(dict), set()
    with (base / "reports/query_membership.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            kind, split, cid = row["aspect"], row["split"], row["caption_id"]
            assert (split, cid) not in captions or captions[split, cid] == row["caption"]
            captions[split, cid] = row["caption"]
            if (kind, split, cid) in queries:
                continue
            queries.add((kind, split, cid))
            for surface, tag in json.loads(row["query_terms_raw"]).items():
                lemma = lemmas[kind, surface, tag]
                if (kind, split, cid, surface) not in invalid and (kind, split, lemma) in catalog:
                    observed[kind, split, lemma].setdefault((surface, tag), cid)
    caption_keys = {key: index for index, key in enumerate(sorted(captions), 1)}
    senses = defaultdict(set)
    for (kind, split, lemma), info in catalog.items():
        for name in info["wordnet_synsets"].split(" | ") if info["wordnet_synsets"] else ():
            senses[kind, split, name].add(lemma)

    def witness(kind, split, lemma, surface, tag):
        forms = observed[kind, split, lemma]
        expression, pos = min(forms, key=lambda pair: (pair != (surface, tag), pair[1] != tag, pair))
        return expression, pos, forms[expression, pos]

    (out / "main").mkdir(parents=True, exist_ok=False)
    (out / "links").mkdir()
    write_csv(out / "captions.csv", ["caption_key", "split", "caption_id", "caption"], (
        {"caption_key": caption_keys[split, cid], "split": split, "caption_id": cid, "caption": text}
        for (split, cid), text in sorted(captions.items())))
    wordnets, meanings, reports = {}, {}, []
    next_mapping = 1
    for item in summary["statistics"]:
        kind, split, region, filename = (item[k] for k in ["aspect", "split", "region", "csv_file"])
        other = "validation" if split == "train" else "train"
        target_prefix = "val" if split == "train" else "train"
        groups, main_rows, caption_sets, counts = {}, {}, defaultdict(set), Counter()
        link_path = out / "links" / filename
        with link_path.open("w", encoding="utf-8-sig", newline="") as target_handle:
            writer = csv.DictWriter(target_handle, LINK_FIELDS)
            writer.writeheader()
            with (base / "csv" / filename).open(encoding="utf-8-sig", newline="") as source_handle:
                for row_number, row in enumerate(csv.DictReader(source_handle), 1):
                    assert row["term_type"] == kind and row["term_region"] == region
                    source = row["source_expression"], row["source_pos"], row["caption_id"], row["source_lemma"]
                    counterpart = "", "", "", ""
                    rank = int(row["synonym_rank"])
                    if rank:
                        expression, tag, cid = row[f"{target_prefix}_expression"], row["target_source_pos"], row[f"{target_prefix}_caption_id"]
                        lemma = lemmas[kind, expression, tag]
                        assert lemma == lemmas[kind, row["synonym_expression"], row["source_pos"]]
                        counterpart = expression, tag, cid, lemma
                        mapping_type = "WORDNET_SYNONYM"
                    elif region == "B" and row["term_coverage_status"] == "DIRECT_LEMMA":
                        expression, tag, cid = witness(kind, other, row["source_lemma"], row["source_expression"], row["source_pos"])
                        counterpart = expression, tag, cid, row["source_lemma"]
                        mapping_type = "DIRECT_LEMMA_REFERENCE"
                    elif region == "B":
                        targets = {lemma for name in row["selected_wordnet_synsets"].split(" | ") if name
                                   for lemma in senses[kind, other, name]}
                        assert targets, (filename, row_number)
                        lemma = min(targets)
                        expression, tag, cid = witness(kind, other, lemma, row["source_expression"], row["source_pos"])
                        counterpart = expression, tag, cid, lemma
                        mapping_type = "WORDNET_COVERAGE_REFERENCE"
                    else:
                        mapping_type = "NO_MAPPING" if region in {"A", "C"} else region
                    train, val = (source, counterpart) if split == "train" else (counterpart, source)
                    for actual_split, entry in [("train", train), ("validation", val)]:
                        if entry[0]:
                            assert observed[kind, actual_split, entry[3]].get(entry[:2]) is not None or region == "UNCLASSIFIED"
                            assert lemmas[kind, entry[0], entry[1]] == entry[3]
                        if entry[2]:
                            assert (actual_split, entry[2]) in caption_keys
                    wn_key = (kind, split, row["source_lemma"], *(row[f] for f in WORDNET_FIELDS))
                    wordnet_id = wordnets.setdefault(wn_key, len(wordnets) + 1)
                    meaning_id = meanings.setdefault(row["meaning"], len(meanings) + 1) if row["meaning"] else ""
                    key = (mapping_type, train[:2] + (train[3],), val[:2] + (val[3],),
                           row["synonym_expression"], wordnet_id, meaning_id, row["term_coverage_status"])
                    if key not in groups:
                        mapping_id = next_mapping
                        next_mapping += 1
                        groups[key] = mapping_id
                        main_rows[mapping_id] = {"mapping_id": mapping_id, "term_type": kind,
                            "source_split": "train" if split == "train" else "val", "term_region": region,
                            "mapping_type": mapping_type, "source_coverage_status": row["term_coverage_status"],
                            "train_expression": train[0], "train_pos": train[1], "train_lemma": train[3],
                            "val_expression": val[0], "val_pos": val[1], "val_lemma": val[3],
                            "replacement_expression": row["synonym_expression"], "source_wordnet_id": wordnet_id,
                            "meaning_id": meaning_id, "full_rows": 0, "source_caption_count": 0,
                            "example_train_caption_id": train[2], "example_val_caption_id": val[2], "requires_review": True}
                    mapping_id = groups[key]
                    main_rows[mapping_id]["full_rows"] += 1
                    caption_sets[mapping_id].add(source[2])
                    writer.writerow({"mapping_id": mapping_id,
                        "train_caption_key": caption_keys.get(("train", train[2]), ""),
                        "val_caption_key": caption_keys.get(("validation", val[2]), ""),
                        "source_data_row_number": row_number, "synonym_rank": rank})
                    counts["rows"] += 1
                    counts["pairs"] += bool(rank)
                    counts["reference_rows"] += mapping_type.endswith("_REFERENCE")
        assert counts["rows"] == item["rows"] and counts["pairs"] == item["pairs"]
        for mapping_id, row in main_rows.items():
            row["source_caption_count"] = len(caption_sets[mapping_id])
        assert sum(row["full_rows"] for row in main_rows.values()) == item["rows"]
        main_path = out / "main" / filename
        write_csv(main_path, MAIN_FIELDS, main_rows.values())
        reports.append({"aspect": kind, "split": split, "region": region, "csv_file": filename,
                        "mapping_rows": len(main_rows), "full_rows": counts["rows"], "pairs": counts["pairs"],
                        "reference_rows": counts["reference_rows"], "main_bytes": main_path.stat().st_size,
                        "links_bytes": link_path.stat().st_size})
        print(f"{filename}: {len(main_rows):,} exact mappings, {counts['rows']:,} caption links", flush=True)
    write_csv(out / "wordnet.csv", ["source_wordnet_id", "term_type", "split", "source_lemma", *WORDNET_FIELDS], (
        {"source_wordnet_id": index, "term_type": key[0], "split": key[1], "source_lemma": key[2],
         **dict(zip(WORDNET_FIELDS, key[3:]))} for key, index in wordnets.items()))
    write_csv(out / "meanings.csv", ["meaning_id", "meaning"], (
        {"meaning_id": index, "meaning": text} for text, index in meanings.items()))
    write_csv(out / "file_counts.csv", list(reports[0]), reports)
    result = {"status": "PASS", "llm_used": False, "dictionary_scope_changed": False,
              "exact_expression_pos_pairing_preserved": True, "all_original_caption_links_preserved": True,
              "original_pair_rows": sum(r["pairs"] for r in reports),
              "original_rows": sum(r["full_rows"] for r in reports),
              "mapping_rows": sum(r["mapping_rows"] for r in reports), "captions": len(captions),
              "wordnet_records": len(wordnets), "meaning_records": len(meanings),
              "main_csv_bytes": sum(r["main_bytes"] for r in reports),
              "all_csv_bytes": sum(p.stat().st_size for p in out.rglob("*.csv")),
              "reference_rule": "Rank-0 B rows add one attested opposite witness for direct-lemma/shared-synset coverage; replacement is blank. Positive-rank pairs preserve exact original witnesses.",
              "files": reports}
    assert result["original_rows"] == proof["counts"]["rows"]
    assert result["original_pair_rows"] == proof["counts"]["pairs"]
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
