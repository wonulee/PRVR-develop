"""Reconstruct every original Train/Val caption link and exact term pairing from the small tables."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from export_mapping_diagram_csv import MAIN_FIELDS, LINK_FIELDS, WORDNET_FIELDS


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent.parent /
                        "diagram_synonym_pairs_final_20261007")
    parser.add_argument("--mapping-dir", type=Path)
    args = parser.parse_args()
    base, out = args.package_dir, args.mapping_dir or args.package_dir / "mapping"
    summary = json.loads((base / "reports/summary.json").read_text())
    captions = {r["caption_key"]: r for r in read_rows(out / "captions.csv")}
    assert len({(r["split"], r["caption_id"]) for r in captions.values()}) == len(captions)
    annotations = {}
    for row in read_rows(base / "reports/query_membership.csv"):
        annotations[row["aspect"], row["split"], row["caption_id"]] = json.loads(row["query_terms_raw"])
    lemmas = {(r["aspect"], r["surface"], r["source_pos"]): r["lemma"]
              for r in read_rows(base / "reports/dictionary_terms.csv")}
    catalog = {(r["aspect"], r["split"], r["lemma"]): r
               for r in read_rows(base / "reports/final_term_regions.csv")}
    wordnets = {r["source_wordnet_id"]: r for r in read_rows(out / "wordnet.csv")}
    meanings = {r["meaning_id"]: r["meaning"] for r in read_rows(out / "meanings.csv")}
    counts, global_ids = Counter(), set()
    for item in summary["statistics"]:
        kind, split, region, name = (item[k] for k in ["aspect", "split", "region", "csv_file"])
        main = {r["mapping_id"]: r for r in read_rows(out / "main" / name)}
        assert not (main.keys() & global_ids)
        global_ids.update(main)
        frequencies, source_ids = Counter(), defaultdict(set)
        row_count = pair_count = 0
        with (base / "csv" / name).open(encoding="utf-8-sig", newline="") as original_handle, (
                out / "links" / name).open(encoding="utf-8-sig", newline="") as links_handle:
            original_reader, links_reader = csv.DictReader(original_handle), csv.DictReader(links_handle)
            assert links_reader.fieldnames == LINK_FIELDS
            for number, (original, link) in enumerate(zip_longest(original_reader, links_reader), 1):
                assert original is not None and link is not None, name
                row = main[link["mapping_id"]]
                assert row["term_type"] == kind and row["term_region"] == region
                assert row["source_split"] == ("train" if split == "train" else "val")
                assert int(link["source_data_row_number"]) == number
                assert link["synonym_rank"] == original["synonym_rank"]
                assert row["replacement_expression"] == original["synonym_expression"]
                assert row["source_coverage_status"] == original["term_coverage_status"]
                assert meanings.get(row["meaning_id"], "") == original["meaning"]
                wn = wordnets[row["source_wordnet_id"]]
                assert wn["term_type"] == kind and wn["split"] == split
                assert wn["source_lemma"] == original["source_lemma"]
                assert all(wn[field] == original[field] for field in WORDNET_FIELDS)
                entries = {}
                for prefix, actual_split in [("train", "train"), ("val", "validation")]:
                    caption = captions.get(link[prefix + "_caption_key"])
                    expression, tag, lemma = (row[prefix + suffix] for suffix in ["_expression", "_pos", "_lemma"])
                    if caption:
                        assert caption["split"] == actual_split
                        if expression:
                            assert annotations[kind, actual_split, caption["caption_id"]].get(expression) == tag
                            assert lemmas[kind, expression, tag] == lemma
                    else:
                        assert not expression and not tag and not lemma
                    entries[prefix] = caption
                source_prefix, target_prefix = ("train", "val") if split == "train" else ("val", "train")
                source = entries[source_prefix]
                assert source["caption_id"] == original["caption_id"] and source["caption"] == original["caption"]
                assert row[source_prefix + "_expression"] == original["source_expression"]
                assert row[source_prefix + "_pos"] == original["source_pos"]
                assert row[source_prefix + "_lemma"] == original["source_lemma"]
                if int(original["synonym_rank"]):
                    target = entries[target_prefix]
                    assert row["mapping_type"] == "WORDNET_SYNONYM"
                    assert target["caption_id"] == original[target_prefix + "_caption_id"]
                    assert target["caption"] == original[target_prefix + "_caption"]
                    assert row[target_prefix + "_expression"] == original[target_prefix + "_expression"]
                    assert row[target_prefix + "_pos"] == original["target_source_pos"]
                    pair_count += 1
                elif row["mapping_type"] == "DIRECT_LEMMA_REFERENCE":
                    assert region == "B" and row["train_lemma"] == row["val_lemma"]
                elif row["mapping_type"] == "WORDNET_COVERAGE_REFERENCE":
                    other = "validation" if split == "train" else "train"
                    target_names = catalog[kind, other, row[target_prefix + "_lemma"]]["wordnet_synsets"]
                    assert set(original["selected_wordnet_synsets"].split(" | ")) & set(target_names.split(" | ")) - {""}
                else:
                    assert entries[target_prefix] is None
                frequencies[link["mapping_id"]] += 1
                source_ids[link["mapping_id"]].add(original["caption_id"])
                row_count += 1
        assert row_count == item["rows"] and pair_count == item["pairs"]
        for identifier, row in main.items():
            assert int(row["full_rows"]) == frequencies[identifier]
            assert int(row["source_caption_count"]) == len(source_ids[identifier])
        counts["original_rows"] += row_count
        counts["original_pair_rows"] += pair_count
        counts["mapping_rows"] += len(main)
        print(f"Verified {name}: {row_count:,} exact caption links", flush=True)
    result = {"status": "PASS", "all_original_expression_pos_pairs_match": True,
              "all_original_source_and_target_captions_match": True, "all_wordnet_fields_match": True,
              "all_reference_witnesses_attested": True, "counts": dict(counts)}
    (out / "roundtrip_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
