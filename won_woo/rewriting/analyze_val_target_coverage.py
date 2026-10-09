"""Count Val vocabulary gaps and observed Train -> Val CSV candidates without an LLM."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from apply_val_synonyms import digest, write_csv
from select_diagram_synonyms import read_captions


def main():
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parent.parent
    base = root / "rewriting/val_domain_analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagram-dir", type=Path, default=root / "diagram_synonym_pairs")
    parser.add_argument("--output-dir", type=Path, default=base / "val_target_coverage_20261007")
    args = parser.parse_args()
    diagram, out = args.diagram_dir, args.output_dir
    metadata = json.loads((diagram / "summary.json").read_text())
    if not metadata["method"].get("dictionary_only"):
        raise ValueError("Coverage requires the regenerated dictionary-only diagram")
    out.mkdir(parents=True, exist_ok=True)
    catalog_path = diagram / "final_term_regions.csv"
    with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        catalog = list(csv.DictReader(handle))
    train_data, train = read_captions(root / "tvr/TextData/tvrtrain.caption.txt")
    val_data, val = read_captions(root / "tvr/TextData/tvrval.caption.txt")
    old_data, original = read_captions(Path(metadata["method"]["train_caption_basis"]["path"]))
    train_vocab = {kind: {r["lemma"] for r in catalog if r["aspect"] == kind and r["split"] == "train"}
                   for kind in ("object", "action")}
    val_terms = {(r["aspect"], r["lemma"]): r for r in catalog if r["split"] == "validation"}
    missing = {key for key in val_terms if key[1] not in train_vocab[key[0]]}
    observations, surfaces, val_ids = {}, defaultdict(set), defaultdict(set)
    for kind in train_vocab:
        for region in ("B", "C"):
            path = diagram / f"{kind}_{region}_val.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = kind, row["source_lemma"]
                    assert key in val_terms and val_terms[key]["term_region"] == region
                    observation = kind, row["caption_id"], row["source_expression"]
                    assert observation not in observations or observations[observation] == row["source_lemma"]
                    observations[observation] = row["source_lemma"]
                    surfaces[key].add(row["source_expression"])
                    val_ids[key].add(row["caption_id"])
    assert set(surfaces) == set(val_terms)
    token_path = "/home/viglab2/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
    tokenizer = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
    pair_count, available_count, length_count = Counter(), Counter(), Counter()
    pair_ids, available_ids, length_ids = defaultdict(set), defaultdict(set), defaultdict(set)
    source_lemmas, examples, patterns = defaultdict(set), {}, {}
    totals, all_candidates = Counter(), defaultdict(set)
    for kind in train_vocab:
        path = diagram / f"{kind}_B_train.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                totals[f"{kind}_rows"] += 1
                if not row["synonym_expression"]:
                    continue
                totals[f"{kind}_pairs"] += 1
                cid = row["caption_id"]
                assert row["caption"].strip() == original[cid].strip()
                target = observations[kind, row["val_caption_id"], row["val_expression"]]
                key = kind, target
                pair_count[key] += 1
                pair_ids[key].add(cid)
                source_lemmas[key].add(row["source_lemma"])
                all_candidates[kind].add(cid)
                if key not in missing:
                    continue
                source = row["source_expression"]
                if source not in patterns:
                    patterns[source] = re.compile(r"(?<!\w)" + re.escape(source) + r"(?!\w)")
                pattern = patterns[source]
                if not pattern.search(train[cid]):
                    continue
                available_count[key] += 1
                available_ids[key].add(cid)
                examples.setdefault(key, {"train_caption_id": cid, "train_source": source,
                    "candidate_val_form": row["synonym_expression"], "train_caption": train[cid],
                    "val_caption_id": row["val_caption_id"], "val_expression": row["val_expression"],
                    "val_caption": row["val_caption"], "meaning": row["meaning"]})
                if 14 <= len(train[cid].split()) <= 20:
                    candidate = pattern.sub(row["synonym_expression"], train[cid])
                    if 14 <= len(candidate.split()) <= 20 and len(tokenizer.encode(candidate)) < 30:
                        length_count[key] += 1
                        length_ids[key].add(cid)
        print(f"Read {path.name}: {totals[f'{kind}_pairs']:,} candidate rows", flush=True)
    # Literal raw-text absence is distinct from absence in the same-POS vocabulary.
    token_index = defaultdict(set)
    lower_train = {cid: cap.casefold() for cid, cap in train.items()}
    for cid, caption in lower_train.items():
        for word in set(re.findall(r"\w+", caption)):
            token_index[word].add(cid)
    report = []
    for key, info in val_terms.items():
        kind, lemma = key
        literal_ids = set()
        if key in missing:
            for form in surfaces[key] | {lemma}:
                words = re.findall(r"\w+", form.casefold())
                if not words:
                    continue
                pattern = re.compile(r"(?<!\w)" + re.escape(form.casefold()) + r"(?!\w)")
                literal_ids.update(cid for cid in token_index[words[0]] if pattern.search(lower_train[cid]))
        report.append({"aspect": kind, "val_lemma": lemma, "diagram_region": info["term_region"],
            "coverage_status": info["coverage_status"], "direct_train_lemma_present": key not in missing,
            "val_query_count": len(val_ids[key]), "val_surface_forms": " | ".join(sorted(surfaces[key])),
            "wordnet_entry_present": bool(info["wordnet_synsets"]),
            "train_synset_connected_lemmas": info["other_split_candidates"],
            "train_exported_candidate_source_lemmas": " | ".join(sorted(source_lemmas[key])),
            "train_candidate_rows": pair_count[key], "train_candidate_query_count": len(pair_ids[key]),
            "current_standalone_candidate_rows": available_count[key] if key in missing else "",
            "current_standalone_candidate_queries": len(available_ids[key]) if key in missing else "",
            "single_replacement_length_pass_rows": length_count[key] if key in missing else "",
            "single_replacement_length_pass_queries": len(length_ids[key]) if key in missing else "",
            "literal_train_query_count_for_missing_term": len(literal_ids) if key in missing else "",
            "example_candidate": json.dumps(examples.get(key, {}), ensure_ascii=False)})
    report.sort(key=lambda r: (r["direct_train_lemma_present"], -r["val_query_count"], r["aspect"], r["val_lemma"]))
    fields = list(report[0])
    write_csv(out / "val_term_coverage.csv", fields, report)
    missing_rows = [r for r in report if not r["direct_train_lemma_present"]]
    raw_absent = {(r["aspect"], r["val_lemma"]) for r in missing_rows
                  if r["literal_train_query_count_for_missing_term"] == 0}
    write_csv(out / "missing_val_terms.csv", fields, missing_rows)
    write_csv(out / "B_synonym_only_val_terms.csv", fields, [r for r in missing_rows if r["diagram_region"] == "B"])
    write_csv(out / "C_no_train_connection.csv", fields, [r for r in missing_rows if r["diagram_region"] == "C"])
    write_csv(out / "B_synonym_only_raw_absent_val_terms.csv", fields, [r for r in missing_rows
              if r["diagram_region"] == "B" and (r["aspect"], r["val_lemma"]) in raw_absent])
    summary = {"aspects": {}, "train_caption_count": len(train), "val_caption_count": len(val),
        "scanned": dict(totals), "dictionary_only": True,
        "train_vocabulary_basis": metadata["method"]["train_caption_basis"], "llm_used": False,
        "validation_scope": metadata["method"]["validation_scope"],
        "train_sha256": digest(train_data), "original_train_sha256": digest(old_data), "val_sha256": digest(val_data),
        "term_catalog_sha256": digest(catalog_path.read_bytes()), "script_sha256": digest(Path(__file__).read_bytes()),
        "tokenizer": token_path, "token_count_includes_special_tokens": True,
        "candidate_scope": "Existing supplied dictionary forms connected by shared same-POS WordNet synsets, not proven contextual synonyms. Length-only pass does not certify meaning, pronoun resolution, visibility or action preservation."}
    for kind in train_vocab:
        rows = [r for r in report if r["aspect"] == kind]
        gaps = {key for key in missing if key[0] == kind}
        synonyms = {key for key in gaps if val_terms[key]["term_region"] == "B"}
        c_terms = gaps - synonyms
        raw_b = synonyms & raw_absent
        raw_c = c_terms & raw_absent
        available_union = set().union(*(available_ids[key] for key in gaps))
        length_union = set().union(*(length_ids[key] for key in gaps))
        val_gap_union = set().union(*(val_ids[key] for key in gaps))
        part = {"train_A_terms": sum(r["aspect"] == kind and r["split"] == "train" and r["term_region"] == "A" for r in catalog),
            "train_B_terms": sum(r["aspect"] == kind and r["split"] == "train" and r["term_region"] == "B" for r in catalog),
            "val_terms": len(rows), "direct_train_lemma_present": len(rows) - len(gaps),
            "missing_train_lemmas": len(gaps), "B_synonym_only_terms": len(synonyms), "C_no_connection_terms": len(c_terms),
            "B_missing_terms_with_exported_train_to_val_pairs": sum(bool(pair_count[key]) for key in synonyms),
            "B_missing_terms_without_exported_train_to_val_pairs": sum(not pair_count[key] for key in synonyms),
            "B_missing_terms_with_current_matching_sources": sum(bool(available_count[key]) for key in synonyms),
            "B_missing_terms_with_single_replacement_length_pass": sum(bool(length_count[key]) for key in synonyms),
            "missing_target_candidate_rows_original": sum(pair_count[key] for key in gaps),
            "missing_target_candidate_rows_current": sum(available_count[key] for key in gaps),
            "missing_target_candidate_queries_current": len(available_union),
            "missing_target_length_only_candidate_rows": sum(length_count[key] for key in gaps),
            "missing_target_length_only_candidate_queries": len(length_union),
            "val_queries_containing_missing_terms": len(val_gap_union),
            "C_terms_without_wordnet_entry": sum(not val_terms[key]["wordnet_synsets"] for key in c_terms),
            "missing_terms_literally_absent_from_raw_train": sum(r["literal_train_query_count_for_missing_term"] == 0 for r in rows if not r["direct_train_lemma_present"]),
            "raw_absent_B_synonym_only_terms": len(raw_b),
            "raw_absent_C_no_connection_terms": len(raw_c),
            "raw_absent_B_candidate_rows_current": sum(available_count[key] for key in raw_b),
            "raw_absent_B_candidate_queries_current": len(set().union(*(available_ids[key] for key in raw_b))),
            "raw_absent_B_length_only_candidate_terms": sum(bool(length_count[key]) for key in raw_b),
            "raw_absent_B_length_only_candidate_rows": sum(length_count[key] for key in raw_b),
            "raw_absent_B_length_only_candidate_queries": len(set().union(*(length_ids[key] for key in raw_b))),
            "all_Val_targets_candidate_train_queries": len(all_candidates[kind]),
            "all_Val_targets_candidate_pairs": totals[f"{kind}_pairs"]}
        assert part["direct_train_lemma_present"] + part["B_synonym_only_terms"] + part["C_no_connection_terms"] == part["val_terms"]
        assert all(pair_count[key] == 0 for key in c_terms)
        summary["aspects"][kind] = part
    summary["deduplicated_object_action_missing_target_train_queries"] = len(set().union(*(available_ids[key] for key in missing)))
    summary["deduplicated_object_action_missing_target_length_only_train_queries"] = len(set().union(*(length_ids[key] for key in missing)))
    summary["deduplicated_val_queries_containing_missing_terms"] = len(set().union(*(val_ids[key] for key in missing)))
    summary["raw_absent_B_deduplicated_train_candidate_queries"] = len(set().union(*(available_ids[key] for key in raw_absent)))
    summary["raw_absent_B_deduplicated_train_length_only_candidate_queries"] = len(set().union(*(length_ids[key] for key in raw_absent)))
    assert digest((root / "tvr/TextData/tvrtrain.caption.txt").read_bytes()) == summary["train_sha256"]
    assert digest((root / "tvr/TextData/tvrval.caption.txt").read_bytes()) == summary["val_sha256"]
    expected = json.loads((diagram / "summary.json").read_text())["statistics"]
    for kind in train_vocab:
        reference = next(r for r in expected if r["aspect"] == kind and r["split"] == "train" and r["region"] == "B")
        assert totals[f"{kind}_rows"] == reference["rows"] and totals[f"{kind}_pairs"] == reference["pairs"]
    summary["verification"] = "Dictionary-only metadata, partition counts, observed Val lemmas, source CSV counts, C zero-pair check and caption hashes passed."
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 공식 사전 기준 Val 용어 조사", "",
             "tvr_id_object_dict.json / tvr_id_action_dict.json의 항목만 사용한 diagram을 조사했습니다. LLM은 사용하지 않았습니다.", "",
             "두 사전은 Train ID 87,174개만 포함하고 Val ID는 없습니다. Val은 동일 aspect 사전에 있는 정확한 표면형·POS 쌍만 추출했습니다.",
             "따라서 사전 밖 Val 신규 어휘의 개수와 치환 가능성은 이 결과로 평가할 수 없습니다. C=0을 Val 전체의 신규 어휘 0개로 해석하면 안 됩니다.", "",
             "Train 용어 집합은 공식 ID 사전과 연결된 원본 캡션 기준이며, 현재 수정된 Train 원문 전체의 재추출 결과가 아닙니다.", "",
             "| 종류 | Train A | Train B | Val 용어 | Train 기본형 없음 | Val C | 유의어 후보 쌍 | 후보가 있는 원본 Train 캡션 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, part in summary["aspects"].items():
        keys = ["train_A_terms", "train_B_terms", "val_terms", "missing_train_lemmas", "C_no_connection_terms",
                "all_Val_targets_candidate_pairs", "all_Val_targets_candidate_train_queries"]
        lines.append("| " + kind + " | " + " | ".join(f"{part[key]:,}" for key in keys) + " |")
    lines += ["", "WordNet 후보는 문맥·동작·시간 구간을 검증한 치환 가능 건수가 아닙니다. CSV의 requires_review=True를 유지합니다.",
              "이전 자동 POS 확장에 따른 596개 미포함 용어 / 95개 후보 용어 집계는 철회하고 이 사전 제한 결과로 대체합니다.", "",
              "val_term_coverage.csv는 Val 기본형별 관측 표면형·근거·후보 수를 담습니다. 사전 밖 항목은 추출하지 않았습니다.",
              "검증 근거는 diagram의 dictionary_scope_verification.json과 dictionary_terms.csv를 확인하세요.", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
