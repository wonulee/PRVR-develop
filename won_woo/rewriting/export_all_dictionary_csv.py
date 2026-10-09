"""Export all dictionary entries, including empty captions and unclassified terms."""

import csv
import json
import hashlib
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, dictionary_sources, write_csv
from build_object_action_regions import distinct_expression
from export_B_synonym_pairs import lemma_issue
from suggest_val_replacements import load_wordnet


WORDNET_FIELDS = ["has_wordnet_entry", "wordnet_candidate_count", "wordnet_candidates",
                  "other_split_candidate_count", "other_split_candidates", "wordnet_synsets",
                  "selected_wordnet_synsets", "context_wordnet_candidates", "wordnet_meanings",
                  "term_coverage_status", "requires_review"]


def captions(path):
    result = {}
    for line in path.read_text().splitlines():
        cid, caption = line.split(maxsplit=1)
        assert cid not in result
        result[cid] = caption
    return result


def make_wordnet_columns(root, evidence, catalog, observed_forms):
    """Lookup WordNet for every supplied term; candidate surfaces stay dictionary-backed."""
    wn = load_wordnet(root / "rewriting/nlp_resources")
    allowed = {"object": OBJECT_TAGS, "action": ACTION_TAGS}
    dictionary_forms, dictionary_senses, opposite_senses = {}, {}, {}
    for (kind, surface, tag), ref in evidence.items():
        lemma = ref["lemma"]
        if tag not in allowed[kind] or not lemma or lemma_issue(kind, surface, lemma, wn):
            continue
        dictionary_forms.setdefault((kind, lemma), set()).add(surface)
    for kind, lemma in dictionary_forms:
        pos = wn.NOUN if kind == "object" else wn.VERB
        for sense in wn.synsets(lemma.replace(" ", "_"), pos=pos):
            dictionary_senses.setdefault((kind, sense.name()), set()).add(lemma)
    for (kind, split, lemma), info in catalog.items():
        for name in info["wordnet_synsets"].split(" | ") if info["wordnet_synsets"] else ():
            opposite_senses.setdefault((kind, split, name), set()).add(lemma)

    @lru_cache(None)
    def columns(kind, split, lemma, tag):
        if not lemma or tag not in allowed[kind]:
            return dict(zip(WORDNET_FIELDS, ["", 0, "", 0, "", "", "", "", "", "", False]))
        pos = wn.NOUN if kind == "object" else wn.VERB
        senses = wn.synsets(lemma.replace(" ", "_"), pos=pos)
        names = sorted({sense.name() for sense in senses})
        words = sorted({surface for name in names for target in dictionary_senses.get((kind, name), ())
                        if distinct_expression(wn, lemma, target, pos)
                        for surface in dictionary_forms[kind, target]})
        other = "validation" if split == "train" else "train"
        opposite = sorted({surface for name in names for target in opposite_senses.get((kind, other, name), ())
                           if distinct_expression(wn, lemma, target, pos)
                           for surface in observed_forms.get((kind, other, target), ())})
        info = catalog.get((kind, split, lemma), {})
        return {"has_wordnet_entry": bool(names), "wordnet_candidate_count": len(words),
                "wordnet_candidates": " | ".join(words), "other_split_candidate_count": len(opposite),
                "other_split_candidates": " | ".join(opposite), "wordnet_synsets": " | ".join(names),
                "selected_wordnet_synsets": " | ".join(names), "context_wordnet_candidates": " | ".join(words),
                "wordnet_meanings": json.dumps([{ "synset": sense.name(), "meaning": sense.definition() }
                                               for sense in senses], ensure_ascii=False),
                "term_coverage_status": info.get("coverage_status", "UNCLASSIFIED"), "requires_review": True}
    return columns


def partition_by_region(out):
    """Partition every full source row once, preserving all columns and values."""
    destination = out / "ABC"
    destination.mkdir(exist_ok=True)
    cases = [("A", "Train"), ("B", "Train"), ("B", "Val"), ("C", "Val")]
    cases += [(region, split) for region in ("EMPTY", "UNCLASSIFIED") for split in ("Train", "Val")]
    reports = []
    for kind in ("object", "action"):
        source = out / f"all_{kind}s.csv"
        with ExitStack() as stack:
            handle = stack.enter_context(source.open(encoding="utf-8-sig", newline=""))
            reader = csv.reader(handle, strict=True)
            header = next(reader)
            split_index, region_index = header.index("출처"), header.index("영역")
            lemma_index = header.index("기본형_정규화")
            writers, paths, counts, digests, lemmas = {}, {}, Counter(), {}, {}
            for case in cases:
                region, split = case
                paths[case] = destination / f"{kind}_{region}_{split.lower()}.csv"
                target = stack.enter_context(paths[case].open("w", encoding="utf-8-sig", newline=""))
                writers[case] = csv.writer(target)
                writers[case].writerow(header)
                digests[case], lemmas[case] = hashlib.sha256(), set()
            source_rows = 0
            for row in reader:
                assert len(row) == len(header)
                case = row[region_index], row[split_index]
                assert case in writers, (kind, case)
                writers[case].writerow(row)
                digests[case].update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
                counts[case] += 1
                source_rows += 1
                if row[lemma_index]:
                    lemmas[case].add(row[lemma_index])
        assert sum(counts.values()) == source_rows
        for case in cases:
            actual, digest = 0, hashlib.sha256()
            with paths[case].open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                assert next(reader) == header
                for row in reader:
                    assert len(row) == len(header)
                    assert (row[region_index], row[split_index]) == case
                    digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
                    actual += 1
            assert actual == counts[case] and digest.digest() == digests[case].digest()
            reports.append({"유형": kind.title(), "영역": case[0], "출처": case[1], "전체항목_행수": actual,
                            "고유기본형수": len(lemmas[case]), "CSV파일": paths[case].name})
        print(f"{kind}: {source_rows:,} full rows partitioned and verified", flush=True)
    write_csv(destination / "file_counts.csv", list(reports[0]), reports)
    (destination / "verification.json").write_text(json.dumps({
        "status": "PASS", "llm_used": False, "all_source_rows_preserved": True,
        "all_columns_and_values_preserved": True, "order_preserved_within_each_file": True,
        "files": reports}, ensure_ascii=False, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# A/B/C별 전체 항목 CSV\n\n"
        "Object와 Action 각각 A_Train, B_Train, B_Val, C_Val로 나눴습니다. "
        "모든 원본 항목을 포함하며 유의어 후보 유무로 필터링하지 않았습니다.\n\n"
        "WordNet 등록 여부·synset·의미·공식 사전 내 후보·반대 split 후보·검토 여부 열을 포함했습니다. "
        "후보는 공식 Object/Action 사전에 있는 표현만 담습니다. 같은 품사 WordNet 의미 전체를 조회하며 문맥 검증 전 정보입니다.\n\n"
        "EMPTY와 UNCLASSIFIED는 별도 파일에 보존했습니다. C_Val은 현재 사전 제한 범위에서 0행이며 헤더를 유지했습니다. "
        "Val은 공식 사전 표면형·POS 허용 범위의 전체 추출입니다.\n\n"
        "file_counts.csv에 각 파일의 전체 항목 행 수와 고유 기본형 수를 기록했습니다. "
        "모든 분할 파일을 다시 읽어 원본의 모든 열·값·파일 내 순서와 일치하는지 검증했습니다. UTF-8 BOM입니다.\n")
    return reports


def main():
    root = Path(__file__).resolve().parent.parent
    base = root / "diagram_synonym_pairs"
    out = base / "all_entries"
    out.mkdir(exist_ok=True)
    metadata = json.loads((base / "summary.json").read_text())
    assert metadata["method"]["dictionary_only"]
    dictionaries, _ = dictionary_sources(base.resolve().parent)
    original = captions(Path(metadata["method"]["train_caption_basis"]["path"]))
    current = captions(root / "tvr/TextData/tvrtrain.caption.txt")
    validation = captions(root / "tvr/TextData/tvrval.caption.txt")
    assert original.keys() == current.keys()
    assert set(dictionaries["object"]) == set(dictionaries["action"]) == set(original)
    with (base / "dictionary_terms.csv").open(encoding="utf-8-sig", newline="") as handle:
        evidence = {(r["aspect"], r["surface"], r["source_pos"]): r for r in csv.DictReader(handle)}
    with (base / "final_term_regions.csv").open(encoding="utf-8-sig", newline="") as handle:
        catalog = {(r["aspect"], r["split"], r["lemma"]): r for r in csv.DictReader(handle)}
    regions = {key: info["term_region"] for key, info in catalog.items()}
    with (base / "no_pair_annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
        unclassified = {(r["aspect"], r["split"], r["caption_id"], r["source_expression"])
                        for r in csv.DictReader(handle) if r["region"] == "UNCLASSIFIED"}
    val_terms = {}
    with (base.resolve().parent / "tvrval_action_object_analysis.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            assert row["caption"].strip() == validation[row["caption_id"]].strip()
            val_terms[row["caption_id"]] = {kind: json.loads(row[f"{kind}s_raw"]) for kind in dictionaries}
    assert set(val_terms) == set(validation)
    observed_forms = {}
    for split, rows in [("train", {cid: {kind: source[cid] for kind, source in dictionaries.items()} for cid in original}),
                        ("validation", val_terms)]:
        for cid, terms in rows.items():
            for kind, raw in terms.items():
                for surface, tag in raw.items():
                    lemma = evidence[kind, surface, tag]["lemma"]
                    key = kind, split, lemma
                    if key in catalog and (kind, split, cid, surface) not in unclassified:
                        observed_forms.setdefault(key, set()).add(surface)
    wordnet_columns = make_wordnet_columns(root, evidence, catalog, observed_forms)
    fields = ["출처", "caption_ID", "caption_사전기준", "caption_현재", "현재caption_수정여부",
              "사전항목", "품사", "기본형_정규화", "영역", "항목상태", "사전근거_ID", "추출기준"] + WORDNET_FIELDS
    caption_fields = fields[:5] + ["Object_전체", "Object_항목수", "Action_전체", "Action_항목수", "추출기준"]
    handles = {kind: (out / f"all_{kind}s.csv").open("w", encoding="utf-8-sig", newline="") for kind in dictionaries}
    writers = {kind: csv.DictWriter(handle, fields) for kind, handle in handles.items()}
    for writer in writers.values():
        writer.writeheader()
    counts = Counter()
    try:
        with (out / "all_captions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, caption_fields)
            writer.writeheader()
            for split, source, now in [("train", original, current), ("validation", validation, validation)]:
                label = "Train" if split == "train" else "Val"
                for cid, caption in source.items():
                    terms = {kind: dictionaries[kind][cid] if split == "train" else val_terms[cid][kind]
                             for kind in dictionaries}
                    common = {"출처": label, "caption_ID": cid, "caption_사전기준": caption,
                              "caption_현재": now[cid], "현재caption_수정여부": caption != now[cid],
                              "추출기준": "공식 ID 사전 원본 전체" if split == "train" else "공식 사전 표면형·POS에 일치하는 Val 전체"}
                    writer.writerow({**common, "Object_전체": json.dumps(terms["object"], ensure_ascii=False),
                                     "Object_항목수": len(terms["object"]),
                                     "Action_전체": json.dumps(terms["action"], ensure_ascii=False), "Action_항목수": len(terms["action"])})
                    counts["captions", split] += 1
                    for kind, raw in terms.items():
                        counts[kind, split, "terms"] += len(raw)
                        for surface, tag in raw.items():
                            ref = evidence[kind, surface, tag]
                            region = regions.get((kind, split, ref["lemma"]), "UNCLASSIFIED")
                            if (kind, split, cid, surface) in unclassified:
                                region = "UNCLASSIFIED"
                            wn_info = dict(wordnet_columns(kind, split, ref["lemma"], tag))
                            wn_info["term_coverage_status"] = catalog[kind, split, ref["lemma"]]["coverage_status"] if region in {"A", "B", "C"} else region
                            writers[kind].writerow({**common, **wn_info, "사전항목": surface, "품사": tag,
                                                   "기본형_정규화": ref["lemma"], "영역": region, "항목상태": "공식사전 항목",
                                                   "사전근거_ID": cid if split == "train" else ref["dictionary_caption_id"]})
                            counts[kind, split, "rows"] += 1
                        if not raw:
                            wn_info = dict(wordnet_columns(kind, split, "", ""))
                            wn_info["term_coverage_status"] = "EMPTY"
                            writers[kind].writerow({**common, **wn_info, "영역": "EMPTY", "항목상태": "추출 항목 없음",
                                                   "사전근거_ID": cid if split == "train" else ""})
                            counts[kind, split, "empty"] += 1
                            counts[kind, split, "rows"] += 1
                    if counts["captions", split] % 25000 == 0:
                        print(f"{label}: {counts['captions', split]:,} captions exported", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    files = []
    for kind in dictionaries:
        path, restored, count = out / f"all_{kind}s.csv", {}, 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                count += 1
                raw = restored.setdefault((row["출처"], row["caption_ID"]), {})
                if row["사전항목"]:
                    assert row["사전항목"] not in raw
                    raw[row["사전항목"]] = row["품사"]
        assert len(restored) == len(original) + len(validation)
        assert {cid: raw for (split, cid), raw in restored.items() if split == "Train"} == dictionaries[kind]
        assert {cid: raw for (split, cid), raw in restored.items() if split == "Val"} == {cid: raw[kind] for cid, raw in val_terms.items()}
        files.append({"csv_file": path.name, "rows": count, "caption_count": len(restored),
                      "raw_terms_preserved": True, "empty_captions_preserved": True, "bytes": path.stat().st_size})
    with (out / "all_captions.csv").open(encoding="utf-8-sig", newline="") as handle:
        count = 0
        for row in csv.DictReader(handle):
            count += 1
            for kind in dictionaries:
                raw = json.loads(row[f"{kind.title()}_전체"])
                assert raw == (dictionaries[kind][row["caption_ID"]] if row["출처"] == "Train" else val_terms[row["caption_ID"]][kind])
        assert count == len(original) + len(validation)
    files.append({"csv_file": "all_captions.csv", "rows": count, "caption_count": count,
                  "raw_terms_preserved": True, "empty_captions_preserved": True, "bytes": (out / "all_captions.csv").stat().st_size})
    report = [{"유형": kind.title(), "출처": "Train" if split == "train" else "Val",
               "원본항목수": counts[kind, split, "terms"], "빈캡션수": counts[kind, split, "empty"], "CSV행수": counts[kind, split, "rows"]}
              for kind in dictionaries for split in ["train", "validation"]]
    write_csv(out / "file_counts.csv", list(report[0]), report)
    proof = {"status": "PASS", "llm_used": False, "files": files,
             "wordnet_columns": WORDNET_FIELDS, "wordnet_candidates_dictionary_only": True,
             "scope": "All supplied Train entries and all dictionary-whitelisted Val entries; unsupported POS, unclassified terms and empty captions retained",
             "train_caption_basis": metadata["method"]["train_caption_basis"]}
    (out / "verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text("# Object/Action 전체 CSV\n\n모든 원본 항목과 WordNet 조회 정보를 함께 담았습니다. 유의어가 없는 행도 보존합니다.\n\n"
                                   "- all_objects.csv / all_actions.csv: 모든 항목·품사·caption ID·원문. EMPTY와 UNCLASSIFIED도 포함\n"
                                   "- all_captions.csv: 전체 캡션별 object/action 원본 사전 목록\n"
                                   "- file_counts.csv: 종류·split별 전체 항목 및 행 수\n\n"
                                   "WordNet 등록 여부·공식 사전 내 유의어 후보 수/목록·반대 split 후보 수/목록·synset·선택 synset·의미·검토 여부를 포함했습니다. "
                                   "후보 표현은 같은 aspect 공식 사전 항목으로 제한합니다. 기본형이 없거나 지원하지 않는 품사는 조회 여부가 빈칸이고 후보 수는 0입니다.\n\n"
                                   "Train은 공식 ID 사전의 모든 항목을 그대로 보존했습니다. 사전 기준 원본 캡션과 현재 캡션을 모두 넣었습니다. "
                                   "Val은 공식 사전 표면형·POS 허용 범위의 전체 추출입니다. 기본형은 정규화 표시이며 사전항목은 원본 표면형입니다. "
                                   "CSV를 다시 읽어 모든 ID의 원본 사전 매핑과 완전히 일치하는지 검증했습니다. UTF-8 BOM입니다.\n")
    partition_by_region(out)
    print(json.dumps(files, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
