"""Join verified mapping rows with representative captions and WordNet for CSV viewing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_val_domain import write_csv
from export_mapping_diagram_csv import MAIN_FIELDS, WORDNET_FIELDS


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent.parent
                        / "diagram_synonym_pairs_final_20261007")
    args = parser.parse_args()
    base = args.package_dir
    source, out = base / "mapping", base / "ABC_csv"
    for filename in ["verification.json", "roundtrip_verification.json"]:
        assert json.loads((source / filename).read_text())["status"] == "PASS"
    counts = read_csv(base / "reports/ABC_mapping_counts.csv")
    inputs = [source / "main" / Path(row["mapping_csv"]).name for row in counts]
    inputs += [source / f for f in ["captions.csv", "wordnet.csv", "meanings.csv",
                                   "verification.json", "roundtrip_verification.json"]]
    inputs.append(base / "reports/ABC_mapping_counts.csv")
    input_hashes = {str(path.relative_to(base)): sha256(path) for path in inputs}
    captions = {(row["split"], row["caption_id"]): row["caption"]
                for row in read_csv(source / "captions.csv")}
    wordnets = {row["source_wordnet_id"]: row for row in read_csv(source / "wordnet.csv")}
    meanings = {row["meaning_id"]: row["meaning"] for row in read_csv(source / "meanings.csv")}
    fields = [*MAIN_FIELDS, "example_train_caption", "example_val_caption", "meaning", *WORDNET_FIELDS]
    out.mkdir(exist_ok=True)
    mapping_ids, reports = set(), []
    for item in counts:
        filename = Path(item["mapping_csv"]).name
        rows = read_csv(source / "main" / filename)
        expanded = []
        for row in rows:
            assert row["mapping_id"] not in mapping_ids
            mapping_ids.add(row["mapping_id"])
            wn = wordnets[row["source_wordnet_id"]]
            split = "train" if row["source_split"] == "train" else "validation"
            prefix = "train" if split == "train" else "val"
            assert (wn["term_type"], wn["split"], wn["source_lemma"]) == (
                row["term_type"], split, row[f"{prefix}_lemma"])
            extra = {f"example_{side}_caption": captions[caption_split, row[f"example_{side}_caption_id"]]
                     if row[f"example_{side}_caption_id"] else ""
                     for side, caption_split in [("train", "train"), ("val", "validation")]}
            expanded.append({**row, **extra, "meaning": meanings[row["meaning_id"]]
                             if row["meaning_id"] else "", **{field: wn[field] for field in WORDNET_FIELDS}})
        path = out / filename
        write_csv(path, fields, expanded)
        restored = read_csv(path)
        assert restored == expanded
        assert [{field: row[field] for field in MAIN_FIELDS} for row in restored] == rows
        assert len(restored) == int(item["mapping_rows"])
        assert sum(int(row["full_rows"]) for row in restored) == int(item["original_rows"])
        reports.append({**item, "mapping_csv": f"../mapping/main/{filename}",
                        "links_csv": f"../mapping/links/{filename}", "csv_file": filename,
                        "bytes": path.stat().st_size})
        print(f"{filename}: {len(restored):,} rows", flush=True)
    assert all(sha256(base / name) == digest for name, digest in input_hashes.items())
    write_csv(out / "file_counts.csv", list(reports[0]), reports)
    table_rows = []
    for label, suffix in [("A·Train", "A_train.csv"), ("B·Train", "B_train.csv"),
                          ("B·Val", "B_val.csv"), ("C·Val", "C_val.csv")]:
        cells = [f"[{int(next(r['mapping_rows'] for r in reports if r['csv_file'] == kind + '_' + suffix)):,}행]"
                 f"({kind}_{suffix})" for kind in ["object", "action"]]
        table_rows.append(f"| {label} | {' | '.join(cells)} |")
    (out / "README.md").write_text(
        "# A/B/C Object·Action 보기용 CSV\n\n"
        "한 행에서 Train/Val 표현·POS, 대표 캡션 ID·문장, WordNet 후보·synset·의미를 함께 확인합니다. "
        "기존 표현별 매핑 20열에 대표 문장 2열, 의미 1열, WordNet 8열을 붙인 31열 CSV입니다. "
        "Excel에서 바로 열 수 있도록 UTF-8 BOM으로 저장했습니다.\n\n"
        "| 영역 | Object | Action |\n| --- | --- | --- |\n" + "\n".join(table_rows) + "\n\n"
        "EMPTY·UNCLASSIFIED도 별도 CSV로 보존했습니다. [파일별 행 수](file_counts.csv)와 "
        "[검증 결과](verification.json)를 확인할 수 있습니다.\n\n"
        "행 수는 표현·POS 대응을 중복 제거한 매핑 수입니다. B에서 한 표현이 여러 표현과 연결되면 여러 행입니다. "
        "`full_rows`는 묶인 원본 행 수, `source_caption_count`는 출발 캡션 수입니다. "
        "전체 16개 파일의 35,930행과 기존 20열의 값·순서를 그대로 보존했습니다.\n\n"
        "`example_train_caption`과 `example_val_caption`은 실제 연결된 대표 문장입니다. "
        "전체 캡션 연결은 같은 파일명의 [mapping/links](../mapping/links/) CSV에서 "
        "`mapping_id`로 확인하며, 문장은 [captions.csv](../mapping/captions.csv)의 `caption_key`로 찾습니다. "
        "원본 1,494,233행에 대한 전체 연결 자료는 그대로 유지했습니다.\n\n"
        "WordNet 열은 출발 split의 기존 조회 결과입니다. `mapping_type`의 `WORDNET_SYNONYM`은 기존 후보, "
        "`DIRECT_LEMMA_REFERENCE`·`WORDNET_COVERAGE_REFERENCE`는 B영역의 실제 반대편 표현 참고 연결입니다. "
        "참고 연결의 `replacement_expression`은 비어 있습니다. `requires_review` 조건도 그대로 유지했습니다.\n\n"
        "기존 JSON 사전의 동일 Object/Action 표현·POS 범위를 유지했습니다. "
        "C·Val=0은 이 제한 범위의 결과입니다. LLM을 사용하지 않았고 캡션을 수정하지 않았습니다.\n\n"
        "재생성: `python rewriting/export_ABC_readable_csv.py` (작업 공간 루트에서 실행).\n",
        encoding="utf-8")
    output_hashes = {path.name: sha256(path) for path in sorted(out.glob("*.csv"))}
    result = {"status": "PASS", "llm_used": False, "dictionary_scope_changed": False,
              "mapping_rows": len(mapping_ids), "case_csv_files": len(reports), "columns": len(fields),
              "original_rows_accounted_for": sum(int(r["original_rows"]) for r in reports),
              "original_mapping_columns_values_and_order_preserved": True,
              "representative_caption_and_wordnet_joins_verified": True, "input_files_unchanged": True,
              "case_csv_bytes": sum(r["bytes"] for r in reports), "input_sha256": input_hashes,
              "output_csv_sha256": output_hashes, "script_sha256": sha256(Path(__file__))}
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
