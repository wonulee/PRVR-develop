"""Apply rank-1 CSV synonyms to TVR train captions, preserving IDs and formatting."""

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    from transformers import AutoTokenizer

    workspace = Path(__file__).resolve().parent.parent
    source_dir = workspace / "rewriting/val_domain_analysis/linguistic_synonyms"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=workspace / "tvr/TextData/tvrtrain.caption.txt")
    parser.add_argument("--pairs", type=Path, default=source_dir / "tvrtrain_synonym_pairs.csv")
    parser.add_argument("--output-dir", type=Path, default=source_dir / "applied_synonyms")
    parser.add_argument("--tokenizer", type=Path, default=Path(
        "/home/viglab2/.cache/huggingface/hub/models--roberta-base/snapshots/"
        "e2da8e2f811d1448a5b465c236feacd80ffbac7b"))
    parser.add_argument("--apply", action="store_true", help="Back up and replace the original caption file.")
    parser.add_argument("--enforce-length-budget", action="store_true",
                        help="Keep the original unless the result has 14-20 words and fewer than 30 tokens.")
    args = parser.parse_args()

    original = args.captions.read_bytes()
    pairs_bytes = args.pairs.read_bytes()
    original_mode = stat.S_IMODE(args.captions.stat().st_mode)
    with args.pairs.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = defaultdict(list)
    keys = set()
    for row in rows:
        if int(row["synonym_rank"]) != 1:
            continue
        key = (row["caption_id"], row["term_type"], row["source_expression"])
        if key in keys or not row["source_expression"] or not row["synonym_expression"]:
            raise ValueError(f"Invalid rank-1 synonym: {key}")
        if any(c in row["synonym_expression"] for c in "\r\n"):
            raise ValueError(f"Multiline synonym: {key}")
        keys.add(key)
        selected[row["caption_id"]].append(row)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    lines = original.decode("utf-8").splitlines(keepends=True)
    updated, changes, audit, ids = [], [], [], set()
    for line in lines:
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        match = re.fullmatch(r"(\S+)([ \t]+)(.*)", body)
        if not match:
            raise ValueError(f"Invalid caption line: {body[:100]!r}")
        caption_id, separator, caption = match.groups()
        if caption_id in ids:
            raise ValueError(f"Duplicate caption ID: {caption_id}")
        ids.add(caption_id)
        candidates = selected.get(caption_id, [])
        if not candidates:
            updated.append(line)
            continue
        replacements = {}
        for row in candidates:
            if row["caption"].strip() != caption.strip():
                raise ValueError(f"CSV caption differs from original: {caption_id}")
            source, target = row["source_expression"], row["synonym_expression"]
            if source in replacements and replacements[source] != target:
                raise ValueError(f"Conflicting replacements: {caption_id}, {source}")
            replacements[source] = target
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(
            re.escape(source) for source in sorted(replacements, key=lambda s: (-len(s), s))) + r")(?!\w)")
        occurrences = Counter()

        def replace(found):
            source = found.group()
            occurrences[source] += 1
            return replacements[source]

        rewritten = pattern.sub(replace, caption)
        budget_rejected = args.enforce_length_budget and not (
            14 <= len(rewritten.split()) <= 20
            and len(tokenizer.encode(rewritten, add_special_tokens=True)) < 30)
        if budget_rejected:
            rewritten = caption
        substitutions = []
        for row in candidates:
            source, target = row["source_expression"], row["synonym_expression"]
            count = occurrences[source]
            status = "length_budget_exceeded" if budget_rejected else "source_not_standalone_token" if not count else (
                "same_expression" if source == target else "applied")
            audit.append({"caption_id": caption_id, "term_type": row["term_type"],
                          "source_expression": source, "synonym_expression": target,
                          "synonym_rank": 1, "occurrences": count, "status": status})
            if status == "applied":
                substitutions.append({"term_type": row["term_type"], "source": source,
                                      "target": target, "occurrences": count,
                                      "meaning": row["meaning"], "val_caption_id": row["val_caption_id"]})
        updated.append(caption_id + separator + rewritten + ending)
        if rewritten != caption:
            before_words, after_words = len(caption.split()), len(rewritten.split())
            before_tokens = len(tokenizer.encode(caption, add_special_tokens=True))
            after_tokens = len(tokenizer.encode(rewritten, add_special_tokens=True))
            changes.append({"caption_id": caption_id, "original_caption": caption,
                            "rewritten_caption": rewritten,
                            "substitutions": json.dumps(substitutions, ensure_ascii=False),
                            "original_word_count": before_words, "rewritten_word_count": after_words,
                            "original_token_count": before_tokens, "rewritten_token_count": after_tokens,
                            "word_budget_pass": 14 <= after_words <= 20,
                            "token_budget_pass": after_tokens < 30})

    if missing := set(selected) - ids:
        raise ValueError(f"CSV IDs missing from captions: {sorted(missing)[:5]}")
    rewritten_bytes = "".join(updated).encode("utf-8")
    if len(updated) != len(lines) or sum(a != b for a, b in zip(lines, updated)) != len(changes):
        raise ValueError("Caption count or change count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview = args.output_dir / "tvrtrain.caption.rewritten.txt"
    preview.write_bytes(rewritten_bytes)
    write_csv(args.output_dir / "caption_changes.csv", [
        "caption_id", "original_caption", "rewritten_caption", "substitutions",
        "original_word_count", "rewritten_word_count", "original_token_count",
        "rewritten_token_count", "word_budget_pass", "token_budget_pass"], changes)
    write_csv(args.output_dir / "replacement_audit.csv", [
        "caption_id", "term_type", "source_expression", "synonym_expression",
        "synonym_rank", "occurrences", "status"], audit)
    summary = {"caption_file": str(args.captions), "source_csv": str(args.pairs),
               "source_csv_sha256": digest(pairs_bytes), "original_sha256": digest(original),
               "rewritten_sha256": digest(rewritten_bytes), "caption_count": len(lines),
               "csv_pair_count": len(rows), "rank_1_pair_count": len(keys),
               "candidate_caption_count": len(selected), "changed_caption_count": len(changes),
               "unchanged_caption_count": len(lines) - len(changes),
               "replacement_status_counts": dict(Counter(row["status"] for row in audit)),
               "replacement_occurrence_count": sum(row["occurrences"] for row in audit if row["status"] == "applied"),
               "changed_captions_within_length_budget": sum(
                   row["word_budget_pass"] and row["token_budget_pass"] for row in changes),
               "selection_rule": "rank 1; simultaneous, case-sensitive, whole-word replacements",
               "length_budget_filter": args.enforce_length_budget,
               "length_budget_policy": (
                   "Reject changes outside 14-20 words or >=30 RoBERTa tokens, including special tokens."
                   if args.enforce_length_budget else
                   "User requested all synonyms, preserving the remaining original sentence."),
               "token_count_includes_special_tokens": True, "tokenizer": str(args.tokenizer),
               "applied_to_original": False, "preview": str(preview)}
    if args.apply:
        backup = args.output_dir / "tvrtrain.caption.original.txt"
        with backup.open("xb") as handle:
            handle.write(original)
        os.chmod(backup, original_mode)
        if backup.read_bytes() != original:
            raise ValueError("Backup verification failed")
        if args.captions.read_bytes() != original or args.pairs.read_bytes() != pairs_bytes:
            raise ValueError("Inputs changed during preparation")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.captions.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(rewritten_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, original_mode)
            os.replace(temporary, args.captions)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        if args.captions.read_bytes() != rewritten_bytes:
            raise ValueError("Applied file verification failed")
        summary.update(applied_to_original=True, backup=str(backup))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
