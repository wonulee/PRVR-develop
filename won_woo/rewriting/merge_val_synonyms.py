"""Keep original TVR captions and append all CSV synonym combinations per video."""

import argparse
import csv
import itertools
import json
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from apply_val_synonyms import digest, write_csv


def variants(caption, term_groups):
    """Apply each combination once; replacement output is never matched again."""
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(source) for source in sorted(term_groups, key=lambda s: (-len(s), s))) + r")(?!\w)")
    combinations = sorted(itertools.product(*term_groups.values()), key=lambda choice: (
        max(int(row["synonym_rank"]) for row in choice),
        tuple(int(row["synonym_rank"]) for row in choice),
        tuple(row["synonym_expression"] for row in choice)))
    for choice in combinations:
        replacements = {row["source_expression"]: row["synonym_expression"] for row in choice}
        occurrences = Counter()

        def replace(match):
            source = match.group()
            occurrences[source] += 1
            return replacements[source]

        candidate = pattern.sub(replace, caption)
        substitutions = [{"term_type": row["term_type"], "source": row["source_expression"],
                          "target": row["synonym_expression"], "rank": int(row["synonym_rank"]),
                          "occurrences": occurrences[row["source_expression"]],
                          "meaning": row["meaning"], "val_caption_id": row["val_caption_id"]}
                         for row in choice]
        yield candidate, substitutions


def main():
    workspace = Path(__file__).resolve().parent.parent
    source_dir = workspace / "rewriting/val_domain_analysis/linguistic_synonyms"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=workspace / "tvr/TextData/tvrtrain.caption.txt")
    parser.add_argument("--pairs", type=Path, default=source_dir / "tvrtrain_synonym_pairs.csv")
    parser.add_argument("--merged", type=Path, default=workspace / "tvr/TextData/tvrtrain.caption_replace_merge_all.txt")
    parser.add_argument("--output-dir", type=Path, default=source_dir / "all_rank_merge")
    parser.add_argument("--apply", action="store_true", help="Back up and update the merged caption file.")
    args = parser.parse_args()
    if args.merged.resolve() in {args.captions.resolve(), args.pairs.resolve()}:
        raise ValueError("Merged output must have a separate path")
    original, pairs_bytes = args.captions.read_bytes(), args.pairs.read_bytes()
    previous_merge = args.merged.read_bytes() if args.merged.exists() else None
    original_lines = original.decode("utf-8").splitlines(keepends=True)
    videos, by_id = defaultdict(list), {}
    last_video, closed_videos = None, set()
    for line in original_lines:
        match = re.fullmatch(r"(\S+)#enc#(\d+)[ \t]+([^\r\n]*)(?:\r?\n)?", line)
        if not match:
            raise ValueError(f"Invalid caption format: {line[:100]!r}")
        video_id, enc, caption = match.groups()
        caption_id = f"{video_id}#enc#{enc}"
        if caption_id in by_id:
            raise ValueError(f"Duplicate caption ID: {caption_id}")
        if video_id != last_video:
            if video_id in closed_videos:
                raise ValueError(f"Video captions must be contiguous: {video_id}")
            if last_video is not None:
                closed_videos.add(last_video)
            last_video = video_id
        record = {"caption_id": caption_id, "video_id": video_id, "enc": int(enc),
                  "caption": caption, "line": line}
        by_id[caption_id] = record
        videos[video_id].append(record)

    with args.pairs.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(lambda: defaultdict(list))
    pair_keys = set()
    for row in rows:
        caption_id, source = row["caption_id"], row["source_expression"]
        rank = int(row["synonym_rank"])
        key = (caption_id, source, rank)
        if key in pair_keys or rank < 1 or not source or not row["synonym_expression"]:
            raise ValueError(f"Invalid candidate: {key}")
        if any(c in row["synonym_expression"] for c in "\r\n"):
            raise ValueError(f"Multiline candidate: {key}")
        pair_keys.add(key)
        if caption_id not in by_id or row["caption"].strip() != by_id[caption_id]["caption"].strip():
            raise ValueError(f"CSV caption differs from original: {caption_id}")
        if row["video_id"] != by_id[caption_id]["video_id"] or int(row["enc"]) != by_id[caption_id]["enc"]:
            raise ValueError(f"CSV ID fields disagree: {caption_id}")
        grouped[caption_id][source].append(row)
    for terms in grouped.values():
        for candidates in terms.values():
            candidates.sort(key=lambda row: int(row["synonym_rank"]))

    merged_lines, comparison, coverage = [], [], Counter()
    for video_id, captions in videos.items():
        encs = [record["enc"] for record in captions]
        if encs != sorted(encs):
            raise ValueError(f"Original enc order is not ascending: {video_id}")
        merged_lines.extend(record["line"] for record in captions)
        next_enc = max(encs) + 1
        seen = {" ".join(record["caption"].split()): record["caption_id"] for record in captions}
        for record in captions:
            caption_id = record["caption_id"]
            if caption_id not in grouped:
                continue
            for variant_index, (candidate, substitutions) in enumerate(
                    variants(record["caption"], grouped[caption_id]), 1):
                candidate_key = " ".join(candidate.split())
                duplicate_id = seen.get(candidate_key, "")
                added_id, added_enc = "", ""
                if candidate_key == " ".join(record["caption"].split()):
                    status = "unchanged"
                elif duplicate_id:
                    status = "duplicate"
                else:
                    if not merged_lines[-1].endswith("\n"):
                        raise ValueError("Cannot append after a caption without its line delimiter")
                    added_enc = next_enc
                    added_id = f"{video_id}#enc#{next_enc}"
                    next_enc += 1
                    seen[candidate_key] = added_id
                    merged_lines.append(f"{added_id} {candidate}\n")
                    status = "added"
                    for change in substitutions:
                        if change["occurrences"] and change["source"] != change["target"]:
                            coverage[(caption_id, change["source"], change["rank"])] += 1
                comparison.append({"source_caption_id": caption_id, "video_id": video_id,
                                   "source_enc": record["enc"], "variant_index": variant_index,
                                   "added_caption_id": added_id, "added_enc": added_enc,
                                   "max_synonym_rank": max(change["rank"] for change in substitutions),
                                   "original_caption": record["caption"], "candidate_caption": candidate,
                                   "substitutions": json.dumps(substitutions, ensure_ascii=False),
                                   "status": status, "duplicate_of_caption_id": duplicate_id})
    output_ids = [line.split(None, 1)[0] for line in merged_lines]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("Duplicate IDs in merged file")
    if [line for line in merged_lines if line.split(None, 1)[0] in by_id] != original_lines:
        raise ValueError("Original lines or ordering changed")
    merged_bytes = "".join(merged_lines).encode("utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview = args.output_dir / "tvrtrain.caption_replace_merge_all.preview.txt"
    preview.write_bytes(merged_bytes)
    write_csv(args.output_dir / "caption_comparison.csv", [
        "source_caption_id", "video_id", "source_enc", "variant_index", "added_caption_id", "added_enc",
        "max_synonym_rank", "original_caption", "candidate_caption", "substitutions", "status",
        "duplicate_of_caption_id"], comparison)
    pair_audit = []
    for row in rows:
        source, target = row["source_expression"], row["synonym_expression"]
        count = coverage[(row["caption_id"], source, int(row["synonym_rank"]))]
        boundary_match = re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", row["caption"])
        status = "included" if count else ("same_expression" if source == target else (
            "source_not_standalone_token" if not boundary_match else "duplicate_caption"))
        pair_audit.append({"source_caption_id": row["caption_id"], "term_type": row["term_type"],
                           "source_expression": source, "synonym_expression": target,
                           "synonym_rank": row["synonym_rank"], "added_variant_count": count, "status": status})
    write_csv(args.output_dir / "synonym_pair_coverage.csv", [
        "source_caption_id", "term_type", "source_expression", "synonym_expression",
        "synonym_rank", "added_variant_count", "status"], pair_audit)
    summary = {"original_file": str(args.captions), "original_sha256": digest(original),
               "source_csv": str(args.pairs), "source_csv_sha256": digest(pairs_bytes),
               "merged_file": str(args.merged), "merged_sha256": digest(merged_bytes),
               "original_caption_count": len(original_lines), "merged_caption_count": len(merged_lines),
               "added_caption_count": len(merged_lines) - len(original_lines),
               "candidate_combination_count": len(comparison),
               "variant_status_counts": dict(Counter(row["status"] for row in comparison)),
               "added_variants_by_max_rank": dict(Counter(
                   row["max_synonym_rank"] for row in comparison if row["status"] == "added")),
               "input_pairs_by_rank": dict(Counter(row["synonym_rank"] for row in rows)),
               "pair_status_counts": dict(Counter(row["status"] for row in pair_audit)),
               "included_pairs_by_rank": dict(Counter(
                   row["synonym_rank"] for row in pair_audit if row["status"] == "included")),
               "combination_rule": "All choices across source terms, ordered by maximum rank then rank tuple.",
               "enc_rule": "Per video, start at original maximum enc + 1 and increase only for added captions.",
               "duplicate_rule": "Compare whitespace-normalized caption text within the same video.",
               "length_budget_filter": False, "applied_to_merged_file": False, "preview": str(preview)}
    if args.apply:
        if args.captions.read_bytes() != original or args.pairs.read_bytes() != pairs_bytes:
            raise ValueError("Inputs changed during preparation")
        if (args.merged.read_bytes() if args.merged.exists() else None) != previous_merge:
            raise ValueError("Merged file changed during preparation")
        mode = stat.S_IMODE(args.merged.stat().st_mode) if args.merged.exists() else 0o664
        if previous_merge is not None:
            backup = args.output_dir / "tvrtrain.caption_replace_merge.before_all_ranks.txt"
            with backup.open("xb") as handle:
                handle.write(previous_merge)
            if backup.read_bytes() != previous_merge:
                raise ValueError("Previous merge backup verification failed")
            summary.update(previous_merge_backup=str(backup), previous_merge_sha256=digest(previous_merge))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.merged.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(merged_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, args.merged)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        if args.merged.read_bytes() != merged_bytes or args.captions.read_bytes() != original:
            raise ValueError("Applied merge or original preservation verification failed")
        summary["applied_to_merged_file"] = True
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
