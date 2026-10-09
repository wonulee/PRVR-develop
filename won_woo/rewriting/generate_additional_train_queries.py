"""Create additional TVR Train queries from the verified CSVs, without an LLM."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from analyze_val_domain import dictionary_vocabulary
from apply_val_synonyms import digest, write_csv
from export_mapping_diagram_csv import WORDNET_FIELDS
from select_diagram_synonyms import BLOCKED, FAMILIES, NONVISUAL, SPEECH, read_captions, syntax_status


TOKENIZER = Path("/home/viglab2/.cache/huggingface/hub/models--roberta-base/snapshots/"
                 "e2da8e2f811d1448a5b465c236feacd80ffbac7b")
FIELDS = ["new_caption_id", "source_caption_id", "video_id", "source_timestamp", "term_type",
          "source_expression", "replacement_expression", "source_pos", "replacement_pos",
          "source_annotation_expression", "source_annotation_pos", "source_dictionary_caption_id",
          "replacement_dictionary_caption_id", "source_annotation_basis", "source_span_start",
          "source_span_end", "mapping_id", "mapping_csv", "val_caption_id", "val_caption",
          "original_caption", "additional_caption", "word_count", "roberta_token_count",
          "shared_wordnet_synsets", "meaning", "requires_video_review", *WORDNET_FIELDS]
AUDIT_FIELDS = ["source_caption_id", "term_type", "source_expression", "replacement_expression",
                "mapping_id", "source_span_start", "status", "new_caption_id"]


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_timestamps(root, originals):
    timestamps, release = {}, defaultdict(set)
    with (root / "tvr/TextData/tvr_train_release.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            release[row["vid_name"], row["desc"]].add(tuple(row["ts"]))
    for cid, caption in originals.items():
        choices = release[cid.rsplit("#enc#", 1)[0], caption]
        if len(choices) == 1:
            timestamps[cid] = list(next(iter(choices)))
    return timestamps


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def generate_unfiltered(args):
    root = Path(__file__).resolve().parent.parent
    base = args.package_dir
    out = args.output_dir or base / ("additional_captions_rank1" if args.rank_one_only else "additional_captions_unfiltered")
    baseline = root / "rewriting/val_domain_analysis/applied_diagram_synonyms_20261007/tvrtrain.caption.original.txt"
    paths = [args.captions, baseline, root / "tvr/TextData/tvrval.caption.txt",
             root / "tvr/TextData/tvrtest.caption.txt", root / "tvr/TextData/tvr_train_release.jsonl"]
    paths += [base / "ABC_csv" / f"{kind}_B_train.csv" for kind in ["object", "action"]]
    if args.rank_one_only:
        paths += [base / "mapping/links" / f"{kind}_B_train.csv" for kind in ["object", "action"]]
    hashes = {str(path): file_sha256(path) for path in paths}
    original_bytes, train = read_captions(args.captions)
    _, originals = read_captions(baseline)
    _, val = read_captions(root / "tvr/TextData/tvrval.caption.txt")
    _, test = read_captions(root / "tvr/TextData/tvrtest.caption.txt")
    timestamps = source_timestamps(root, originals)
    pairs, groups, mapping_counts = {}, defaultdict(list), Counter()
    for kind in ["object", "action"]:
        rows = csv_rows(base / "ABC_csv" / f"{kind}_B_train.csv")
        rank1_rows, first_candidates = {}, {}
        if args.rank_one_only:
            with (base / "mapping/links" / f"{kind}_B_train.csv").open(encoding="utf-8-sig", newline="") as handle:
                for link in csv.DictReader(handle):
                    if link["synonym_rank"] == "1":
                        rank1_rows.setdefault(link["mapping_id"], int(link["source_data_row_number"]))
            for row in rows:
                if row["mapping_id"] not in rank1_rows or not row["replacement_expression"]:
                    continue
                option = (rank1_rows[row["mapping_id"]], int(row["mapping_id"]), row["replacement_expression"])
                source = row["train_expression"]
                if source not in first_candidates or option < first_candidates[source]:
                    first_candidates[source] = option
            mapping_counts[kind + "/distinct_sources_with_rank1"] = len(first_candidates)
        for row in rows:
            mapping_counts[kind + "/input_mapping_rows"] += 1
            source, target = row["train_expression"], row["replacement_expression"]
            if not source or not target:
                mapping_counts[kind + "/no_replacement_in_csv"] += 1
                continue
            mapping_counts[kind + "/replacement_mapping_rows"] += 1
            if args.rank_one_only and (row["mapping_id"] not in rank1_rows or
                                      target != first_candidates[source][2]):
                mapping_counts[kind + "/not_selected_rank1_candidate"] += 1
                continue
            mapping_counts[kind + "/selected_mapping_rows"] += 1
            key = kind, source, target
            if key not in pairs:
                pairs[key] = {"pair_id": len(pairs) + 1, "term_type": kind,
                              "source_expression": source, "replacement_expression": target,
                              "mapping_ids": [], "mapping_csv": f"ABC_csv/{kind}_B_train.csv"}
                if args.rank_one_only:
                    pairs[key].update(synonym_rank=1, rank1_source_data_row_number=rank1_rows[row["mapping_id"]])
            pairs[key]["mapping_ids"].append(row["mapping_id"])
            if args.rank_one_only:
                pairs[key]["rank1_source_data_row_number"] = min(
                    pairs[key]["rank1_source_data_row_number"], rank1_rows[row["mapping_id"]])
    for pair in pairs.values():
        pair["mapping_ids"] = " | ".join(pair["mapping_ids"])
        groups[pair["source_expression"]].append(pair)
    special = sorted(source for source in groups if not re.fullmatch(r"\w+", source))
    special_pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, special)) + r")(?!\w)") if special else None
    word_pattern = re.compile(r"\w+")
    seen, next_enc = defaultdict(dict), defaultdict(int)
    for cid, caption in train.items():
        seen[cid.rsplit("#enc#", 1)[0]][caption] = cid
    original_ids = train.keys() | val.keys() | test.keys()
    for cid in original_ids:
        video, number = cid.rsplit("#enc#", 1)
        next_enc[video] = max(next_enc[video], int(number) + 1)
    fields = ["new_caption_id", "source_caption_id", "video_id", "term_type", "source_expression",
              "replacement_expression", "pair_id", "mapping_ids", "mapping_csv", "source_span_start",
              "source_span_end", "original_caption", "additional_caption", "source_timestamp", "timestamp_status"]
    if args.rank_one_only:
        fields += ["synonym_rank", "rank1_source_data_row_number"]
        assert len(pairs) == len({(pair["term_type"], pair["source_expression"]) for pair in pairs.values()})
    link_fields = ["new_caption_id", "source_caption_id", "term_type", "pair_id",
                   "source_span_start", "source_span_end", "status"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "links").mkdir(exist_ok=True)
    write_csv(out / "replacement_pairs.csv", list(next(iter(pairs.values()))), pairs.values())
    files, active, counts, source_ids = [], {}, Counter(), set()
    with ExitStack() as stack:
        def append_csv(group, row):
            state = active.get(group)
            if state is None or state[2]["rows"] == args.csv_part_rows:
                if state is not None:
                    state[0].close()
                part = sum(item["group"] == group for item in files) + 1
                relative = f"links/generation_links_{part:04d}.csv" if group == "links" else f"{group}_additional_queries_{part:04d}.csv"
                handle = stack.enter_context((out / relative).open("w", encoding="utf-8-sig", newline=""))
                writer = csv.DictWriter(handle, fieldnames=link_fields if group == "links" else fields)
                writer.writeheader()
                info = {"group": group, "csv_file": relative, "rows": 0}
                files.append(info)
                state = handle, writer, info
                active[group] = state
            state[1].writerow(row)
            state[2]["rows"] += 1

        added = stack.enter_context((out / "tvrtrain.additional.caption.txt").open("w", encoding="utf-8"))
        merged = stack.enter_context((out / "tvrtrain.caption.with_additional.txt").open("wb"))
        merged.write(original_bytes)
        if not original_bytes.endswith(b"\n"):
            merged.write(b"\n")
        for index, (cid, caption) in enumerate(train.items(), 1):
            video = cid.rsplit("#enc#", 1)[0]
            matches = [m for m in word_pattern.finditer(caption) if m.group() in groups]
            if special_pattern:
                matches.extend(special_pattern.finditer(caption))
            for found in sorted(matches, key=lambda m: (m.start(), -len(m.group()))):
                source, start, end = found.group(), found.start(), found.end()
                for pair in groups[source]:
                    counts["candidate_occurrences"] += 1
                    target = pair["replacement_expression"]
                    rewritten = caption[:start] + target + caption[end:]
                    new_id = seen[video].get(rewritten)
                    status = "duplicate_current_query" if new_id in original_ids else "duplicate_generated_query" if new_id else "generated"
                    counts[status] += 1
                    if status == "generated":
                        new_id = f"{video}#enc#{next_enc[video]}"
                        next_enc[video] += 1
                        assert new_id not in original_ids
                        seen[video][rewritten] = new_id
                        source_ids.add(cid)
                        counts[pair["term_type"]] += 1
                        if cid not in timestamps:
                            counts["generated_without_resolved_timestamp"] += 1
                        append_csv(pair["term_type"], {**pair, "new_caption_id": new_id,
                            "source_caption_id": cid, "video_id": video, "source_span_start": start,
                            "source_span_end": end, "original_caption": caption, "additional_caption": rewritten,
                            "source_timestamp": json.dumps(timestamps[cid]) if cid in timestamps else "",
                            "timestamp_status": "resolved_from_release" if cid in timestamps else "unresolved_kept"})
                        line = f"{new_id} {rewritten}\n"
                        added.write(line)
                        merged.write(line.encode("utf-8"))
                    append_csv("links", {"new_caption_id": new_id, "source_caption_id": cid,
                        "term_type": pair["term_type"], "pair_id": pair["pair_id"], "source_span_start": start,
                        "source_span_end": end, "status": status})
            if index % 10000 == 0:
                print(f"Read {index:,}/{len(train):,} captions; generated {counts['generated']:,} queries", flush=True)
    for info in files:
        info["bytes"] = (out / info["csv_file"]).stat().st_size
    write_csv(out / "file_counts.csv", ["group", "csv_file", "rows", "bytes"], files)
    verified_ids, verified_queries, verified_count = set(), set(), 0
    pair_index = {str(pair["pair_id"]): pair for pair in pairs.values()}
    for info in files:
        if info["group"] == "links":
            continue
        with (out / info["csv_file"]).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                source = train[row["source_caption_id"]]
                start, end = int(row["source_span_start"]), int(row["source_span_end"])
                assert source[start:end] == row["source_expression"]
                assert row["additional_caption"] == source[:start] + row["replacement_expression"] + source[end:]
                assert row["original_caption"] == source
                pair = pair_index[row["pair_id"]]
                assert all(row[field] == str(value) for field, value in pair.items())
                assert row["new_caption_id"] not in original_ids and row["new_caption_id"] not in verified_ids
                assert row["new_caption_id"].rsplit("#enc#", 1)[0] == row["video_id"] == row["source_caption_id"].rsplit("#enc#", 1)[0]
                fingerprint = digest((row["video_id"] + "\0" + row["additional_caption"]).encode())
                assert fingerprint not in verified_queries
                verified_ids.add(row["new_caption_id"])
                verified_queries.add(fingerprint)
                verified_count += 1
    assert verified_count == counts["generated"]
    assert sum(info["rows"] for info in files if info["group"] == "links") == counts["candidate_occurrences"]
    with (out / "tvrtrain.additional.caption.txt").open(encoding="utf-8") as handle:
        assert sum(1 for line in handle) == counts["generated"]
    with (out / "tvrtrain.caption.with_additional.txt").open("rb") as handle:
        assert handle.read(len(original_bytes)) == original_bytes
    assert all(file_sha256(Path(path)) == value for path, value in hashes.items())
    output_paths = [out / info["csv_file"] for info in files]
    output_paths += [out / name for name in ["replacement_pairs.csv", "file_counts.csv", "tvrtrain.additional.caption.txt", "tvrtrain.caption.with_additional.txt"]]
    result = dict(status="PASS", llm_used=False, input_queries=len(train), generated_queries=counts["generated"],
        source_queries=len(source_ids), generated_by_aspect={kind: counts[kind] for kind in ["object", "action"]},
        merged_queries=len(train) + counts["generated"], counts=dict(counts), input_mapping_counts=dict(mapping_counts),
        distinct_csv_replacement_pairs=len(pairs), csv_part_max_rows=args.csv_part_rows,
        rank_one_only=args.rank_one_only,
        rank_policy="Original synonym_rank=1 from mapping/links; one replacement per aspect/exact source; earliest original data row wins ties" if args.rank_one_only else "All CSV replacement candidates",
        filters={name: False for name in ["word_budget", "subword_token_budget", "dictionary_recheck", "pos_match",
                                        "context_rules", "conservative_lexical_families", "timestamp_required", "pre_replacement_baseline_dedup"]},
        policy="Case-sensitive whole-expression matches; one CSV replacement per query; deduplicate current/generated queries per video. No linguistic filters. Candidate selection follows rank_policy.",
        metadata_only="Existing CSV POS/WordNet data remain in ABC_csv and are linked by mapping_ids; missing timestamps are kept blank.",
        all_candidate_source_links_preserved=True, original_files_unchanged=True, original_train_bytes_preserved=True,
        checks="CSV reconstruction, replacement-pair values, ID uniqueness, video IDs, deduplication, counts, input hashes; no linguistic quality checks",
        input_sha256=hashes, script_sha256=file_sha256(Path(__file__)),
        output_sha256={str(path.relative_to(out)): file_sha256(path) for path in output_paths})
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if "sha256" not in key}, ensure_ascii=False, indent=2))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=root / "diagram_synonym_pairs_final_20261007")
    parser.add_argument("--captions", type=Path, default=root / "tvr/TextData/tvrtrain.caption.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-rule-filters", action="store_true",
                        help="Use CSV replacements without length, dictionary, POS or context filters.")
    parser.add_argument("--csv-part-rows", type=int, default=200000)
    parser.add_argument("--rank-one-only", action="store_true", help="Use one original rank-1 candidate per aspect/source expression.")
    args = parser.parse_args()
    if args.rank_one_only and not args.no_rule_filters:
        parser.error("--rank-one-only requires --no-rule-filters")
    if args.no_rule_filters:
        if args.csv_part_rows < 1:
            parser.error("--csv-part-rows must be positive")
        generate_unfiltered(args)
        return
    import nltk
    import spacy
    from transformers import AutoTokenizer

    base, out = args.package_dir, args.output_dir or args.package_dir / "additional_captions"
    baseline = root / "rewriting/val_domain_analysis/applied_diagram_synonyms_20261007/tvrtrain.caption.original.txt"
    input_paths = [args.captions, baseline, root / "tvr/TextData/tvrval.caption.txt",
                   root / "tvr/TextData/tvrtest.caption.txt", root / "tvr/TextData/tvr_train_release.jsonl"]
    input_paths += [root / f"tvr/TextData/tvr_id_{kind}_dict.json" for kind in FAMILIES]
    input_paths += [base / "ABC_csv" / f"{kind}_B_train.csv" for kind in FAMILIES]
    input_hashes = {str(path): digest(path.read_bytes()) for path in input_paths}
    proof = json.loads((base / "ABC_csv/verification.json").read_text())
    assert proof["status"] == "PASS" and not proof["llm_used"]
    for kind in FAMILIES:
        path = base / "ABC_csv" / f"{kind}_B_train.csv"
        assert input_hashes[str(path)] == proof["output_csv_sha256"][path.name]
    original_bytes, train = read_captions(args.captions)
    _, originals = read_captions(baseline)
    _, val = read_captions(root / "tvr/TextData/tvrval.caption.txt")
    _, test = read_captions(root / "tvr/TextData/tvrtest.caption.txt")
    assert train.keys() == originals.keys()
    dictionaries = {kind: json.loads((root / f"tvr/TextData/tvr_id_{kind}_dict.json").read_text())
                    for kind in FAMILIES}
    vocab = {kind: dictionary_vocabulary(data) for kind, data in dictionaries.items()}
    timestamps = source_timestamps(root, originals)
    nltk.data.path.insert(0, str(root / "rewriting/nlp_resources"))
    from nltk.corpus import wordnet as wn

    spacy.util.fix_random_seed(2026)
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    mappings, mapping_counts = defaultdict(list), Counter()
    for kind, families in FAMILIES.items():
        families = [set(family.split()) for family in families]
        for row in csv_rows(base / "ABC_csv" / f"{kind}_B_train.csv"):
            reason = "usable"
            if row["mapping_type"] != "WORDNET_SYNONYM":
                reason = "reference_without_replacement"
            elif row["train_pos"] != row["val_pos"] or row["replacement_expression"] != row["val_expression"]:
                reason = "observed_val_form_or_pos_mismatch"
            elif not any({row["train_lemma"], row["val_lemma"]} <= family for family in families):
                reason = "outside_existing_conservative_families"
            elif row["train_expression"] == row["val_expression"]:
                reason = "identical_expression"
            mapping_counts[kind + "/" + reason] += 1
            if reason != "usable":
                continue
            for field in ["train", "val"]:
                assert row[f"{field}_pos"] in vocab[kind].get(row[f"{field}_expression"], {})
            assert row["example_val_caption"].strip() == val[row["example_val_caption_id"]].strip()
            row["example_val_caption"] = val[row["example_val_caption_id"]]
            pos = wn.NOUN if kind == "object" else wn.VERB
            shared = sorted({sense.name() for sense in wn.synsets(row["train_lemma"], pos=pos)} &
                            {sense.name() for sense in wn.synsets(row["val_lemma"], pos=pos)})
            assert shared
            row["shared_synsets"] = " | ".join(shared)
            mappings[kind, row["train_expression"], row["train_pos"]].append(row)
    eligible = {cid: caption for cid, caption in train.items()
                if 14 <= len(caption.split()) <= 20 and not BLOCKED.search(caption)}
    docs = dict(zip(eligible, nlp.pipe(eligible.values(), batch_size=128)))
    witness_texts = sorted({row["example_val_caption"] for rows in mappings.values() for row in rows})
    witnesses = dict(zip(witness_texts, nlp.pipe(witness_texts, batch_size=128)))
    print(f"Parsed {len(eligible):,} eligible current Train captions; {sum(map(len, mappings.values()))} mappings", flush=True)
    audit, proposed = [], []
    for cid, doc in docs.items():
        baseline_tokens = nlp.make_doc(originals[cid])
        aligned = len(doc) == len(baseline_tokens)
        for token in doc:
            kind = "object" if token.pos_ == "NOUN" else "action" if token.pos_ == "VERB" else ""
            for row in mappings.get((kind, token.text, token.tag_), []):
                target = row["val_expression"]
                entry = dict(source_caption_id=cid, term_type=kind, source_expression=token.text,
                             replacement_expression=target, mapping_id=row["mapping_id"],
                             source_span_start=token.idx, status="", new_caption_id="")
                audit.append(entry)
                annotation = token.text
                basis = "original_id_annotation"
                if dictionaries[kind][cid].get(annotation) != token.tag_:
                    annotation = baseline_tokens[token.i].text if aligned else ""
                    basis = "projected_previous_substitution"
                    if dictionaries[kind][cid].get(annotation) != token.tag_ or annotation == token.text:
                        entry["status"] = "no_caption_specific_dictionary_evidence"
                        continue
                witness_tokens = [t for t in witnesses[row["example_val_caption"]]
                                  if t.text == target and t.tag_ == row["val_pos"] and t.pos_ == token.pos_]
                status = "" if witness_tokens else "val_witness_context_pos_mismatch"
                if not status and kind == "action" and any(
                        c.dep_ == "prt" for t in [token, *witness_tokens] for c in t.children):
                    status = "phrasal_verb_preserved"
                if not status:
                    status = syntax_status([token], kind, row["train_lemma"])
                if not status:
                    status = syntax_status(witness_tokens, kind, row["val_lemma"])
                if not status and any(t.pos_ == "VERB" and t.lemma_ in NONVISUAL for t in doc):
                    status = "nonvisual_caption_preserved"
                if not status and any(t.lemma_ in SPEECH and any(c.dep_ == "ccomp" for c in t.children) for t in doc):
                    status = "dialogue_content_preserved"
                if not status and token.i and doc[token.i - 1].lower_ in {"a", "an"} and (
                        (doc[token.i - 1].lower_ == "an") != (target[0].lower() in "aeiou")):
                    status = "article_mismatch"
                if not status and cid not in timestamps:
                    status = "unresolved_original_timestamp"
                caption = doc.text[:token.idx] + target + doc.text[token.idx + len(token.text):]
                words, tokens = len(caption.split()), len(tokenizer.encode(caption, add_special_tokens=True))
                if not status and not (14 <= words <= 20 and tokens < 30):
                    status = "length_budget_exceeded"
                if status:
                    entry["status"] = status
                    continue
                proposed.append((entry, row, annotation, basis, caption, words, tokens))
    rewritten_docs = nlp.pipe((item[4] for item in proposed), batch_size=128)
    seen = {(cid.rsplit("#enc#", 1)[0], text): cid for cid, text in train.items()}
    previous_queries = {(cid.rsplit("#enc#", 1)[0], text) for cid, text in originals.items()}
    all_ids = train.keys() | val.keys() | test.keys()
    next_enc = defaultdict(int)
    for cid in all_ids:
        video, number = cid.rsplit("#enc#", 1)
        next_enc[video] = max(next_enc[video], int(number) + 1)
    additions = []
    for item, doc in zip(proposed, rewritten_docs):
        entry, row, annotation, basis, caption, words, tokens = item
        cid, kind, start = entry["source_caption_id"], entry["term_type"], entry["source_span_start"]
        source_token = next(t for t in docs[cid] if t.idx == start)
        target_token = next((t for t in doc if t.idx == start), None)
        if target_token is None or target_token.text != row["val_expression"] or (
                target_token.pos_, target_token.tag_, target_token.dep_) != (
                source_token.pos_, source_token.tag_, source_token.dep_):
            entry["status"] = "rewritten_context_pos_or_dependency_mismatch"
            continue
        video = cid.rsplit("#enc#", 1)[0]
        if (video, caption) in seen or (video, caption) in previous_queries:
            entry["status"] = "duplicate_pre_replacement_query_for_video" if (
                video, caption) in previous_queries else "duplicate_query_for_video"
            continue
        new_id = f"{video}#enc#{next_enc[video]}"
        next_enc[video] += 1
        seen[video, caption] = new_id
        entry.update(status="generated", new_caption_id=new_id)
        additions.append(dict(new_caption_id=new_id, source_caption_id=cid, video_id=video,
            source_timestamp=json.dumps(timestamps[cid]), term_type=kind, source_expression=source_token.text,
            replacement_expression=target_token.text, source_pos=source_token.tag_, replacement_pos=target_token.tag_,
            source_annotation_expression=annotation, source_annotation_pos=source_token.tag_,
            source_dictionary_caption_id=vocab[kind][source_token.text][source_token.tag_],
            replacement_dictionary_caption_id=vocab[kind][target_token.text][target_token.tag_],
            source_annotation_basis=basis, source_span_start=start, source_span_end=start + len(source_token.text),
            mapping_id=row["mapping_id"], mapping_csv=f"ABC_csv/{kind}_B_train.csv",
            val_caption_id=row["example_val_caption_id"], val_caption=row["example_val_caption"],
            original_caption=train[cid], additional_caption=caption, word_count=words, roberta_token_count=tokens,
            shared_wordnet_synsets=row["shared_synsets"], meaning=row["meaning"], requires_video_review=True,
            **{field: row[field] for field in WORDNET_FIELDS}))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "additional_queries.csv", FIELDS, additions)
    for kind in FAMILIES:
        write_csv(out / f"{kind}_additional_queries.csv", FIELDS, (r for r in additions if r["term_type"] == kind))
    write_csv(out / "candidate_audit.csv", AUDIT_FIELDS, audit)
    text = "".join(f"{r['new_caption_id']} {r['additional_caption']}\n" for r in additions)
    (out / "tvrtrain.additional.caption.txt").write_text(text, encoding="utf-8")
    separator = b"" if original_bytes.endswith(b"\n") else b"\n"
    merged = original_bytes + separator + text.encode("utf-8")
    (out / "tvrtrain.caption.with_additional.txt").write_bytes(merged)
    with (out / "additional_queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in additions:
            handle.write(json.dumps({**row, "source_timestamp": json.loads(row["source_timestamp"])},
                                    ensure_ascii=False) + "\n")
    for row in csv_rows(out / "additional_queries.csv"):
        cid = row["source_caption_id"]
        start, end = int(row["source_span_start"]), int(row["source_span_end"])
        assert train[cid][start:end] == row["source_expression"]
        assert row["additional_caption"] == train[cid][:start] + row["replacement_expression"] + train[cid][end:]
        assert row["source_annotation_pos"] == dictionaries[row["term_type"]][cid][row["source_annotation_expression"]]
        assert json.loads(row["source_timestamp"]) == timestamps[cid]
        assert row["new_caption_id"] not in all_ids
        assert (row["video_id"], row["additional_caption"]) not in previous_queries
        assert row["additional_caption"] not in {caption for source_id, caption in train.items()
                                                 if source_id.rsplit("#enc#", 1)[0] == row["video_id"]}
        assert 14 <= len(row["additional_caption"].split()) <= 20
        assert len(tokenizer.encode(row["additional_caption"], add_special_tokens=True)) < 30
    _, saved = read_captions(out / "tvrtrain.additional.caption.txt") if additions else (b"", {})
    _, combined = read_captions(out / "tvrtrain.caption.with_additional.txt")
    assert len(saved) == len(additions) and len(combined) == len(train) + len(additions)
    assert all(combined[cid] == caption for cid, caption in train.items())
    assert merged.startswith(original_bytes)
    assert all(digest(Path(path).read_bytes()) == value for path, value in input_hashes.items())
    summary = dict(status="PASS", llm_used=False, seed=2026, input_queries=len(train),
        eligible_current_queries=len(eligible), generated_queries=len(additions),
        source_queries=len({r["source_caption_id"] for r in additions}),
        generated_by_aspect=dict(Counter(r["term_type"] for r in additions)),
        original_annotation_basis=dict(Counter(r["source_annotation_basis"] for r in additions)),
        unresolved_source_timestamp_queries=len(train) - len(timestamps),
        timestamp_policy="Unique timestamp for the exact video/baseline-caption pair in tvr_train_release.jsonl",
        merged_queries=len(combined), mapping_filter_counts=dict(mapping_counts),
        candidate_status_counts=dict(Counter(r["status"] for r in audit)),
        policy="One observed Train-to-Val same-Penn-POS WordNet synonym per query; existing conservative families and syntax guards; 14-20 words, <30 RoBERTa tokens including special tokens.",
        novelty_policy="Exclude existing current and pre-replacement baseline queries for the same video, plus generated duplicates",
        dictionary_scope_changed=False, original_files_unchanged=True, original_train_bytes_preserved=True,
        video_ids_and_source_timestamps_preserved=True, video_grounding_verified=False,
        tokenizer=str(TOKENIZER), spacy_model="en_core_web_md", spacy_version=spacy.__version__,
        wordnet_version=wn.get_version(), input_sha256=input_hashes,
        script_sha256={Path(__file__).name: digest(Path(__file__).read_bytes()),
                       "select_diagram_synonyms.py": digest(Path(__file__).with_name("select_diagram_synonyms.py").read_bytes())},
        output_sha256={path.name: digest(path.read_bytes()) for path in sorted(out.iterdir()) if path.suffix in {".csv", ".txt", ".jsonl"}})
    (out / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if "sha256" not in k}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
