"""Prepare, encode and select rank-one Object/Action queries with a frozen teacher.

All outputs are isolated. The Train captions, old candidate CSVs, features and
checkpoint are never overwritten. Selected captions remain provisional until
their visual/event equivalence is reviewed.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Holmes"))

from generate_additional_train_queries import file_sha256, source_timestamps
from select_diagram_synonyms import read_captions
from multi_query_candidate_validation import validate_candidate
from multi_query_subset_selection import select_subset


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def batches(items, size):
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def csv_writer(stack, path, fields):
    handle = stack.enter_context(Path(path).open("w", encoding="utf-8-sig", newline=""))
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return writer


def verify_rank_one_inputs(args):
    raw_dir = args.package_dir / "additional_captions_rank1"
    proof = json.loads((raw_dir / "verification.json").read_text())
    if proof["status"] != "PASS" or not proof["rank_one_only"] or proof["llm_used"]:
        raise ValueError("Expected verified no-LLM rank-one inputs")
    expected_links = {name for name in proof["output_sha256"] if name.startswith("links/generation_links_")}
    actual_links = {str(path.relative_to(raw_dir)) for path in (raw_dir / "links").glob("generation_links_*.csv")}
    if expected_links != actual_links:
        raise ValueError("Generation link parts are missing or unexpected")
    for name, expected in proof["output_sha256"].items():
        if file_sha256(raw_dir / name) != expected:
            raise ValueError(f"Rank-one output changed: {name}")
    for path, expected in proof["input_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Rank-one provenance input changed: {path}")
    return proof


def prepare(args):
    import h5py
    import spacy
    from transformers import RobertaTokenizer

    out, base = args.output_dir, args.package_dir
    if (out / "prepared.json").exists():
        raise FileExistsError("Preparation already exists; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = base / "additional_captions_rank1"
    _, train = read_captions(args.captions)
    _, val = read_captions(ROOT / "tvr/TextData/tvrval.caption.txt")
    _, old_added = read_captions(raw_dir / "tvrtrain.additional.caption.txt")
    proof = verify_rank_one_inputs(args)
    if not proof["rank_one_only"] or proof["llm_used"]:
        raise ValueError("Expected the existing no-LLM rank-one input")
    if file_sha256(args.captions) != proof["input_sha256"][str(args.captions)]:
        raise ValueError("Current Train text differs from the rank-one generation source")

    pair_rows = list(csv_rows(raw_dir / "replacement_pairs.csv"))
    mapping = {}
    earliest = {}
    input_paths = [args.captions, ROOT / "tvr/TextData/tvrval.caption.txt",
                   ROOT / "tvr/TextData/tvrtest.caption.txt", raw_dir / "verification.json",
                   raw_dir / "replacement_pairs.csv", raw_dir / "tvrtrain.additional.caption.txt"]
    for kind in ("object", "action"):
        mapped_path = base / "ABC_csv" / f"{kind}_B_train.csv"
        link_path = base / "mapping/links" / f"{kind}_B_train.csv"
        input_paths += [mapped_path, link_path]
        mapping.update({(kind, row["mapping_id"]): row for row in csv_rows(mapped_path)})
        for link in csv_rows(link_path):
            if link["synonym_rank"] == "1":
                key = kind, link["mapping_id"]
                earliest[key] = min(earliest.get(key, float("inf")),
                                    int(link["source_data_row_number"]))
    pairs = {}
    for pair in pair_rows:
        kind = pair["term_type"]
        map_id = min(pair["mapping_ids"].split(" | "),
                     key=lambda mid: (earliest[kind, mid], int(mid)))
        row = mapping[kind, map_id]
        if row["train_expression"] != pair["source_expression"]:
            raise ValueError("Source expression differs from its mapping")
        actual_witness = val.get(row["example_val_caption_id"])
        if actual_witness is None or actual_witness.strip() != row["example_val_caption"].strip():
            raise ValueError("Val witness text has changed")
        pairs[pair["pair_id"]] = {**pair, "mapping_id": map_id,
            "val_expression": row["val_expression"], "source_pos": row["train_pos"],
            "val_pos": row["val_pos"], "val_caption_id": row["example_val_caption_id"],
            "val_caption": actual_witness}

    grouped = defaultdict(list)
    raw_count = Counter()
    for path in sorted((raw_dir / "links").glob("generation_links_*.csv")):
        input_paths.append(path)
        for link in csv_rows(path):
            cid, old_id = link["source_caption_id"], link["new_caption_id"]
            if cid not in train or old_id not in old_added:
                raise ValueError("Generation link references an unknown source/candidate")
            pair = pairs[link["pair_id"]]
            start, end = int(link["source_span_start"]), int(link["source_span_end"])
            source = train[cid]
            if source[start:end] != pair["source_expression"]:
                raise ValueError("Generation link span differs from the current original")
            raw_caption = source[:start] + pair["replacement_expression"] + source[end:]
            if raw_caption != old_added[old_id] or cid.split("#")[0] != old_id.split("#")[0]:
                raise ValueError("Generation link does not reconstruct its raw candidate")
            grouped[cid].append(link)
            raw_count[cid] += 1

    next_enc = defaultdict(int)
    # New research IDs cannot accidentally refer to old, differently worded features.
    with h5py.File(args.base_features, "r") as features:
        for cid in features.keys():
            video, number = cid.rsplit("#enc#", 1)
            next_enc[video] = max(next_enc[video], int(number) + 1)
    _, test = read_captions(ROOT / "tvr/TextData/tvrtest.caption.txt")
    for cid in train.keys() | val.keys() | test.keys() | old_added.keys():
        video, number = cid.rsplit("#enc#", 1)
        next_enc[video] = max(next_enc[video], int(number) + 1)
    baseline = ROOT / "rewriting/val_domain_analysis/applied_diagram_synonyms_20261007/tvrtrain.caption.original.txt"
    _, original_before_edits = read_captions(baseline)
    timestamps = source_timestamps(ROOT, original_before_edits)
    input_paths += [baseline, ROOT / "tvr/TextData/tvr_train_release.jsonl"]
    hashes = {str(path): file_sha256(path) for path in input_paths}

    spacy.util.fix_random_seed(args.seed)
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta, local_files_only=True,
                                                do_lower_case=False)
    witness_texts = sorted({pair["val_caption"] for pair in pairs.values()})
    witness_docs = dict(zip(witness_texts, nlp.pipe(witness_texts, batch_size=256)))
    counts, reasons = Counter(), Counter()
    eligible_texts, seen_texts = {}, {}
    audit_fields = None
    eligible_fields = ["source_caption_id", "new_caption_id", "previous_candidate_id", "term_type",
        "pair_id", "mapping_id", "source_expression", "replacement_expression", "source_pos", "val_pos",
        "source_span_start", "source_span_end", "val_caption_id", "val_caption", "source_timestamp",
        "timestamp_status", "original_caption", "additional_caption", "visual_grounding_status",
        "event_preservation_status", "text_validation_status", "synonym_rank"]
    started = time.monotonic()
    with ExitStack() as stack:
        eligible_writer = csv_writer(stack, out / "eligible_candidate_links.csv", eligible_fields)
        docs = nlp.pipe(((text, cid) for cid, text in train.items()), as_tuples=True,
                        batch_size=256, n_process=args.nlp_processes)
        processed = 0
        for source_batch in batches(docs, 128):
            pending, parse_texts = [], set()
            for source_doc, cid in source_batch:
                for link in grouped[cid]:
                    pair = pairs[link["pair_id"]]
                    start, end = int(link["source_span_start"]), int(link["source_span_end"])
                    target = pair["val_expression"]
                    rewritten = source_doc.text[:start] + target + source_doc.text[end:]
                    span = source_doc.char_span(start, end, alignment_mode="strict")
                    expected = "NOUN" if pair["term_type"] == "object" else "VERB"
                    prescreen = span is not None and len(span) == 1 and span[0].pos_ == expected
                    if prescreen and rewritten != source_doc.text:
                        parse_texts.add(rewritten)
                    pending.append((source_doc, cid, link, pair, rewritten, prescreen))
            ordered_texts = sorted(parse_texts)
            candidate_docs = dict(zip(ordered_texts, nlp.pipe(ordered_texts, batch_size=256)))
            for source_doc, cid, link, pair, rewritten, prescreen in pending:
                new_doc = candidate_docs.get(rewritten)
                if new_doc is None:
                    new_doc = nlp.make_doc(rewritten)
                validation = validate_candidate(source_doc, new_doc, witness_docs[pair["val_caption"]],
                    term_type=pair["term_type"], source_span_start=link["source_span_start"],
                    source_span_end=link["source_span_end"], source_expression=pair["source_expression"],
                    val_expression=pair["val_expression"], source_pos=pair["source_pos"],
                    val_pos=pair["val_pos"], timestamp_status="resolved_from_release" if cid in timestamps
                    else "unresolved_kept", tokenizer=tokenizer)
                if args.length_filter and validation["text_validation_status"] == "eligible" and not (
                    validation["candidate_within_14_20_words"] and validation["candidate_under_30_subword_tokens"]):
                    validation.update(text_validation_status="rejected", validation_status="rejected",
                                      validation_reason="requested_length_budget")
                validation["candidate_length_filter_applied"] = args.length_filter
                row = {"source_caption_id": cid, "previous_candidate_id": link["new_caption_id"],
                    "new_caption_id": "", "term_type": pair["term_type"], "pair_id": pair["pair_id"],
                    "mapping_id": pair["mapping_id"], "source_expression": pair["source_expression"],
                    "raw_replacement_expression": pair["replacement_expression"],
                    "replacement_expression": pair["val_expression"], "source_pos": pair["source_pos"],
                    "val_pos": pair["val_pos"], "source_span_start": link["source_span_start"],
                    "source_span_end": link["source_span_end"], "val_caption_id": pair["val_caption_id"],
                    "val_caption": pair["val_caption"], "source_timestamp": json.dumps(timestamps[cid])
                    if cid in timestamps else "", "timestamp_status": "resolved_from_release"
                    if cid in timestamps else "unresolved_kept", "original_caption": source_doc.text,
                    "additional_caption": rewritten, "synonym_rank": 1,
                    "validation_stage": "full_text_checks" if prescreen else "source_pos_prescreen",
                    **validation}
                status = validation["text_validation_status"]
                counts[status] += 1
                counts[pair["term_type"] + "/" + status] += 1
                reasons[validation["validation_reason"]] += 1
                if status == "eligible":
                    video = cid.split("#", 1)[0]
                    key = video, rewritten
                    new_id = seen_texts.get(key)
                    if new_id is None:
                        new_id = f"{video}#enc#{next_enc[video]}"
                        next_enc[video] += 1
                        seen_texts[key] = new_id
                        eligible_texts[new_id] = rewritten
                    row["new_caption_id"] = new_id
                    eligible_writer.writerow(row)
                if audit_fields is None:
                    audit_fields = list(row)
                    audit_writer = csv_writer(stack, out / "candidate_validation.csv", audit_fields)
                audit_writer.writerow(row)
            processed += len(source_batch)
            if processed % 1024 == 0 or processed == len(train):
                print(f"Validated {processed}/{len(train)} originals; eligible links={counts['eligible']}, "
                      f"unique={len(eligible_texts)}; elapsed={time.monotonic()-started:.1f}s", flush=True)
        with (out / "eligible_candidates.caption.txt").open("w") as handle:
            for cid, text in eligible_texts.items():
                handle.write(f"{cid} {text}\n")
        with (out / "scoring.caption.txt").open("w") as handle:
            for cid, text in {**train, **eligible_texts}.items():
                handle.write(f"{cid} {text}\n")
        raw_writer = csv_writer(stack, out / "source_candidate_counts.csv",
                               ["source_caption_id", "raw_candidate_count"])
        for cid in train:
            raw_writer.writerow({"source_caption_id": cid, "raw_candidate_count": raw_count[cid]})
    result = dict(status="PREPARED", input_queries=len(train), input_unique_candidates=len(old_added),
        input_candidate_links=sum(raw_count.values()), text_validation_counts=dict(counts),
        validation_reasons=dict(reasons), unique_eligible_queries=len(eligible_texts),
        train_captions=str(args.captions), roberta=str(args.roberta), seed=args.seed,
        llm_used=False, length_filter=args.length_filter, rank_one_only=True,
        replacement_policy="earliest original rank-one witness val_expression; no generated inflections",
        visual_grounding_status="not_verified", input_sha256=hashes,
        prepared_output_sha256={name: file_sha256(out / name) for name in (
            "candidate_validation.csv", "eligible_candidate_links.csv", "eligible_candidates.caption.txt",
            "scoring.caption.txt", "source_candidate_counts.csv")})
    if sum(raw_count.values()) != sum(counts[key] for key in ("eligible", "rejected", "review_required")):
        raise ValueError("Candidate audit lost generation links")
    if (sum(raw_count.values()) != proof["counts"]["candidate_occurrences"]
            or len(old_added) != proof["generated_queries"]):
        raise ValueError("Candidate count differs from the verified rank-one input")
    json_write(out / "prepared.json", result)
    print(json.dumps({k: v for k, v in result.items() if "sha256" not in k}, ensure_ascii=False, indent=2))


def extract_features(args):
    import h5py
    import numpy as np
    import torch
    from utils.extract_tvr_roberta_features import extract

    out = args.output_dir
    prepared = json.loads((out / "prepared.json").read_text())
    input_path = out / "scoring.caption.txt"
    if file_sha256(input_path) != prepared["prepared_output_sha256"][input_path.name]:
        raise ValueError("Prepared captions changed")
    features = out / "query_features.hdf5"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    extract(argparse.Namespace(caption_file=[input_path], output=features, model=str(args.roberta),
        batch_size=args.extraction_batch, max_length=30, layer=-2, expected_dim=768,
        device=args.device, normalize=False, keep_special_tokens=False, overwrite=False))
    _, texts = read_captions(input_path)
    with h5py.File(features, "r+") as handle:
        for cid, text in texts.items():
            handle[cid].attrs["caption_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        handle.attrs["input_caption_file_sha256"] = file_sha256(input_path)
        handle.attrs["seed"] = args.seed
    json_write(out / "feature_manifest.json", dict(status="PASS", query_count=len(texts),
        input_caption_file_sha256=file_sha256(input_path), features=str(features),
        features_sha256=file_sha256(features), model=str(args.roberta), hidden_state_layer=-2,
        max_length_including_special_tokens=30, special_tokens_removed=True,
        l2_normalized=False, dataset_caption_hashes=True, llm_used=False))


def review_prepared_candidates(args):
    """Recheck only first-pass eligible texts; preserve every audited input link."""
    import spacy
    from multi_query_candidate_validation import strict_post_review

    out = args.output_dir
    prepared = json.loads((out / "prepared.json").read_text())
    if prepared.get("strict_post_review_applied"):
        raise ValueError("Strict post-review has already been applied")
    for name, expected in prepared["prepared_output_sha256"].items():
        if file_sha256(out / name) != expected:
            raise ValueError(f"Prepared output changed: {name}")
    audit = out / "candidate_validation.csv"
    eligible = [row for row in csv_rows(audit) if row["text_validation_status"] == "eligible"]
    texts = sorted({row[key] for row in eligible for key in (
        "original_caption", "additional_caption")})
    spacy.util.fix_random_seed(args.seed)
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    docs = dict(zip(texts, nlp.pipe(texts, batch_size=256, n_process=args.nlp_processes)))
    decisions = {}
    for row in eligible:
        key = (row["source_caption_id"], row["previous_candidate_id"], row["pair_id"],
               row["source_span_start"], row["source_span_end"])
        decisions[key] = strict_post_review(docs[row["original_caption"]], docs[row["additional_caption"]],
            source_span_start=row["source_span_start"], source_span_end=row["source_span_end"],
            val_expression=row["replacement_expression"], term_type=row["term_type"], validation=row)
    with audit.open(encoding="utf-8-sig", newline="") as handle:
        fields = next(csv.reader(handle))
    extra_fields = sorted(set().union(*(set(row) for row in decisions.values())) - set(fields))
    fields += extra_fields
    with (out / "eligible_candidate_links.csv").open(encoding="utf-8-sig", newline="") as handle:
        eligible_fields = next(csv.reader(handle))
    counts, reasons, new_texts = Counter(), Counter(), {}
    temporary_audit = audit.with_suffix(".review.tmp")
    temporary_links = out / "eligible_candidate_links.review.tmp"
    with ExitStack() as stack:
        writer = csv_writer(stack, temporary_audit, fields)
        link_writer = csv_writer(stack, temporary_links, eligible_fields)
        for row in csv_rows(audit):
            key = (row["source_caption_id"], row["previous_candidate_id"], row["pair_id"],
                   row["source_span_start"], row["source_span_end"])
            if key in decisions:
                row.update(decisions[key])
            status = row["text_validation_status"]
            counts[status] += 1
            counts[row["term_type"] + "/" + status] += 1
            reasons[row["validation_reason"]] += 1
            if status == "eligible":
                link_writer.writerow(row)
                new_texts[row["new_caption_id"]] = row["additional_caption"]
            writer.writerow(row)
    temporary_audit.replace(audit)
    temporary_links.replace(out / "eligible_candidate_links.csv")
    _, train = read_captions(Path(prepared["train_captions"]))
    with (out / "eligible_candidates.caption.txt").open("w") as handle:
        for cid, text in new_texts.items():
            handle.write(f"{cid} {text}\n")
    with (out / "scoring.caption.txt").open("w") as handle:
        for cid, text in {**train, **new_texts}.items():
            handle.write(f"{cid} {text}\n")
    assert sum(counts[key] for key in ("eligible", "rejected", "review_required")) == prepared["input_candidate_links"]
    prepared.update(first_pass_eligible_links=len(eligible), first_pass_eligible_unique_queries=prepared["unique_eligible_queries"],
        strict_post_review_applied=True, text_validation_counts=dict(counts), validation_reasons=dict(reasons),
        unique_eligible_queries=len(new_texts),
        prepared_output_sha256={name: file_sha256(out / name) for name in prepared["prepared_output_sha256"]})
    json_write(out / "prepared.json", prepared)
    print(json.dumps({key: value for key, value in prepared.items() if "sha256" not in key}, ensure_ascii=False, indent=2))


def run_selection(args):
    import h5py
    import numpy as np
    from multi_query_native_scoring import NativeScorer

    out = args.output_dir
    prepared = json.loads((out / "prepared.json").read_text())
    if not prepared.get("strict_post_review_applied"):
        raise ValueError("Run the strict post-review phase before scoring")
    features_proof = json.loads((out / "feature_manifest.json").read_text())
    for name, expected in prepared["prepared_output_sha256"].items():
        if file_sha256(out / name) != expected:
            raise ValueError(f"Prepared output changed: {name}")
    input_proof = verify_rank_one_inputs(args)
    if (prepared["input_candidate_links"] != input_proof["counts"]["candidate_occurrences"]
            or prepared["input_unique_candidates"] != input_proof["generated_queries"]):
        raise ValueError("Prepared data omitted rank-one candidate sources")
    original_bytes, train = read_captions(Path(prepared["train_captions"]))
    if file_sha256(Path(prepared["train_captions"])) != prepared["input_sha256"][prepared["train_captions"]]:
        raise ValueError("Train captions changed after preparation")
    _, eligible_texts = read_captions(out / "eligible_candidates.caption.txt") if prepared["unique_eligible_queries"] else (b"", {})
    grouped = defaultdict(dict)
    for row in csv_rows(out / "eligible_candidate_links.csv"):
        grouped[row["source_caption_id"]].setdefault(row["new_caption_id"], []).append(row)
    raw_counts = {row["source_caption_id"]: int(row["raw_candidate_count"])
                  for row in csv_rows(out / "source_candidate_counts.csv")}
    videos = list(dict.fromkeys(cid.split("#", 1)[0] for cid in train))
    video_index = {video: i for i, video in enumerate(videos)}
    feature_path = out / "query_features.hdf5"
    if file_sha256(feature_path) != features_proof["features_sha256"]:
        raise ValueError("Fresh feature file changed")
    with h5py.File(feature_path, "r") as handle:
        for cid, text in {**train, **eligible_texts}.items():
            if handle[cid].attrs.get("caption_sha256") != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError(f"Feature/caption mismatch: {cid}")
    scorer = NativeScorer(args.checkpoint, feature_path, ROOT, device=args.device,
        video_ids=videos, query_batch=args.query_batch, video_batch=args.video_batch,
        context_batch=args.context_batch, cache_path=out / "encoded_contexts.pt", seed=args.seed)
    audit_fields = ["source_caption_id", "original_caption", "raw_candidate_count", "eligible_candidate_count",
        "score_status", "selected_candidate_count", "selected_candidate_ids", "selected_aspects",
        "baseline_margin", "selected_margin", "margin_gain", "objective", "baseline_positive_score",
        "selected_positive_score", "baseline_hardest_negative_video_id", "selected_hardest_negative_video_id",
        "original_uncertainty", "baseline_rank", "selected_rank", "evaluated_single_count",
        "evaluated_pair_count", "removal_margin_gains", "duplicate_margin_gains", "selection_reason",
        "visual_grounding_status", "label_aware_train_selection"]
    subset_fields = ["source_caption_id", "candidate_ids", "candidate_count", "margin", "margin_gain",
        "objective", "positive_score", "hardest_negative_video_id", "hardest_negative_score", "eligible",
        "selected", "rejection_reasons", "removal_margin_gains", "duplicate_margin_gains",
        "duplicate_controls", "complementary_margin_gain", "best_constituent_single_margin"]
    selected_link_fields = ["source_caption_id", "new_caption_id", "previous_candidate_id", "term_type",
        "source_expression", "replacement_expression", "mapping_id", "val_caption_id", "source_span_start",
        "source_span_end", "original_caption", "additional_caption", "source_timestamp", "timestamp_status",
        "selection_margin_gain", "visual_grounding_status", "text_validation_status"]
    selected_ids, rows_by_source, counts = set(), {}, Counter()
    active = [cid for cid in train if grouped[cid]]
    started = time.monotonic()
    with ExitStack() as stack:
        subset_writer = csv_writer(stack, out / "subset_evaluations.csv", subset_fields)
        link_writer = csv_writer(stack, out / "selected_query_links.csv", selected_link_fields)
        for source_batch in batches(active, args.source_batch):
            query_ids = list(dict.fromkeys(qid for cid in source_batch for qid in [cid, *sorted(grouped[cid])]))
            scores, uncertainties = scorer.score(query_ids)
            positions = {cid: i for i, cid in enumerate(query_ids)}
            for cid in source_batch:
                candidates = sorted(grouped[cid])
                base_scores = scores[positions[cid]]
                candidate_scores = scores[[positions[qid] for qid in candidates]]
                positive = video_index[cid.split("#", 1)[0]]
                result = select_subset(base_scores, candidate_scores, positive, candidates, videos,
                    original_query_id=cid, lambda_penalty=args.lambda_penalty, epsilon=args.epsilon,
                    require_duplicate_gain=True)
                chosen = result["selected"]
                combined = (base_scores.astype(np.float64) + sum((candidate_scores[i].astype(np.float64)
                    for i in result["selected_candidate_indices"]), np.zeros(len(videos)))) / (
                    1 + result["selected_candidate_count"])
                # Stable rank is the same tie policy used by the native evaluation.
                def rank(values):
                    return int(np.count_nonzero(values > values[positive]) +
                        np.count_nonzero(values[:positive] == values[positive]) + 1)
                aspects = sorted({row["term_type"] for qid in result["selected_candidate_ids"]
                                  for row in grouped[cid][qid]})
                rows_by_source[cid] = dict(source_caption_id=cid, original_caption=train[cid],
                    raw_candidate_count=raw_counts[cid], eligible_candidate_count=len(candidates),
                    score_status="full_train_gallery_scored", selected_candidate_count=result["selected_candidate_count"],
                    selected_candidate_ids=" | ".join(result["selected_candidate_ids"]), selected_aspects=" | ".join(aspects),
                    baseline_margin=result["baseline"]["margin"], selected_margin=chosen["margin"],
                    margin_gain=chosen["margin_gain"], objective=chosen["objective"],
                    baseline_positive_score=result["baseline"]["positive_score"],
                    selected_positive_score=chosen["positive_score"],
                    baseline_hardest_negative_video_id=result["baseline"]["hardest_negative_video_id"],
                    selected_hardest_negative_video_id=chosen["hardest_negative_video_id"],
                    original_uncertainty=float(uncertainties[positions[cid]]), baseline_rank=rank(base_scores),
                    selected_rank=rank(combined), evaluated_single_count=result["evaluated_single_count"],
                    evaluated_pair_count=result["evaluated_pair_count"],
                    removal_margin_gains=json.dumps(chosen.get("removal_margin_gains", [])),
                    duplicate_margin_gains=json.dumps(chosen.get("duplicate_margin_gains", [])),
                    selection_reason=result["selection_reason"], visual_grounding_status="not_verified",
                    label_aware_train_selection=True)
                for evaluation in result["evaluations"]:
                    subset_writer.writerow({"source_caption_id": cid, **{key: json.dumps(value) if isinstance(value, list)
                        else value for key, value in evaluation.items()}})
                for qid in result["selected_candidate_ids"]:
                    selected_ids.add(qid)
                    for link in grouped[cid][qid]:
                        link_writer.writerow({**link, "selection_margin_gain": chosen["margin_gain"]})
                        counts["selected_links/" + link["term_type"]] += 1
                counts["sources_with_" + str(result["selected_candidate_count"]) + "_added_queries"] += 1
                counts["evaluated_singles"] += result["evaluated_single_count"]
                counts["evaluated_pairs"] += result["evaluated_pair_count"]
            print(f"Selected {len(rows_by_source)}/{len(active)} scored sources; unique selected={len(selected_ids)}; "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        writer = csv_writer(stack, out / "selection_audit.csv", audit_fields)
        for cid, original in train.items():
            row = rows_by_source.get(cid)
            if row is None:
                row = dict(source_caption_id=cid, original_caption=original, raw_candidate_count=raw_counts[cid],
                    eligible_candidate_count=0, score_status="not_needed_no_eligible_candidates",
                    selected_candidate_count=0, selected_candidate_ids="", evaluated_single_count=0,
                    evaluated_pair_count=0, selection_reason="no_textually_eligible_rank_one_candidate",
                    visual_grounding_status="not_verified", label_aware_train_selection=True)
                counts["sources_with_0_added_queries"] += 1
            writer.writerow(row)
    scorer.close()
    added_bytes = "".join(f"{cid} {eligible_texts[cid]}\n" for cid in eligible_texts if cid in selected_ids).encode()
    (out / "tvrtrain.additional.selected.caption.txt").write_bytes(added_bytes)
    (out / "tvrtrain.caption.selected.txt").write_bytes(original_bytes +
        (b"\n" if original_bytes and not original_bytes.endswith(b"\n") else b"") + added_bytes)
    feature_link = out / "roberta_tvr_query_feat.selected.hdf5"
    _, val = read_captions(ROOT / "tvr/TextData/tvrval.caption.txt")
    _, test = read_captions(ROOT / "tvr/TextData/tvrtest.caption.txt")
    with h5py.File(feature_link, "w") as handle:
        for cid in list(train) + sorted(selected_ids):
            handle[cid] = h5py.ExternalLink(str(feature_path.resolve()), cid)
        for cid in val.keys() | test.keys():
            if cid not in handle:
                handle[cid] = h5py.ExternalLink(str(args.base_features.resolve()), cid)
        handle.attrs["manifest"] = str((out / "verification.json").resolve())
    # Keep both targets open read-only: reopening a large target for every
    # external link repeatedly discards its HDF5 metadata cache.
    with h5py.File(feature_path, "r") as fresh_target, h5py.File(args.base_features, "r") as base_target, h5py.File(feature_link, "r") as handle:
        for index, cid in enumerate(train.keys() | selected_ids | val.keys() | test.keys(), 1):
            shape = handle[cid].shape
            if len(shape) != 2 or shape[0] < 1 or shape[1] != 768:
                raise ValueError(f"Invalid linked feature: {cid}: {shape}")
            if index % 10000 == 0:
                print(f"Verified linked features {index}", flush=True)
    dataset = out / "dataset/tvr"
    (dataset / "TextData").mkdir(parents=True, exist_ok=True)
    targets = {"tvrtrain.caption.txt": out / "tvrtrain.caption.selected.txt",
               "roberta_tvr_query_feat.hdf5": feature_link,
               "tvrval.caption.txt": ROOT / "tvr/TextData/tvrval.caption.txt",
               "tvrtest.caption.txt": ROOT / "tvr/TextData/tvrtest.caption.txt"}
    for name, target in targets.items():
        link = dataset / "TextData" / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        link.symlink_to(target.resolve())
    visual_link = dataset / "FeatureData"
    if not (visual_link.is_symlink() and visual_link.resolve() == (ROOT / "tvr/FeatureData").resolve()):
        visual_link.symlink_to((ROOT / "tvr/FeatureData").resolve(), target_is_directory=True)
    _, merged = read_captions(out / "tvrtrain.caption.selected.txt")
    assert len(merged) == len(train) + len(selected_ids)
    assert all(merged[cid] == caption for cid, caption in train.items())
    assert not selected_ids & train.keys()
    assert sum(counts["sources_with_" + str(k) + "_added_queries"] for k in range(3)) == len(train)
    term_counts, selected_terms = defaultdict(Counter), Counter()
    for row in csv_rows(out / "candidate_validation.csv"):
        key = row["term_type"], row["source_expression"], row["replacement_expression"]
        term_counts[key]["input_links"] += 1
        term_counts[key][row["text_validation_status"] + "_links"] += 1
    for row in csv_rows(out / "selected_query_links.csv"):
        selected_terms[row["term_type"], row["source_expression"], row["replacement_expression"]] += 1
    with ExitStack() as stack:
        writer = csv_writer(stack, out / "term_selection_summary.csv", ["term_type", "train_expression",
            "val_expression", "input_links", "eligible_links", "rejected_links", "review_required_links",
            "selected_links"])
        for key, totals in sorted(term_counts.items()):
            writer.writerow(dict(term_type=key[0], train_expression=key[1], val_expression=key[2],
                **{field: totals[field] for field in (
                    "input_links", "eligible_links", "rejected_links", "review_required_links")},
                selected_links=selected_terms[key]))
    if any(file_sha256(Path(path)) != expected for path, expected in prepared["input_sha256"].items()):
        raise ValueError("An input changed during the run")
    summary = dict(status="PASS_STRUCTURAL_AND_SCORING", input_queries=len(train),
        input_candidate_links=prepared["input_candidate_links"], eligible_unique_queries=len(eligible_texts),
        source_queries_with_raw_candidates=sum(count > 0 for count in raw_counts.values()),
        scored_sources=len(active), unscored_sources_without_candidates=len(train)-len(active),
        selected_unique_queries=len(selected_ids), merged_queries=len(merged), gallery_videos=len(videos),
        counts=dict(counts), validation_counts=prepared["text_validation_counts"],
        checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
        teacher_training_seed=scorer.cfg.get("seed"), selection_seed=args.seed,
        lambda_penalty=args.lambda_penalty, epsilon=args.epsilon, require_duplicate_gain=True,
        original_always_included=True, max_added_queries_per_source=2, mean_scores_before_max_negative=True,
        llm_used=False, rank_one_only=True, length_filter=prepared["length_filter"],
        strict_post_review_applied=prepared["strict_post_review_applied"],
        visual_grounding_status="not_verified", selection_status="provisional_text_checked",
        evaluation_scope="in_sample_label_aware_train_teacher_selection_not_generalization",
        val_vocabulary_used=True, val_test_captions_identical=val == test,
        features_verified_against_caption_hashes=True, originals_unchanged=True,
        code_sha256={str(path): file_sha256(path) for path in (
            Path(__file__), ROOT / "rewriting/multi_query_candidate_validation.py",
            ROOT / "rewriting/multi_query_subset_selection.py", ROOT / "rewriting/multi_query_native_scoring.py")},
        scoring_manifest=scorer.manifest, fresh_feature_manifest=features_proof,
        output_sha256={name: file_sha256(out / name) for name in (
            "selection_audit.csv", "selected_query_links.csv", "subset_evaluations.csv",
            "term_selection_summary.csv",
            "tvrtrain.additional.selected.caption.txt", "tvrtrain.caption.selected.txt",
            "roberta_tvr_query_feat.selected.hdf5")})
    scored_rows = list(rows_by_source.values())
    if scored_rows:
        summary["scored_source_retrieval_diagnostics"] = {key: {
            f"R@{k}": 100 * sum(row[key + "_rank"] <= k for row in scored_rows) / len(scored_rows)
            for k in (1, 5, 10)} for key in ("baseline", "selected")}
        summary["mean_scored_source_margin_gain"] = float(np.mean([row["margin_gain"] for row in scored_rows]))
    json_write(out / "verification.json", summary)
    (out / "README.md").write_text(
        "# 원문별 Object/Action multi-query 선택 결과\n\n"
        f"원문 {len(train):,}개를 보존하고 추가 {len(selected_ids):,}개를 선택했습니다. "
        f"최종 TXT는 {len(merged):,}개입니다. 원문당 추가 후보는 최대 2개입니다.\n\n"
        "- `selection_audit.csv`: 모든 원문의 후보 수, 선택, margin, uncertainty와 사유.\n"
        "- `candidate_validation.csv`: 모든 생성 연결의 원문·Val 대응 및 텍스트 검사 상태.\n"
        "- `selected_query_links.csv`: 선택된 원문→추가 쿼리→Val 표현 연결.\n"
        "- `term_selection_summary.csv`: Train→Val 표현별 입력·통과·보류·선택 개수.\n"
        "- `subset_evaluations.csv`: 원문·모든 단독 후보·모든 후보 쌍 비교.\n"
        "- `tvrtrain.caption.selected.txt`: 학습용 원문+선택 추가 쿼리.\n"
        "- `roberta_tvr_query_feat.selected.hdf5`: 별도 추출 특징과 기존 Val 특징의 external links.\n"
        "- `dataset/`: 기존 loader를 사용할 수 있는 분리 데이터 경로.\n\n"
        "선택은 동결 teacher와 전체 Train gallery의 평균 쿼리 점수로 평가했습니다. "
        "원문 대비 개선, 후보 삭제, 동일 후보 반복 대조를 확인했습니다. "
        f"수치 허용오차 epsilon={args.epsilon}과 lambda={args.lambda_penalty}이며 성능에 맞춰 튜닝하지 않았습니다.\n\n"
        "텍스트 검사 통과는 임시 연구 후보 상태입니다. 영상·구간의 의미 보존을 확정한 결과가 아닙니다. "
        "정답을 이용한 Train 내부 선택이며, 독립 평가나 추가학습 성능을 증명하지 않습니다. "
        "Val 어휘를 사용했고 현재 Val/Test 캡션이 동일합니다.\n\n"
        "HDF5 external links와 dataset symlink가 있어 전체 출력 폴더를 유지해야 합니다. "
        "현재 원본 파일은 변경하지 않았습니다.\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if "manifest" not in key and "sha256" not in key},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "review", "extract", "score", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/multi_query_selection_20261008")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "diagram_synonym_pairs_final_20261007")
    parser.add_argument("--captions", type=Path, default=ROOT / "tvr/TextData/tvrtrain.caption.txt")
    parser.add_argument("--base-features", type=Path, default=ROOT / "tvr/TextData/roberta_tvr_query_feat.hdf5")
    parser.add_argument("--roberta", type=Path, default=ROOT / "tvr/query_only/roberta-base_tuned_model")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Holmes_ICML/tvr-multi/tvr/20261007-221616/best.ckpt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--length-filter", action="store_true")
    parser.add_argument("--nlp-processes", type=int, default=4)
    parser.add_argument("--extraction-batch", type=int, default=256)
    parser.add_argument("--query-batch", type=int, default=128)
    parser.add_argument("--video-batch", type=int, default=512)
    parser.add_argument("--context-batch", type=int, default=64)
    parser.add_argument("--source-batch", type=int, default=64)
    parser.add_argument("--lambda-penalty", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if min(args.nlp_processes, args.extraction_batch, args.query_batch, args.video_batch,
           args.context_batch, args.source_batch) < 1:
        parser.error("Batch sizes and process count must be positive")
    for phase, function in (("prepare", prepare), ("review", review_prepared_candidates),
                            ("extract", extract_features), ("score", run_selection)):
        if args.phase in (phase, "all"):
            function(args)


if __name__ == "__main__":
    main()
