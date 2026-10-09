"""Build a train-only evidence bank, then run guarded multi-query PRVR retrieval."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import h5py
import numpy as np

from utils.holmes_ar import read_captions
from .backbones import WORKSPACE, A3Retriever, HolmesRetriever, RobertaFeatures
from .core import Settings, RiskBank, InsufficientRoleBank, candidate_utility, final_scores, unit_vectors, utility
from .language import LanguageModel, LanguageRejected, lexical_rejection


def fingerprint(path):
    path = Path(path).resolve()
    if path.is_dir():
        return {p.name: fingerprint(p) for p in sorted(path.iterdir()) if p.is_file()}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def feature_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def retrieval_identity(args):
    result = dict(backbone=args.backbone, checkpoint=fingerprint(args.checkpoint),
                  roberta=fingerprint(args.roberta), layer=args.layer, max_tokens=args.max_tokens,
                  context_batch=args.context_batch)
    if args.backbone == "a3prvr":
        result.update(a3_config=fingerprint(args.a3_config), i3d=feature_identity(args.i3d_features),
                      clip=feature_identity(args.clip_features))
    else:
        result["data_root"] = str(args.data_root.resolve())
        result["visual_features"] = {str(p): feature_identity(p) for p in sorted(
            (args.data_root / "tvr/FeatureData").glob("*/feature.hdf5"))}
    return result


def source_identity():
    paths = [p for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("test_")]
    paths += list((WORKSPACE / "third_party/A3PRVR/method").rglob("*.py"))
    paths += list((WORKSPACE / "Holmes/src/Models").rglob("*.py"))
    paths += list((WORKSPACE / "Holmes/multisrc/Models").rglob("*.py"))
    paths += [WORKSPACE / "Holmes/Rewriter/evaluate.py", WORKSPACE / "Holmes/utils/holmes_ar.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(paths))).hexdigest()


def load_settings(args):
    if args.config is None:
        raise ValueError("--config is required for build-bank or --use-method; tune on dev, not test")
    return Settings(**json.loads(args.config.read_text()))


def retriever(args):
    if args.backbone == "a3prvr":
        result = A3Retriever(args.checkpoint, args.a3_config, args.i3d_features,
                             args.clip_features, args.device, args.context_batch)
        max_length = result.cfg.max_desc_l
    else:
        result = HolmesRetriever(args.checkpoint, args.backbone.removeprefix("holmes-"),
                                 args.data_root, args.device, args.context_batch)
        max_length = result.cfg["max_desc_l"]
    if (args.command == "build-bank" or args.use_method) and args.max_tokens != max_length:
        raise ValueError("Feature/rewrite token budget must equal checkpoint max_desc_l")
    return result


def save_npz(path, **values):
    with Path(path).open("xb") as stream:
        np.savez(stream, **values)


def audit_write(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def build_bank(args):
    settings = load_settings(args)
    rows = read_captions(args.train_captions)
    videos = sorted({cid.split("#", 1)[0] for cid in rows})
    for path in args.held_out_captions:
        held_out = read_captions(path)
        if set(videos) & {cid.split("#", 1)[0] for cid in held_out}:
            raise ValueError(f"Training bank overlaps held-out videos in {path}")
    if args.reference_videos > len(videos) or args.bank_queries > len(rows):
        raise ValueError("Requested train bank is larger than available data")
    if settings.top_m > args.reference_videos or args.reference_videos < 2:
        raise ValueError("Reference bank must have >=2 videos and >= top_m")
    rng = random.Random(args.seed)
    reference_videos = sorted(rng.sample(videos, args.reference_videos))
    query_ids = rng.sample(sorted(rows), args.bank_queries)
    identity = retrieval_identity(args)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(args={k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v for k, v in vars(args).items()},
                    retrieval=identity, settings=asdict(settings), source_sha256=source_identity(),
                    train=fingerprint(args.train_captions), seed=args.seed,
                    reference_videos=reference_videos, sampled_query_ids=query_ids)
    manifest["args"]["held_out_captions"] = [str(x) for x in args.held_out_captions]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    text = RobertaFeatures(args.roberta, args.device, args.max_tokens, args.layer)
    retrieval = retriever(args)
    llm = LanguageModel(args.model, args.llm_device or args.device, args.generation_tokens)
    roles, features, embeddings, evidence_ids = [], [], [], []
    with (args.output / "audit.jsonl").open("x") as audit:
        for number, cid in enumerate(query_ids, 1):
            query = rows[cid]
            try:
                graph = llm.decompose(query)
            except LanguageRejected as exc:
                audit_write(audit, dict(id=cid, original=query, accepted=False, reason=str(exc), raw_response=llm.last_response))
                print(f"[bank {number}/{len(query_ids)}] {cid}: {exc}", flush=True)
                continue
            evidence = graph["evidence"]
            vectors = text.spans(query, evidence)
            phrase_features, _ = text.encode([e["text"] for e in evidence])
            scores = retrieval.score(phrase_features, reference_videos)
            for index, (row, score, vector) in enumerate(zip(evidence, scores, vectors)):
                roles.append(row["role"])
                features.append(score)
                embeddings.append(vector)
                evidence_ids.append(f"{cid}:{index}")
            audit_write(audit, dict(id=cid, original=query, accepted=True, graph=graph))
            print(f"[bank {number}/{len(query_ids)}] {cid}: {len(evidence)} evidence", flush=True)
    if not roles:
        raise ValueError("No valid train evidence; inspect bank audit, do not fabricate a calibration")
    bank = RiskBank(roles, features, embeddings, settings)
    if not bank.calibration:
        raise ValueError("No role supports leave-one-out neighbors; increase train bank or use an explicit dev setting")
    save_npz(args.output / "bank.npz", roles=np.asarray(roles), scores=np.asarray(features),
             embeddings=np.asarray(embeddings), evidence_ids=np.asarray(evidence_ids),
             reference_videos=np.asarray(reference_videos), train_videos=np.asarray(videos),
             metadata=json.dumps(manifest))
    (args.output / "complete.json").write_text(json.dumps(dict(evidence=len(roles),
        calibrated_roles=sorted(bank.calibration), llm_calls=llm.calls, generated_tokens=llm.generated_tokens, llm_commit=llm.model_commit), indent=2))


def load_bank(args, settings):
    with np.load(args.bank, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata["args"]["model"] != args.model or metadata["args"]["generation_tokens"] != args.generation_tokens:
            raise ValueError("Bank decomposition LLM/generation settings differ")
        if metadata["retrieval"] != retrieval_identity(args):
            raise ValueError("Bank retriever/feature extractor/config differs; rebuild the bank")
        if metadata["source_sha256"] != source_identity():
            raise ValueError("Bank code version differs; rebuild to preserve reproducibility")
        bank_settings = metadata["settings"]
        for key in ("top_m", "neighbors", "evidence_temperature"):
            if bank_settings[key] != getattr(settings, key):
                raise ValueError(f"Bank calibration setting differs: {key}")
        return (RiskBank(saved["roles"], saved["scores"], saved["embeddings"], settings),
                saved["reference_videos"].tolist(), set(saved["train_videos"].tolist()), metadata)


def process_query(query, original, video_ids, bank, reference_videos, retrieval, text, llm, settings, mode):
    start = time.perf_counter()
    record = dict(original=query, gate=False, candidates=[], reason=None)
    try:
        graph = llm.decompose(query)
    except LanguageRejected as exc:
        record["reason"] = str(exc)
        record["raw_response"] = getattr(llm, "last_response", None)
        return original.copy(), record
    record["graph"] = graph
    evidence = graph["evidence"]
    vectors = text.spans(query, evidence)
    phrase_features, _ = text.encode([e["text"] for e in evidence])
    phrase_scores = retrieval.score(phrase_features, reference_videos)
    try:
        controlled = bank.analyze(evidence, phrase_scores, vectors)
    except InsufficientRoleBank as exc:
        record["reason"] = str(exc)
        record["raw_response"] = getattr(llm, "last_response", None)
        return original.copy(), record
    record["evidence"] = controlled
    record["gate"] = any(e["risk"] >= settings.risk_threshold for e in controlled)
    if not record["gate"]:
        record["reason"] = "gate_closed"
        return original.copy(), record
    try:
        candidates = llm.rewrite(query, graph, controlled, settings.candidates, text.max_length)
    except LanguageRejected as exc:
        record["reason"] = str(exc)
        record["raw_response"] = getattr(llm, "last_response", None)
        return original.copy(), record
    record["generation_candidates"] = candidates
    record["rewrite_response_schema"] = getattr(llm, "rewrite_schema", None)
    valid, judgments = [], []
    seen = {query.strip()}
    for candidate in candidates:
        reason = lexical_rejection(query, candidate, text.tokenizer, text.max_length)
        if reason is None and candidate.strip() in seen:
            reason = "duplicate_candidate"
        if isinstance(candidate, str):
            seen.add(candidate.strip())
        if reason:
            record["candidates"].append(dict(text=candidate, accepted=False, reason=reason))
            continue
        try:
            judgment = llm.judge(query, candidate, graph, controlled)
        except LanguageRejected as exc:
            record["candidates"].append(dict(text=candidate, accepted=False, reason=str(exc), raw_response=getattr(llm, "last_response", None)))
            continue
        if judgment["hard_violation"]:
            record["candidates"].append(dict(text=candidate, accepted=False, reason="semantic_hard_violation", judgment=judgment))
            continue
        valid.append(candidate)
        judgments.append(judgment)
    original_utility, original_quality = utility(original, 1, 1, 1, 0, settings)
    record.update(original_utility=original_utility, original_quality=original_quality)
    scores, utilities, indices = [], [], []
    if valid:
        # Full-context text vectors are only semantic proxies; retrieval still uses native token features.
        semantic_features, _ = text.encode([query] + valid, full_context=True)
        semantic_vectors = unit_vectors([x.mean(axis=0) for x in semantic_features])
        retrieval_features, _ = text.encode(valid)
        candidate_scores = retrieval.score(retrieval_features, video_ids)
        for candidate, score, judgment, vector in zip(valid, candidate_scores, judgments, semantic_vectors[1:]):
            semantic = float(np.clip((semantic_vectors[0] @ vector + 1) / 2, 0, 1))
            info = candidate_utility(score, semantic, judgment["preserved"], judgment["drift"],
                                     judgment["hard_violation"], controlled, settings)
            record["candidates"].append(dict(text=candidate, judgment=judgment, **info,
                                             reason=None if info["accepted"] else "protected_evidence_not_preserved"))
            if info["accepted"]:
                scores.append(score)
                utilities.append(info["utility"])
                indices.append(len(record["candidates"]) - 1)
    final, selected = final_scores(original, scores, utilities, original_utility, settings, mode=mode)
    record["selected_candidate_indices"] = [indices[i] for i in selected]
    record["reason"] = "selected" if selected else "no_improving_valid_candidate"
    record["seconds"] = time.perf_counter() - start
    return final, record


def evaluate(args):
    from Rewriter.evaluate import ranks
    all_rows = read_captions(args.captions)
    video_rows = read_captions(args.video_captions or args.captions)
    video_ids = list(dict.fromkeys(cid.split("#", 1)[0] for cid in video_rows))
    if len(video_ids) < 2 or any(cid.split("#", 1)[0] not in video_ids for cid in all_rows):
        raise ValueError("Candidate pool must contain >=2 videos and every target video")
    rows = dict(list(all_rows.items())[:args.limit]) if args.limit else all_rows
    settings = load_settings(args) if args.use_method else None
    bank = None
    if args.use_method:
        if args.bank is None:
            raise ValueError("--bank is required with --use-method")
        bank, reference_videos, train_videos, bank_metadata = load_bank(args, settings)
        if set(video_ids) & train_videos:
            raise ValueError("Evaluation video pool overlaps the train-only reference bank source")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(args={k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v for k, v in vars(args).items()},
                    retrieval=retrieval_identity(args), settings=asdict(settings) if settings else None,
                    captions=fingerprint(args.captions), video_captions=fingerprint(args.video_captions or args.captions),
                    original_features=feature_identity(args.features), source_sha256=source_identity(),
                    software=dict(numpy=np.__version__), candidate_videos=len(video_ids), queries=len(rows))
    if bank is not None:
        manifest["bank"] = fingerprint(args.bank)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    retrieval = retriever(args)
    retrieval.prepare(video_ids)
    if bank is not None:
        text = RobertaFeatures(args.roberta, args.device, args.max_tokens, args.layer)
        llm = LanguageModel(args.model, args.llm_device or args.device, args.generation_tokens)
    original, final, records = [], [], []
    with h5py.File(args.features, "r") as features, (args.output / "audit.jsonl").open("x") as audit:
        ids = list(rows)
        for offset in range(0, len(ids), args.query_batch):
            batch_ids = ids[offset:offset + args.query_batch]
            if any(cid not in features for cid in batch_ids):
                raise ValueError("Missing original query feature; no evaluation samples are skipped")
            originals = retrieval.score([features[cid][...] for cid in batch_ids], video_ids)
            for cid, score in zip(batch_ids, originals):
                start = time.perf_counter()
                if bank is not None:
                    new, record = process_query(rows[cid], score, video_ids, bank, reference_videos,
                                                retrieval, text, llm, settings, args.mode)
                else:
                    new, record = score.copy(), dict(original=rows[cid], gate=False, reason="method_off", candidates=[])
                record.update(id=cid, seconds=time.perf_counter() - start)
                original.append(score)
                final.append(new)
                records.append(record)
                audit_write(audit, record)
                print(f"[query {len(records)}/{len(rows)}] {cid}: {record['reason']}", flush=True)
    original, final = np.asarray(original), np.asarray(final)
    save_npz(args.output / "scores.npz", query_ids=np.asarray(ids), video_ids=np.asarray(video_ids),
             original=original, final=final)
    base_rank, base_top = ranks(original, video_ids, ids)
    final_rank, final_top = ranks(final, video_ids, ids)
    summary = {}
    for name, rank in (("original", base_rank), ("method", final_rank)):
        metrics = {f"R@{k}": float(100 * (rank <= k).mean()) for k in (1, 5, 10, 100)}
        summary[name] = dict(**metrics, SumR=sum(metrics.values()), mAP=float((1 / rank).mean()))
    summary.update(queries=len(rows), candidate_videos=len(video_ids),
                   limited_queries=bool(args.limit), improved=int((final_rank < base_rank).sum()),
                   worsened=int((final_rank > base_rank).sum()), unchanged=int((final_rank == base_rank).sum()),
                   lost_original_top1=int(((base_rank == 1) & (final_rank > 1)).sum()),
                   gate_open=sum(row["gate"] for row in records),
                   llm_calls=llm.calls if bank is not None else 0,
                   generated_tokens=llm.generated_tokens if bank is not None else 0,
                   llm_commit=llm.model_commit if bank is not None else None,
                   evaluation_protocol="single query; single ground-truth video; mAP equals MRR")
    with (args.output / "queries.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "original_rank", "final_rank", "original_top1", "final_top1", "gate", "reason"])
        for i, cid in enumerate(ids):
            writer.writerow([cid, base_rank[i], final_rank[i], base_top[i], final_top[i], records[i]["gate"], records[i]["reason"]])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "complete.json").write_text(json.dumps(dict(queries=len(rows), verified_execution=True)))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-bank", "run"))
    parser.add_argument("--backbone", choices=("a3prvr", "holmes-src", "holmes-multisrc"), default="a3prvr")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--a3-config", type=Path, default=WORKSPACE / "checkpoints/a3prvr/tvr_config.json")
    parser.add_argument("--i3d-features", type=Path, default=WORKSPACE / "tvr/FeatureData/i3d_resnet/feature.hdf5")
    parser.add_argument("--clip-features", type=Path, default=WORKSPACE / "tvr/FeatureData/new_clip_vit_32_tvr_vid_features.hdf5")
    parser.add_argument("--data-root", type=Path, default=WORKSPACE)
    parser.add_argument("--roberta", type=Path, default=WORKSPACE / "tvr/query_only/roberta-base_tuned_model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--llm-device", help="Optional separate device for the frozen LLM")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--layer", type=int, default=-2)
    parser.add_argument("--generation-tokens", type=int, default=768)
    parser.add_argument("--context-batch", type=int, default=64)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--train-captions", type=Path, default=WORKSPACE / "tvr/TextData/tvrtrain.caption.txt")
    parser.add_argument("--held-out-captions", type=Path, nargs="+", default=[WORKSPACE / f"tvr/TextData/tvr{s}.caption.txt" for s in ("val", "test")])
    parser.add_argument("--reference-videos", type=int, default=64)
    parser.add_argument("--bank-queries", type=int, default=128)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--video-captions", type=Path, help="Full candidate pool, unaffected by --limit")
    parser.add_argument("--features", type=Path, help="Original native token features, never regenerated or overwritten")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--use-method", action="store_true", help="Default OFF preserves original native retrieval")
    parser.add_argument("--mode", choices=("selection", "fusion"), default="fusion")
    parser.add_argument("--limit", type=int, help="Explicit diagnostic query limit; candidate pool remains full")
    args = parser.parse_args()
    if args.checkpoint is None:
        if args.backbone == "a3prvr":
            args.checkpoint = WORKSPACE / "checkpoints/a3prvr/tvr.ckpt"
        elif args.backbone == "holmes-multisrc":
            args.checkpoint = WORKSPACE / "Holmes/tvr-multi-tvr.ckpt"
        else:
            parser.error("--checkpoint is required for holmes-src")
    for name in ("max_tokens", "generation_tokens", "context_batch", "query_batch", "reference_videos", "bank_queries"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_tokens < 3 or (args.limit is not None and args.limit < 1):
        parser.error("max-tokens must be >=3 and limit positive")
    if args.command == "run" and (args.captions is None or args.features is None):
        parser.error("run requires --captions and --features")
    import torch
    from transformers import set_seed
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    {"build-bank": build_bank, "run": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
