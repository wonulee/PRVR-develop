"""Full TVR rewriting and fusion experiment, with restartable stages."""
import argparse
import codecs
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from utils.holmes_ar import positive, read_captions
from .evaluate import ROOT


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def manifest(args):
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(args).items()
              if k not in ("check_only", "device")}
    sources = list((ROOT / "Rewriter").glob("*.py"))
    sources += list((ROOT / args.backbone / "Models").rglob("*.py"))
    sources += [ROOT / "utils/holmes_ar.py", ROOT / "utils/extract_tvr_roberta_features.py"]
    config["source_sha256"] = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted(sources)
    )).hexdigest()
    config["checkpoint"] = identity(args.checkpoint)
    config["captions"] = {
        split: hashlib.sha256((args.data_root / f"tvr/TextData/tvr{split}.caption.txt").read_bytes()).hexdigest()
        for split in args.splits
    }
    config["visual_features"] = [identity(p) for p in sorted((args.data_root / "tvr/FeatureData").glob("*/feature.hdf5"))]
    config["roberta_files"] = [identity(p) for p in sorted(args.roberta.iterdir()) if p.is_file()]
    return config


def stage(work, name, command, outputs):
    marker = work / "stages" / (name + ".json")
    if marker.exists() and all(p.exists() for p in outputs):
        print(f"[skip] {name}", flush=True)
        return
    log = work / "logs" / (name + ".log")
    print(f"[start] {name} -> {log}", flush=True)
    start = time.time()
    # Read chunks rather than lines so tqdm's carriage-return updates appear immediately.
    with log.open("ab") as stream, subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ) as process:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while chunk := process.stdout.read1(65536):
                stream.write(chunk)
                stream.flush()
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    if not all(p.exists() for p in outputs):
        raise RuntimeError(f"{name} finished without required outputs")
    marker.write_text(json.dumps(dict(command=command, elapsed_seconds=time.time() - start), indent=2))
    print(f"[done] {name}", flush=True)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(args):
    import numpy as np
    from .evaluate import ranks, uncertainty, variants
    work = args.output
    with np.load(work / "val.original.npz", allow_pickle=False) as val:
        threshold = float(np.quantile(uncertainty(val), 1 - args.ratio))
    (work / "threshold.json").write_text(json.dumps(dict(
        source_split="val", selection="max branch uncertainty >= threshold",
        validation_target_ratio=args.ratio, threshold=threshold,
    ), indent=2))
    summary = []
    for split in args.splits:
        originals = read_captions(args.data_root / f"tvr/TextData/tvr{split}.caption.txt")
        with np.load(work / f"{split}.original.npz", allow_pickle=False) as original:
            for mode in args.modes:
                with np.load(work / f"{split}.{mode}.npz", allow_pickle=False) as rewritten:
                    score_variants, u, active = variants(original, rewritten, threshold)
                    ids, videos = original["query_ids"], original["video_ids"]
                    accepted = set(rewritten["query_ids"])
                    chosen = set(active.tolist())
                    details = [dict(caption_id=cid, query=originals[cid], uncertainty=float(u[i]),
                                    accepted_rewrite=cid in accepted, selected_for_fusion=i in chosen)
                               for i, cid in enumerate(ids)]
                    for method, scores in score_variants.items():
                        rank, top1 = ranks(scores, videos, ids)
                        metrics = {f"R@{k}": float(100 * (rank <= k).mean()) for k in (1, 5, 10, 100)}
                        summary.append(dict(split=split, context=mode, method=method, queries=len(ids),
                                            accepted_rewrites=len(accepted), fused_queries=len(chosen)
                                            if method == "adaptive_dst" else
                                            (len(accepted) if method in ("dst", "average") else 0),
                                            **metrics, Rsum=sum(metrics.values())))
                        for i, row in enumerate(details):
                            row[method + "_rank"] = int(rank[i])
                            row[method + "_top1"] = top1[i]
                    write_csv(work / f"{split}.{mode}.queries.csv", details)
    write_csv(work / "summary.csv", summary)
    (work / "complete.json").write_text(json.dumps(dict(status="complete", threshold=threshold)))
    print(f"Completed: {work / 'summary.csv'}", flush=True)


def preflight(args):
    import torch
    from transformers import AutoConfig, AutoTokenizer, RobertaTokenizer
    from .context_manager import build_video_context_db
    from .evaluate import load_model, check_features
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("A CUDA device is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, cfg = load_model(args.checkpoint, args.backbone, "cpu")
    del model
    tokenizer = RobertaTokenizer.from_pretrained(str(args.roberta), local_files_only=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not llm_tokenizer.chat_template:
        raise ValueError("LLM tokenizer must have a chat template")
    AutoConfig.from_pretrained(args.model, local_files_only=True)
    visual = args.data_root / "tvr/FeatureData" / cfg["visual_feature"] / "feature.hdf5"
    counts, seen = {}, set()
    for split in args.splits:
        contexts = build_video_context_db(args.data_root / f"tvr/TextData/tvr{split}.caption.txt")
        if seen.intersection(contexts):
            raise ValueError("Validation and test video sets overlap")
        seen.update(contexts)
        check_features(visual, contexts, cfg["visual_feat_dim"])
        counts[split] = dict(queries=sum(map(len, contexts.values())), videos=len(contexts))
    print(json.dumps(dict(status="ready", splits=counts, checkpoint=str(args.checkpoint),
                          backbone=args.backbone, device=args.device,
                          special_tokens=tokenizer.num_special_tokens_to_add()), indent=2), flush=True)


def run(args):
    preflight(args)
    if args.check_only:
        return
    work = args.output
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = manifest(args)
        manifest_path = work / "experiment.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != settings:
                raise ValueError("Experiment inputs/options/code changed; use a new --output directory")
        else:
            if any(p.name != ".lock" for p in work.iterdir()):
                raise ValueError("Output directory contains an unrelated experiment")
            manifest_path.write_text(json.dumps(settings, indent=2))
        for directory in ("logs", "stages"):
            (work / directory).mkdir(exist_ok=True)
        py = [sys.executable, "-u", "-m"]
        captions = {s: args.data_root / f"tvr/TextData/tvr{s}.caption.txt" for s in args.splits}
        original_features = work / "original.hdf5"
        extract = py + ["Rewriter.extract_roberta_feat", "--model", str(args.roberta),
                        "--max-length", "30", "--layer", "-2", "--device", args.device,
                        "--batch-size", str(args.feature_batch)]
        caption_args = [item for path in captions.values() for item in ("--caption-file", str(path))]
        stage(work, "extract-original", extract + caption_args + [
            "--output", str(original_features), "--overwrite"], [original_features])
        scoring = py + ["Rewriter.evaluate", "--checkpoint", str(args.checkpoint),
                        "--backbone", args.backbone, "--data-root", str(args.data_root),
                        "--device", args.device, "--query-batch", str(args.query_batch),
                        "--context-batch", str(args.context_batch)]
        for split in args.splits:
            score_file = work / f"{split}.original.npz"
            stage(work, f"score-{split}-original", scoring + [
                "--captions", str(captions[split]), "--video-captions", str(captions[split]),
                "--features", str(original_features), "--output", str(score_file)], [score_file])
            for mode in args.modes:
                rewrite_dir = work / f"{split}.{mode}"
                command = py + ["Rewriter.mca_rewriter", "--caption-file", str(captions[split]),
                                "--output", str(rewrite_dir), "--model", args.model,
                                "--roberta", str(args.roberta), "--device", args.device, "--resume"]
                if mode == "target-only":
                    command += ["--no-siblings"]
                rewritten_captions = rewrite_dir / "rewritten.caption.txt"
                stage(work, f"rewrite-{split}-{mode}", command,
                      [rewritten_captions, rewrite_dir / "run.json"])
                features = rewrite_dir / "rewritten.hdf5"
                if rewritten_captions.read_text().strip():
                    stage(work, f"extract-{split}-{mode}", extract + [
                        "--caption-file", str(rewritten_captions),
                        "--output", str(features), "--overwrite"], [features])
                score_file = work / f"{split}.{mode}.npz"
                stage(work, f"score-{split}-{mode}", scoring + [
                    "--captions", str(rewritten_captions), "--video-captions", str(captions[split]),
                    "--features", str(features), "--output", str(score_file)], [score_file])
        report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "tvr-multi-tvr.ckpt")
    p.add_argument("--backbone", choices=("src", "multisrc"), default="multisrc")
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--roberta", type=Path, default=ROOT.parent / "tvr/query_only/roberta-base_tuned_model")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", type=Path, default=ROOT.parent / "experiments/mca_full")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"],
                   help="Default val; test requires a distinct held-out split")
    p.add_argument("--modes", nargs="+", choices=("target-only", "oracle-context"),
                   default=["target-only", "oracle-context"])
    p.add_argument("--ratio", type=float, default=0.25, help="Validation uncertainty quantile gate")
    p.add_argument("--query-batch", type=positive, default=16)
    p.add_argument("--context-batch", type=positive, default=64)
    p.add_argument("--feature-batch", type=positive, default=64)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if not 0 < args.ratio <= 1:
        p.error("--ratio must be in (0, 1]")
    if "val" not in args.splits or len(args.splits) != len(set(args.splits)):
        p.error("--splits must include val exactly once; optionally add test")
    if len(args.modes) != len(set(args.modes)):
        p.error("Duplicate context modes")
    for name in ("checkpoint", "data_root", "roberta", "output"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == "__main__":
    main()
