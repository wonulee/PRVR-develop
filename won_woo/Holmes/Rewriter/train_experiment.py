"""Full train -> validation-selected best checkpoint -> test, using native Holmes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

from utils.holmes_ar import positive, read_captions, training_captions
from .evaluate import ROOT, check_features
from .experiment import identity, stage, write_csv
from .train import atomic_json


def read_splits(data_root):
    text = data_root / "tvr/TextData"
    splits = {"train": training_captions(data_root)}
    splits.update({name: read_captions(text / f"tvr{name}.caption.txt") for name in ("val", "test")})
    union = {}
    for rows in splits.values():
        for cid, query in rows.items():
            if cid in union and union[cid] != query:
                raise ValueError(f"Conflicting captions share ID {cid}")
            union[cid] = query
    videos = {name: {cid.split("#")[0] for cid in rows} for name, rows in splits.items()}
    info = dict(counts={name: dict(queries=len(rows), videos=len(videos[name])) for name, rows in splits.items()},
                val_test_identical=splits["val"] == splits["test"],
                val_test_shared_videos=len(videos["val"] & videos["test"]),
                independent_test=not bool(videos["val"] & videos["test"]))
    return splits, union, info


def append_rewrites(originals, rewritten):
    augmented = dict(originals)
    additions = {}
    for cid, query in rewritten.items():
        if cid not in originals:
            raise ValueError(f"Rewrite is not from the training split: {cid}")
        if query == originals[cid]:
            continue
        new_id = cid + "#ar"
        if new_id in augmented:
            raise ValueError(f"Augmented ID collision: {new_id}")
        augmented[new_id] = additions[new_id] = query
    return augmented, additions


def save_captions(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for cid, query in rows.items():
            stream.write(f"{cid} {query}\n")
    temporary.replace(path)


def link_features(output, base, extra=None):
    """Native HDF5 external links share immutable feature arrays across conditions."""
    import h5py
    import os
    temporary = output.with_suffix(".tmp")
    with h5py.File(temporary, "w") as target:
        for source in [base] + ([extra] if extra else []):
            with h5py.File(source, "r") as features:
                for cid in features:
                    if cid in target:
                        raise ValueError(f"Duplicate HDF5 feature: {cid}")
                    target[cid] = h5py.ExternalLink(os.path.relpath(source, output.parent), cid)
    temporary.replace(output)


def dataset_layout(output, data_root, train_captions, features):
    text = output / "tvr/TextData"
    text.mkdir(parents=True, exist_ok=True)
    sources = {
        text / "tvrtrain.caption.txt": train_captions,
        text / "tvrval.caption.txt": data_root / "tvr/TextData/tvrval.caption.txt",
        text / "tvrtest.caption.txt": data_root / "tvr/TextData/tvrtest.caption.txt",
        text / "roberta_tvr_query_feat.hdf5": features,
        text.parent / "FeatureData": data_root / "tvr/FeatureData",
    }
    for target, source in sources.items():
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Dataset link changed: {target}")
        elif target.exists():
            raise FileExistsError(target)
        else:
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def preflight(args, splits, info):
    import torch
    from transformers import AutoTokenizer, RobertaConfig, RobertaTokenizer
    sys.path.insert(0, str(ROOT / args.backbone))
    from Configs.tvr import cfg
    if not torch.cuda.is_available() or int(args.gpu) >= torch.cuda.device_count():
        raise ValueError("Requested CUDA GPU is unavailable")
    if info["counts"]["train"]["videos"] < 2 or args.batch_size < 2 or info["counts"]["train"]["videos"] % args.batch_size == 1:
        raise ValueError("Every contrastive batch needs at least two videos; adjust --batch-size")
    RobertaTokenizer.from_pretrained(str(args.roberta), local_files_only=True)
    config = RobertaConfig.from_pretrained(str(args.roberta), local_files_only=True)
    if config.hidden_size != cfg["q_feat_size"]:
        raise ValueError("RoBERTa dimension differs from Holmes q_feat_size")
    if any(c != "original" for c in args.conditions):
        llm = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if not llm.chat_template:
            raise ValueError("LLM has no chat template")
    videos = {cid.split("#")[0] for rows in splits.values() for cid in rows}
    check_features(args.data_root / "tvr/FeatureData" / cfg["visual_feature"] / "feature.hdf5",
                   videos, cfg["visual_feat_dim"])
    print(json.dumps(dict(status="ready", backbone=args.backbone,
                          epochs=args.epochs or cfg["n_epoch"], batch_size=args.batch_size,
                          conditions=args.conditions, **info), indent=2), flush=True)
    if not info["independent_test"]:
        print("NOTE: val/test overlap; test results will be labeled non-independent.", flush=True)


def signature(args):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("stage", "check_only", "gpu")}
    source = sorted((ROOT / args.backbone).rglob("*.py")) + sorted((ROOT / "Rewriter").glob("*.py"))
    source += [ROOT / "utils/holmes_ar.py", ROOT / "utils/extract_tvr_roberta_features.py"]
    config["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in source)).hexdigest()
    config["captions"] = {s: hashlib.sha256(
        (args.data_root / f"tvr/TextData/tvr{s}.caption.txt").read_bytes()).hexdigest()
        for s in ("train", "val", "test")}
    config["roberta_files"] = [identity(p) for p in sorted(args.roberta.iterdir()) if p.is_file()]
    config["visual_features"] = [identity(p) for p in sorted((args.data_root / "tvr/FeatureData").glob("*/feature.hdf5"))]
    return config


def run(args):
    splits, originals, info = read_splits(args.data_root)
    preflight(args, splits, info)
    if args.check_only:
        return
    work = args.output
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = signature(args)
        manifest = work / "experiment.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != settings:
                raise ValueError("Run settings, data or code changed; use a new --output")
        else:
            if any(p.name != ".lock" for p in work.iterdir()):
                raise ValueError("Output directory belongs to another run")
            atomic_json(manifest, settings)
        for name in ("logs", "stages"):
            (work / name).mkdir(exist_ok=True)
        atomic_json(work / "splits.json", info)
        py = [sys.executable, "-u", "-m"]
        device = "cuda:" + args.gpu
        base_features = work / "original.hdf5"
        extractor = py + ["Rewriter.extract_roberta_feat", "--model", str(args.roberta),
                          "--device", device, "--max-length", "30", "--layer", "-2",
                          "--batch-size", str(args.feature_batch), "--overwrite"]
        if args.stage in ("all", "prepare"):
            # Finish every train rewrite before feature extraction or model training.
            for condition in args.conditions:
                directory = work / condition
                directory.mkdir(exist_ok=True)
                train_rows = splits["train"]
                if condition != "original":
                    rewrites = directory / "rewrites"
                    command = py + ["Rewriter.mca_rewriter", "--caption-file",
                                    str(args.data_root / "tvr/TextData/tvrtrain.caption.txt"),
                                    "--output", str(rewrites), "--model", args.model,
                                    "--roberta", str(args.roberta), "--device", device, "--resume"]
                    if condition == "target-only":
                        command += ["--no-siblings"]
                    rewritten_file = rewrites / "rewritten.caption.txt"
                    stage(work, f"rewrite-{condition}", command, [rewritten_file, rewrites / "run.json"])
                    rewritten = read_captions(rewritten_file) if rewritten_file.read_text().strip() else {}
                    train_rows, additions = append_rewrites(train_rows, rewritten)
                    save_captions(directory / "added.caption.txt", additions)
                    atomic_json(directory / "augmentation.json", dict(
                        original=len(splits["train"]), added=len(additions), total=len(train_rows)))
                save_captions(directory / "train.caption.txt", train_rows)

            # Identical val/test caption IDs are extracted only once.
            all_captions = work / "original.caption.txt"
            save_captions(all_captions, originals)
            stage(work, "extract-original", extractor + ["--caption-file", str(all_captions),
                  "--output", str(base_features)], [base_features])
            for condition in args.conditions:
                directory = work / condition
                features = base_features
                if condition != "original":
                    additions_file = directory / "added.caption.txt"
                    extra = None
                    if additions_file.read_text().strip():
                        extra = directory / "added.hdf5"
                        stage(work, f"extract-{condition}", extractor + ["--caption-file",
                              str(additions_file), "--output", str(extra)], [extra])
                    features = directory / "features.hdf5"
                    link_features(features, base_features, extra)
                dataset_layout(directory / "dataset", args.data_root, directory / "train.caption.txt", features)

        phases = ("train", "test") if args.stage == "all" else (
            () if args.stage == "prepare" else (args.stage,))
        for phase in phases:
            for condition in args.conditions:
                directory = work / condition
                dataset = directory / "dataset"
                if not (dataset / "tvr/TextData/roberta_tvr_query_feat.hdf5").is_file():
                    raise ValueError(f"Prepare features first: {condition}; use --stage prepare or all")
                training = directory / "training"
                command = py + ["Rewriter.train", "--backbone", args.backbone,
                                "--data-root", str(dataset), "--output", str(training),
                                "--gpu", args.gpu, "--batch-size", str(args.batch_size),
                                "--query-batch", str(args.query_batch), "--context-batch", str(args.context_batch),
                                "--seed", str(args.seed), "--mode", phase]
                if args.epochs:
                    command += ["--epochs", str(args.epochs)]
                outputs = ([training / "best.ckpt", training / "last.ckpt", training / "train_complete.json"]
                           if phase == "train" else [training / "test_metrics.json"])
                stage(work, f"{phase}-{condition}", command, outputs)
        reports = []
        for condition in args.conditions:
            path = work / condition / "training/test_metrics.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            training_info = json.loads((path.parent / "train_complete.json").read_text())
            for split, values in (("val_best", result["best_validation"]), ("test", result["test"])):
                reports.append(dict(condition=condition, split=split, best_epoch=result["best_epoch"],
                                    independent_test=result["independent_test"],
                                    train_queries=training_info["train_queries"],
                                    added_queries=training_info["train_queries"] - len(splits["train"]),
                                    **dict(zip(("R@1", "R@5", "R@10", "R@100", "Rsum"), values))))
        if reports:
            write_csv(work / "summary.csv", reports)
            if len(reports) == 2 * len(args.conditions):
                atomic_json(work / "complete.json", dict(status="complete", **info))
        print(f"Finished stage {args.stage}: {work}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT.parent)
    p.add_argument("--output", type=Path, default=ROOT.parent / "experiments/mca_train_test")
    p.add_argument("--backbone", choices=("src", "multisrc"), default="multisrc")
    p.add_argument("--roberta", type=Path, default=ROOT.parent / "tvr/query_only/roberta-base_tuned_model")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--conditions", nargs="+", choices=("original", "target-only", "context"),
                   default=["original", "context"])
    p.add_argument("--stage", choices=("all", "prepare", "train", "test"), default="all")
    p.add_argument("--gpu", default="1")
    p.add_argument("--epochs", type=positive)
    p.add_argument("--batch-size", type=positive, default=128)
    p.add_argument("--query-batch", type=positive, default=16)
    p.add_argument("--context-batch", type=positive, default=64)
    p.add_argument("--feature-batch", type=positive, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if not args.gpu.isdigit() or len(args.conditions) != len(set(args.conditions)):
        p.error("Specify one GPU index and distinct conditions")
    for name in ("data_root", "output", "roberta"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == "__main__":
    main()
