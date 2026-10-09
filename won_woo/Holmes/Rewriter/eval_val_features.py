"""One-command Holmes TVR validation from a text-feature HDF5.

Default fusion scores original and supplied features with the same checkpoint.
Only val IDs present in the supplied HDF5 are fused; other IDs retain original
scores. Use --mode single for a full-val feature replacement experiment.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from utils.holmes_ar import positive, read_captions
from .evaluate import ROOT, combined, ranks, score, variants


DATA_ROOT = ROOT.parent
DEFAULT_CHECKPOINT = DATA_ROOT / "Holmes_ICML/tvr-multi/tvr/20260914-235629(extract)/best.ckpt"
DEFAULT_CAPTIONS = DATA_ROOT / "tvr/TextData/tvrval.caption.txt"
DEFAULT_ORIGINAL_FEATURES = DATA_ROOT / "tvr/TextData/roberta_tvr_query_feat.hdf5"


def recall_metrics(rank):
    if not len(rank):
        raise ValueError("Cannot evaluate zero queries")
    metrics = {f"R@{k}": float(100 * np.mean(rank <= k)) for k in (1, 5, 10, 100)}
    return dict(queries=len(rank), **metrics, Rsum=float(sum(metrics.values())))


def score_args(args, captions, features, output):
    return SimpleNamespace(
        checkpoint=args.checkpoint, backbone="multisrc", data_root=args.data_root,
        captions=captions, video_captions=args.val_captions, features=features,
        output=output, device=args.device, query_batch=args.query_batch,
        context_batch=args.context_batch, preflight=False,
    )


def evaluate(args):
    for path in (args.checkpoint, args.val_captions, args.val_features):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.mode == "fusion" and not args.original_features.is_file():
        raise FileNotFoundError(args.original_features)

    val_rows = read_captions(args.val_captions)
    with h5py.File(args.val_features, "r") as features:
        covered = [cid for cid in val_rows if cid in features]
    if not covered:
        raise ValueError("Validation feature HDF5 contains no val query IDs")
    if args.mode == "single" and len(covered) != len(val_rows):
        raise ValueError(
            f"Single mode needs every val ID: {len(covered)}/{len(val_rows)} found. "
            "Use --mode fusion for a partial rewrite HDF5."
        )

    output = args.output_dir or args.val_features.parent / f"{args.val_features.stem}.holmes_eval"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        checkpoint=str(args.checkpoint.resolve()),
        val_captions=str(args.val_captions.resolve()),
        val_features=str(args.val_features.resolve()),
        mode=args.mode,
        covered_queries=len(covered),
        total_queries=len(val_rows),
        output_dir=str(output.resolve()),
    )

    if args.mode == "single":
        score_path = output / "val.single.npz"
        score(score_args(args, args.val_captions, args.val_features, score_path))
        with np.load(score_path, allow_pickle=False) as result:
            rank, _ = ranks(combined(result), result["video_ids"], result["query_ids"])
            report["single"] = recall_metrics(rank)
    else:
        # The dataset loader reads IDs from this file; query vectors come from HDF5.
        selected_captions = output / "selected.caption.txt"
        selected_captions.write_text(
            "".join(f"{cid} {val_rows[cid]}\n" for cid in covered), encoding="utf-8"
        )
        original_path = output / "val.original.npz"
        rewrite_path = output / "val.rewritten.npz"
        score(score_args(args, args.val_captions, args.original_features, original_path))
        score(score_args(args, selected_captions, args.val_features, rewrite_path))
        with np.load(original_path, allow_pickle=False) as original, \
                np.load(rewrite_path, allow_pickle=False) as rewritten:
            methods, _, _ = variants(original, rewritten, float("inf"))
            selected = np.isin(original["query_ids"], rewritten["query_ids"])
            original_rank, _ = ranks(
                methods["original"], original["video_ids"], original["query_ids"]
            )
            fused_rank, _ = ranks(
                methods["average"], original["video_ids"], original["query_ids"]
            )
            report["original_features"] = str(args.original_features.resolve())
            report["original"] = recall_metrics(original_rank)
            report["original_plus_rewrite"] = recall_metrics(fused_rank)
            report["covered_original"] = recall_metrics(original_rank[selected])
            report["covered_original_plus_rewrite"] = recall_metrics(fused_rank[selected])
            report["delta_full"] = {
                key: report["original_plus_rewrite"][key] - report["original"][key]
                for key in ("R@1", "R@5", "R@10", "R@100", "Rsum")
            }

    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output / "metrics.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-features", type=Path, required=True,
                        help="Full or partial val-query RoBERTa HDF5 to evaluate")
    parser.add_argument("--mode", choices=("fusion", "single"), default="fusion",
                        help="Fusion averages original+new scores; single requires all val IDs")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--val-captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--original-features", type=Path, default=DEFAULT_ORIGINAL_FEATURES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch", type=positive, default=128)
    parser.add_argument("--context-batch", type=positive, default=64)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
