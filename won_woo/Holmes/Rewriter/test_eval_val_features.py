"""Offline validation-wrapper check: python -m Rewriter.test_eval_val_features."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np

from utils.holmes_ar import read_captions
from . import eval_val_features as evaluator


def check():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        captions = root / "val.caption.txt"
        captions.write_text(
            "a#enc#0 An actor walks.\nb#enc#0 A woman sits.\nc#enc#0 A dog runs.\n"
        )
        base = root / "base.hdf5"
        partial = root / "partial.hdf5"
        full = root / "full.hdf5"
        for path, ids in ((base, ("a", "b", "c")), (partial, ("b",)),
                          (full, ("a", "b", "c"))):
            with h5py.File(path, "w") as file:
                for cid in ids:
                    file.create_dataset(f"{cid}#enc#0", data=np.zeros((1, 768)))

        calls = []
        old_score = evaluator.score

        def fake_score(args):
            ids = list(read_captions(args.captions))
            calls.append((args.features.name, ids))
            videos = np.array(["a", "b", "c"])
            base_scores = {
                "a#enc#0": [0.8, 0.1, 0.1],
                "b#enc#0": [0.8, 0.6, 0.1],
                "c#enc#0": [0.1, 0.1, 0.8],
            }
            new_scores = {
                "a#enc#0": [0.8, 0.1, 0.1],
                "b#enc#0": [0.1, 0.9, 0.1],
                "c#enc#0": [0.1, 0.1, 0.8],
            }
            chosen = base_scores if args.features == base else new_scores
            matrix = np.array([chosen[cid] for cid in ids], dtype=np.float32)
            np.savez(args.output, query_ids=np.array(ids), video_ids=videos,
                     clip=matrix, frame=matrix, tau=0.2,
                     clip_weight=0.5, frame_weight=0.5)

        evaluator.score = fake_score
        try:
            args = SimpleNamespace(
                checkpoint=captions, val_captions=captions, val_features=partial,
                original_features=base, data_root=root, output_dir=root / "fusion",
                mode="fusion", device="cuda:0", query_batch=2, context_batch=2,
            )
            report = evaluator.evaluate(args)
            assert report["covered_queries"] == 1 and report["total_queries"] == 3
            assert np.isclose(report["original"]["R@1"], 200 / 3)
            assert report["original_plus_rewrite"]["R@1"] == 100
            assert report["covered_original"]["R@1"] == 0
            assert report["covered_original_plus_rewrite"]["R@1"] == 100
            assert np.isclose(report["delta_full"]["R@1"], 100 / 3)
            assert calls == [
                ("base.hdf5", list(read_captions(captions))),
                ("partial.hdf5", ["b#enc#0"]),
            ]
            assert (args.output_dir / "selected.caption.txt").read_text() == (
                "b#enc#0 A woman sits.\n"
            )
            assert (args.output_dir / "metrics.json").is_file()

            args.mode, args.output_dir = "single", root / "invalid"
            try:
                evaluator.evaluate(args)
            except ValueError as exc:
                assert "1/3" in str(exc)
            else:
                raise AssertionError("Partial HDF5 was accepted as full val")
            args.val_features, args.output_dir = full, root / "single"
            single = evaluator.evaluate(args)
            assert single["single"]["queries"] == 3
        finally:
            evaluator.score = old_score
    print("Full-val and partial fusion metrics, ID selection, and coverage checks passed.")


if __name__ == "__main__":
    check()
