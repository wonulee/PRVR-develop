"""Checks for full training integration; --gpu exercises real Holmes forward/backward."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from .train_experiment import append_rewrites, dataset_layout, link_features, read_splits, save_captions


def fixture(root):
    import h5py
    import numpy as np
    text = root / "tvr/TextData"
    visual = root / "tvr/FeatureData/i3d_resnet"
    text.mkdir(parents=True)
    visual.mkdir(parents=True)
    splits = {
        "train": {"a#enc#0": "A man walks.", "a#enc#1": "A man sits.",
                  "b#enc#0": "A dog runs.", "b#enc#1": "A dog sits."},
        "val": {"v#enc#0": "A woman stands."},
        "test": {"t#enc#0": "A cat sleeps."},
    }
    for split, rows in splits.items():
        save_captions(text / f"tvr{split}.caption.txt", rows)
    rng = np.random.default_rng(7)
    with h5py.File(text / "roberta_tvr_query_feat.hdf5", "w") as f:
        for rows in splits.values():
            for cid in rows:
                f[cid] = rng.normal(size=(5, 768)).astype("float32")
    with h5py.File(visual / "feature.hdf5", "w") as f:
        for vid in ("a", "b", "v", "t"):
            f[vid] = rng.normal(size=(8, 3072)).astype("float32")
    (visual / "video2frames.txt").write_text(repr({vid: [] for vid in ("a", "b", "v", "t")}))
    return splits


def check():
    import h5py
    import numpy as np
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        splits = fixture(root)
        rows, union, info = read_splits(root)
        assert len(union) == 6 and info["independent_test"]
        augmented, added = append_rewrites(rows["train"], {"a#enc#0": "A man is walking."})
        assert len(augmented) == 5 and added == {"a#enc#0#ar": "A man is walking."}
        assert all(augmented[cid] == text for cid, text in rows["train"].items())
        try:
            append_rewrites(rows["train"], {"v#enc#0": "A woman is standing."})
        except ValueError:
            pass
        else:
            raise AssertionError("Held-out augmentation accepted")
        extra = root / "extra.hdf5"
        with h5py.File(extra, "w") as f:
            f["a#enc#0#ar"] = np.ones((6, 768), dtype="float32")
        features = root / "combined.hdf5"
        link_features(features, root / "tvr/TextData/roberta_tvr_query_feat.hdf5", extra)
        with h5py.File(features) as f:
            assert len(f) == 7 and f["a#enc#0#ar"].shape == (6, 768)
            assert f["a#enc#0"].shape == (5, 768)
        captions = root / "augmented.caption.txt"
        save_captions(captions, augmented)
        dataset_layout(root / "isolated", root, captions, features)
        save_captions(root / "tvr/TextData/tvrtest.caption.txt", rows["val"])
        _, union, info = read_splits(root)
        assert len(union) == 5 and info["val_test_identical"] and not info["independent_test"]
        save_captions(root / "tvr/TextData/tvrtest.caption.txt", {"a#enc#9": "Leaked query."})
        try:
            read_splits(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Training/held-out overlap accepted")
    print("Train-only augmentation, feature links, duplicate val/test labels and leakage checks passed.")


def gpu_check(gpu, backbone):
    import Rewriter.train as training
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root)
        # Test the actual augmented native data loader, not just original training.
        text = root / "tvr/TextData"
        from utils.holmes_ar import read_captions
        import h5py
        import numpy as np
        originals = read_captions(text / "tvrtrain.caption.txt")
        augmented, _ = append_rewrites(originals, {"a#enc#0": "A man is walking."})
        save_captions(text / "tvrtrain.caption.txt", augmented)
        with h5py.File(text / "roberta_tvr_query_feat.hdf5", "a") as f:
            f["a#enc#0#ar"] = np.ones((6, 768), dtype="float32")
        args = SimpleNamespace(gpu=gpu, backbone=backbone, data_root=root,
                               output=root / "training", epochs=2, batch_size=2,
                               query_batch=1, context_batch=1, seed=0, mode="train")
        original_entry = training.native_entry

        def interrupted_entry(*a):
            entry = original_entry(*a)
            train_epoch = entry.train_one_epoch

            def interrupt(epoch, *args, **kwargs):
                if epoch == 1:
                    raise RuntimeError("test interruption after the first saved epoch")
                return train_epoch(epoch, *args, **kwargs)
            entry.train_one_epoch = interrupt
            return entry

        with patch.object(training, "native_entry", interrupted_entry):
            try:
                training.run(args)
            except RuntimeError as exc:
                assert "test interruption" in str(exc)
            else:
                raise AssertionError("Expected simulated interruption")
        assert training.load_checkpoint(args.output / "last.ckpt")["epoch"] == 0
        assert not (args.output / "train_complete.json").exists()
        training.run(args)
        state = training.load_checkpoint(args.output / "last.ckpt")
        assert state["epoch"] == 1 and "rng" in state and "optimizer" in state
        assert (args.output / "train_complete.json").exists()
        args.mode = "test"
        training.run(args)
        result = json.loads((args.output / "test_metrics.json").read_text())
        assert result["independent_test"] and len(result["test"]) == 5
        assert result["checkpoint"].endswith("best.ckpt")
        assert all(0 <= x <= 100 for x in result["test"][:4])
        # A completed training run can be re-entered without repeating an epoch.
        args.mode = "train"
        training.run(args)
        assert training.load_checkpoint(args.output / "last.ckpt")["epoch"] == 1
    print(f"{backbone}: native GPU backward, epoch resume, best-checkpoint test and completion checks passed.")


def pipeline_check(gpu):
    import subprocess
    import sys
    from .evaluate import ROOT
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root / "data")
        output = root / "run"
        command = [
            sys.executable, "-u", "-m", "Rewriter.train_experiment",
            "--data-root", str(root / "data"), "--output", str(output),
            "--conditions", "original", "context", "--gpu", gpu,
            "--epochs", "1", "--batch-size", "2", "--query-batch", "1", "--context-batch", "1",
        ]
        log = root / "pipeline.log"
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            print(log.read_text())
            for path in sorted((output / "logs").glob("*.log")):
                print(path.name, path.read_text()[-6000:])
            raise AssertionError("Full pipeline failed")
        order = log.read_text()
        expected = ("rewrite-context", "extract-original", "train-original", "train-context",
                    "test-original", "test-context")
        positions = [order.index("[start] " + name + " ->") for name in expected]
        assert positions == sorted(positions), "Rewriting/features must precede all training and tests"
        assert (output / "complete.json").exists()
        assert len((output / "summary.csv").read_text().splitlines()) == 5
        best = output / "context/training/best.ckpt"
        modified = best.stat().st_mtime_ns
        with log.open("a") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert best.stat().st_mtime_ns == modified
        assert "[skip] train-context" in log.read_text()
    print("CLI pipeline: original/context preparation, real LLM, training, test, summary and stage reuse passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu")
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--backbone", choices=("src", "multisrc"), default="multisrc")
    args = p.parse_args()
    check()
    if args.pipeline:
        if not args.gpu:
            p.error("--pipeline requires --gpu")
        pipeline_check(args.gpu)
    elif args.gpu:
        gpu_check(args.gpu, args.backbone)
