"""Run: python utils/test_holmes_ar.py [--smoke]."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import holmes_ar as ar


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + ["<s>", "</s>"]


def check():
    rows = {f"v{i}#enc#{j}": "A person walks." for i in range(8) for j in range(2)}
    part = ar.subset(rows, 0.25, 7)
    assert len(part) == 4 and part == ar.subset(rows, 0.25, 7)
    scored = [dict(id=cid, query=q, u=(i + 1) / 20) for i, (cid, q) in enumerate(rows.items())]
    chosen = ar.choose(scored, "uncertain", 0.25, 0)
    assert chosen == {row["id"] for row in scored[-4:]}
    assert len(ar.choose(scored, "random", 0.25, 0)) == len(chosen)
    assert ar.choose(scored, "none", 1, 0) == set()
    token = Tokenizer()
    assert ar.rejection("He walks before eating.", "A man walks after eating.", token, 30) == "temporal_or_negation_change"
    assert ar.rejection("He does not run.", "He runs.", token, 30) == "temporal_or_negation_change"
    assert ar.rejection("He doesn't run.", "He runs.", token, 30) == "temporal_or_negation_change"
    assert ar.rejection("He walks.", "A man walks.", token, 5) is None
    assert ar.rejection("He walks.", "A man walks.", token, 4) == "token_budget"
    assert ar.rejection("He walks.", "He walks.", token, 30) == "unchanged"
    assert ar.rejection("He walks.", None, token, 30) == "invalid_text"
    assert ar.video_id("v#enc#0#ar") == "v"
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        text = root / "tvr" / "TextData"
        text.mkdir(parents=True)
        for split, values in (("train", rows), ("val", {"val#enc#0": "Walk."}), ("test", {"test#enc#0": "Run."})):
            ar.write_captions(text / f"tvr{split}.caption.txt", values)
        assert ar.training_captions(root) == rows
        report = root / "screening.json"
        report.write_text(json.dumps(dict(split="train", data_root=str(root), queries=scored)))
        args = SimpleNamespace(screening=report, output=root / "none", selection="none",
                               ratio=0.25, seed=0, siblings=False, model="unused", roberta="unused", max_tokens=30)
        ar.rewrite(args)
        assert ar.read_captions(args.output / "train.caption.txt") == rows
        (text / "tvrval.caption.txt").write_text("v0#enc#99 Leakage.\n")
        try:
            ar.training_captions(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Held-out video leakage was not rejected")
    print("Selection, guardrails, original preservation, and leakage checks passed.")


def smoke():
    sys.path.insert(0, str(ar.SRC))
    import h5py
    import numpy as np
    import torch
    from Configs.tvr import cfg
    from Models.builder import get_models
    from Datasets.builder import get_datasets
    from Losses.loss import EDL_loss
    assert torch.cuda.is_available(), "--smoke requires the Holmes CUDA environment"
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        text = root / "tvr" / "TextData"
        visual = root / "tvr" / "FeatureData" / "i3d_resnet"
        text.mkdir(parents=True)
        visual.mkdir(parents=True)
        ids = ("train_a", "train_b", "val", "test")
        for split, vids in (("train", ids[:2]), ("val", ids[2:3]), ("test", ids[3:])):
            ar.write_captions(text / f"tvr{split}.caption.txt", {f"{v}#enc#0": "A person walks." for v in vids})
        feature_path = text / "roberta_tvr_query_feat.hdf5"
        rng = np.random.default_rng(0)
        with h5py.File(feature_path, "w") as file:
            for vid in ids:
                file.create_dataset(vid + "#enc#0", data=rng.normal(size=(5, 768)).astype("float32"))
        with h5py.File(visual / "feature.hdf5", "w") as file:
            for vid in ids:
                file.create_dataset(vid, data=rng.normal(size=(8, 3072)).astype("float32"))
        (visual / "video2frames.txt").write_text(repr({v: [] for v in ids}))
        small_cfg = dict(cfg, hidden_size=8, n_heads=2)
        checkpoint = root / "tiny.ckpt"
        torch.save(dict(config=small_cfg, state_dict=get_models(small_cfg).state_dict()), checkpoint)
        ar.screen(SimpleNamespace(data_root=root, fraction=1, seed=0, checkpoint=checkpoint,
                                  output=root / "screen", text_features=feature_path, context_batch=1, query_batch=1))
        report = json.loads((root / "screen" / "screening.json").read_text())
        assert len(report["queries"]) == 2
        assert all(0 < row["u"] <= 1 for row in report["queries"])
        generator = MagicMock()
        generator.tokenizer.apply_chat_template.return_value = "test prompt"
        generator.side_effect = [
            [{"generated_text": json.dumps({"query": "word " * 40})}],
            [{"generated_text": json.dumps({"query": "A person is walking."})}],
            [{"generated_text": "invalid JSON"}],
        ]
        fake_transformers = SimpleNamespace(
            pipeline=MagicMock(return_value=generator),
            RobertaTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: MagicMock()),
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            ar.rewrite(SimpleNamespace(screening=root / "screen" / "screening.json",
                                      output=root / "rewrites", selection="all", ratio=1, seed=0,
                                      siblings=True, model="mock", roberta="mock", max_tokens=30))
        generated = ar.read_captions(root / "rewrites" / "train.caption.txt")
        assert len(generated) == 3 and generator.call_count == 3
        assert generated["train_a#enc#0"] == generated["train_b#enc#0"] == "A person walks."
        assert sum(cid.endswith("#ar") for cid in generated) == 1
        scores = torch.tensor([[0.1, 0.2, -0.1]])
        edl = EDL_loss(small_cfg)
        expected = 3 / edl.get_alpha(scores).sum(-1)
        streamed = 3 / sum(edl.get_alpha(block).sum(-1) for block in scores.split(1, dim=-1))
        assert torch.allclose(expected, streamed)
        captions = root / "augmented.caption.txt"
        ar.write_captions(captions, {"train_a#enc#0": "A person walks.", "train_a#enc#0#ar": "A person is walking."})
        with h5py.File(feature_path, "a") as file:
            file.create_dataset("train_a#enc#0#ar", data=np.ones((6, 768), dtype="float32"))
        with patch.object(ar.runpy, "run_path") as entry:
            ar.train(SimpleNamespace(data_root=root, captions=captions, text_features=feature_path,
                                     output=root / "training", seed=0, epochs=1, batch_size=1, gpu="0"))
            entry.assert_called_once()
        _, loader, _, _, _, _ = get_datasets(cfg)
        batch = next(iter(loader))
        assert batch["text_labels"] == [0, 0]
        assert batch["text_feat"].shape == (2, 6, 768)
    print("Synthetic GPU screening, EDL accumulation, and training data integration passed.")


if __name__ == "__main__":
    check()
    if "--smoke" in sys.argv:
        smoke()
