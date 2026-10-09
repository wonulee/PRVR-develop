"""Checks: python -m Rewriter.test_experiment [--gpu cuda:1]."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from .evaluate import dst_probabilities, variants, ranks, score, ROOT
from .experiment import report
from .mca_rewriter import rewrite_file
from .test_rewriter import Generator, Tokenizer


def check():
    a = np.array([[0.2, 1., 5.], [3., 2., 4.]])
    b = np.array([[4., 2., 1.], [0., 1., 2.]])
    k = a.shape[1]
    s1, s2 = (a + 1).sum(1, keepdims=True), (b + 1).sum(1, keepdims=True)
    b1, b2, u1, u2 = a / s1, b / s2, k / s1, k / s2
    outer = b1[:, :, None] * b2[:, None, :]
    conflict = outer.sum((1, 2)) - np.diagonal(outer, axis1=1, axis2=2).sum(1)
    belief = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - conflict[:, None])
    unknown = u1 * u2 / (1 - conflict[:, None])
    np.testing.assert_allclose(dst_probabilities(a, b), belief + unknown / k)
    np.testing.assert_allclose(dst_probabilities(a, np.zeros_like(a)), (a + 1) / s1)
    np.testing.assert_allclose(dst_probabilities(a, b), dst_probabilities(b, a))
    original = dict(query_ids=np.array(["a#enc#0", "b#enc#0", "c#enc#0"]),
                    video_ids=np.array(["a", "b", "c"]), tau=0.2, clip_weight=0.5, frame_weight=0.5,
                    clip=np.array([[0.7, 0.1, 0.2], [0.8, 0.5, 0.1], [0.3, 0.2, 0.6]]),
                    frame=np.array([[0.7, 0.1, 0.2], [0.8, 0.5, 0.1], [0.3, 0.2, 0.6]]))
    new = dict(original, query_ids=np.array(["c#enc#0", "b#enc#0"]),
               clip=np.array([[0.1, 0.2, 0.9], [0.1, 0.8, 0.2]]),
               frame=np.array([[0.1, 0.2, 0.9], [0.1, 0.8, 0.2]]))
    v, u, active = variants(original, new, 2)
    np.testing.assert_array_equal(v["adaptive_dst"], v["original"])
    assert len(active) == 0
    assert (ranks(v["rewritten"], original["video_ids"], original["query_ids"])[0] == 1).all()
    for scores in v.values():
        np.testing.assert_array_equal(scores[0], v["original"][0])
    empty = dict(new, query_ids=np.array([], dtype=str), clip=np.empty((0, 3)), frame=np.empty((0, 3)))
    v, _, _ = variants(original, empty, 0)
    for scores in v.values():
        np.testing.assert_array_equal(scores, v["original"])
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        captions = root / "tvr/TextData"
        captions.mkdir(parents=True)
        caption_path = captions / "tvrval.caption.txt"
        caption_path.write_text("a#enc#0 He walks.\nb#enc#0 She sits.\nc#enc#0 A dog runs.\n")
        np.savez(root / "val.original.npz", **original)
        np.savez(root / "val.target-only.npz", **new)
        report(SimpleNamespace(output=root, data_root=root, ratio=0.25, modes=["target-only"], splits=["val"]))
        assert (root / "summary.csv").exists()
        assert len((root / "val.target-only.queries.csv").read_text().splitlines()) == 4
        output = root / "rewrites"
        try:
            rewrite_file(caption_path, output, Generator('{"query": "He is walking."}'), Tokenizer())
        except StopIteration:
            pass
        else:
            raise AssertionError("Interruption was not propagated")
        with (output / "audit.jsonl").open("ab") as stream:
            stream.write(b'{"id":')
        stats = rewrite_file(caption_path, output, Generator('{"query": "She is sitting."}',
                                                            '{"query": "A dog is running."}'),
                             Tokenizer(), resume=True)
        assert stats == dict(total=3, accepted=3, rejected=0)
        assert len((output / "audit.jsonl").read_text().splitlines()) == 3
        # A completed run must not call the generator again.
        rewrite_file(caption_path, output, Generator(), Tokenizer(), resume=True)
        try:
            rewrite_file(caption_path, output, Generator(), Tokenizer(), max_tokens=3, resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Resume accepted a changed token budget")
    print("DST equivalence, alignment, fallback, metrics and interrupted-resume checks passed.")


def gpu_check(device):
    import torch
    from .evaluate import load_model
    from utils.holmes_ar import read_captions, write_captions
    data = ROOT.parent
    source = read_captions(data / "tvr/TextData/tvrval.caption.txt")
    videos = list(dict.fromkeys(cid.split("#")[0] for cid in source))[:3]
    rows = {cid: q for cid, q in source.items() if cid.split("#")[0] in videos}
    features = data / "tvr/TextData/roberta_tvr_query_feat_extract.hdf5"
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        captions = root / "captions.txt"
        write_captions(captions, rows)
        args = SimpleNamespace(device=device, checkpoint=ROOT / "tvr-multi-tvr.ckpt",
                               backbone="multisrc", data_root=data, captions=captions,
                               video_captions=captions, features=features, query_batch=7,
                               context_batch=2, output=root / "scores.npz", preflight=False)
        score(args)  # 15 targets: the final batch has one query.
        model, cfg = load_model(args.checkpoint, "multisrc", device)
        from torch.utils.data import DataLoader
        from Datasets.data_provider import VisDataSet4PRVR, TxtDataSet4PRVR, collate_frame_val, collate_text_val
        from Validations.validations import validations
        cfg["eval_context_bsz"] = 3
        validator = validations(cfg)
        contexts = DataLoader(VisDataSet4PRVR(str(data / "tvr/FeatureData/i3d_resnet/feature.hdf5"),
                                            None, cfg, video_ids=videos),
                              batch_size=3, collate_fn=collate_frame_val)
        queries = DataLoader(TxtDataSet4PRVR(str(captions), str(features), cfg),
                             batch_size=7, collate_fn=collate_text_val)
        with torch.inference_mode():
            context = validator.compute_context_info(model, contexts)
            expected, ids = validator.compute_query2ctx_info(model, queries, context)
        with np.load(args.output) as actual:
            positions = {cid: i for i, cid in enumerate(actual["query_ids"])}
            combined = actual["clip"] * actual["clip_weight"] + actual["frame"] * actual["frame_weight"]
            np.testing.assert_allclose(combined[[positions[cid] for cid in ids]],
                                       expected.cpu().numpy(), atol=2e-6, rtol=2e-6)
    print("Real checkpoint: streamed scoring equals existing evaluation; singleton batch passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu")
    args = p.parse_args()
    check()
    if args.gpu:
        gpu_check(args.gpu)
