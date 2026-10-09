"""Run from Holmes: python -m EvidenceRetrieval.test_method [--gpu cuda:1]."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from .core import Settings, RiskBank, InsufficientRoleBank, concentration, final_scores, candidate_utility, unit_vectors
from .language import LanguageModel, LanguageRejected, validate_decomposition, lexical_rejection
from .run import process_query


def must_fail(function, exception=ValueError):
    try:
        function()
    except exception:
        return
    raise AssertionError("Expected validation failure")


def settings():
    return Settings(**json.loads((Path(__file__).parent / "example_config.json").read_text()))


def offline_check():
    cfg = replace(settings(), top_m=2, neighbors=1)
    np.testing.assert_allclose(concentration([1, 1, 1], .1), 0, atol=1e-15)
    assert concentration([1, 0, 0], .01) > .999
    assert np.isfinite(concentration([1e100, -1e100], 1))
    must_fail(lambda: concentration([1], .1))
    must_fail(lambda: concentration([np.nan, 1], .1))
    must_fail(lambda: unit_vectors([[0, 0]]))
    must_fail(lambda: replace(cfg, rho=.6))
    must_fail(lambda: replace(cfg, risk_weights=[1, 1, 1]))
    must_fail(lambda: replace(cfg, candidates=1.5))
    scores = np.array([[.8, .2, .1], [.7, .1, .1], [.2, .8, .1], [.1, .7, .1]])
    vectors = [[1, 0], [.9, .1], [0, 1], [.1, .9]]
    roles = ["actor", "actor", "action", "action"]
    bank = RiskBank(roles, scores, vectors, cfg)
    assert set(bank.calibration) == {"actor", "action"}
    evidence = [dict(id="e0", role="actor", text="A man"), dict(id="e1", role="action", text="walks")]
    analyzed = bank.analyze(evidence, scores[[0, 2]], np.array(vectors)[[0, 2]])
    assert all(0 <= e[key] <= 1 for e in analyzed for key in ("G", "D", "OOD", "risk"))
    assert all(e["protected"] for e in analyzed)
    # Leave-one-out calibration cannot include its own perfect similarity of 1.
    assert bank.calibration["actor"][2].min() > 0
    must_fail(lambda: bank.analyze([dict(role="spatial")], scores[:1], vectors[:1]), InsufficientRoleBank)
    must_fail(lambda: RiskBank(roles, scores, vectors, replace(cfg, top_m=4)))
    original = np.array([.2, .2, .2], dtype=np.float32)
    rewritten = np.array([[.9, .1, .1], [.1, .9, .1]])
    for kwargs in (dict(enabled=False), dict(gate=False)):
        result, chosen = final_scores(original, rewritten, [1, 2], 0, cfg, **kwargs)
        np.testing.assert_array_equal(result, original)
        assert result.dtype == original.dtype and not chosen
    result, _ = final_scores(original, [], [], 0, cfg)
    np.testing.assert_array_equal(result, original)
    result, _ = final_scores(original, rewritten, [1, 1], 1, replace(cfg, improvement_margin=0))
    np.testing.assert_array_equal(result, original)  # ties retain original
    result, chosen = final_scores(original, rewritten, [0, 1], .5, cfg)
    np.testing.assert_allclose(result, (1 - cfg.rho) * original + cfg.rho * rewritten[1])
    assert chosen == [1]
    result, chosen = final_scores(original, rewritten, [0, 1], .5, cfg, mode="selection")
    np.testing.assert_array_equal(result, rewritten[1].astype(np.float32))
    assert chosen == [1]
    info = candidate_utility(rewritten[0], 1, [1, 0], 0, False, analyzed, cfg)
    assert not info["accepted"]  # action cannot be lost even if retrieval confidence improves
    query = "A man walks."
    payload = dict(evidence=[dict(id="e0", role="actor", text="A man"),
                             dict(id="e1", role="action", text="walks")],
                   relations=[dict(source="e0", relation="agent_of", target="e1")])
    graph = validate_decomposition(query, payload)
    assert graph["evidence"][1]["start"] == 6
    must_fail(lambda: validate_decomposition(query, dict(payload, relations=[])), LanguageRejected)
    incomplete = dict(payload, evidence=payload["evidence"][:1])
    must_fail(lambda: validate_decomposition(query, incomplete), LanguageRejected)
    repeated = dict(payload, evidence=payload["evidence"] + payload["evidence"][:1])
    must_fail(lambda: validate_decomposition(query, repeated), LanguageRejected)
    language = LanguageModel.__new__(LanguageModel)
    for payload in (["A man is walking."], {"candidates": ["A man is walking."]}):
        language.ask = lambda *args: payload
        assert language.rewrite(query, graph, [], 1, 30) == ["A man is walking."]
    language.ask = lambda *args: ["A man is walking.", "He walks."]
    must_fail(lambda: language.rewrite(query, graph, [], 1, 30), LanguageRejected)
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [0] + text.split() + [1]
    assert lexical_rejection("He never walks.", "He walks.", Tokenizer(), 30) == "temporal_or_negation_change"
    assert lexical_rejection("He holds 2 cups.", "He holds 3 cups.", Tokenizer(), 30) == "quantity_change"

    class Text:
        tokenizer, max_length = Tokenizer(), 30
        def spans(self, query, evidence):
            return np.ones((len(evidence), 2))
        def encode(self, texts, **kwargs):
            return [np.ones((2, 2)) for _ in texts], []
    class Retrieval:
        def score(self, features, ids):
            return np.tile([.9, .1, .1], (len(features), 1))
    class Bank:
        def __init__(self, risk):
            self.risk = risk
        def analyze(self, evidence, *args):
            return [dict(e, protected=True, risk=self.risk, G=1, D=1) for e in evidence]
    class LLM:
        rewrites = 0
        def decompose(self, query):
            return graph
        def rewrite(self, *args):
            self.rewrites += 1
            return ["A man is walking.", "A man never walks.", query]
        def judge(self, *args):
            return dict(preserved=[1, 1], hard_violation=False, drift=0)
    llm = LLM()
    result, audit = process_query(query, original, ["a", "b", "c"], Bank(0), ["x", "y", "z"],
                                  Retrieval(), Text(), llm, cfg, "fusion")
    np.testing.assert_array_equal(result, original)
    assert llm.rewrites == 0 and audit["reason"] == "gate_closed"
    result, audit = process_query(query, original, ["a", "b", "c"], Bank(1), ["x", "y", "z"],
                                  Retrieval(), Text(), llm, cfg, "fusion")
    assert llm.rewrites == 1 and audit["reason"] == "selected"
    assert len(audit["selected_candidate_indices"]) == 1
    np.testing.assert_allclose(result, (1 - cfg.rho) * original + cfg.rho * np.array([.9, .1, .1]))
    assert len(audit["candidates"]) == 3
    print("PASS: equations, leave-one-out, role errors, guards, gate-before-generation, OFF/fallback, selection/fusion")


def gpu_check(device):
    import torch
    import h5py
    from .backbones import A3Retriever, HolmesRetriever, WORKSPACE, normalized_text, RobertaFeatures
    from utils.holmes_ar import read_captions, write_captions
    from Rewriter.evaluate import score as native_holmes_score, combined
    rows = read_captions(WORKSPACE / "tvr/TextData/tvrval.caption.txt")
    videos = list(dict.fromkeys(x.split("#")[0] for x in rows))[:3]
    ids = list(rows)[:2]
    feature_file = WORKSPACE / "tvr/TextData/roberta_tvr_query_feat_extract.hdf5"
    with h5py.File(feature_file, "r") as features:
        query = [features[cid][...] for cid in ids]
    a3 = A3Retriever(WORKSPACE / "checkpoints/a3prvr/tvr.ckpt", WORKSPACE / "checkpoints/a3prvr/tvr_config.json",
                     WORKSPACE / "tvr/FeatureData/i3d_resnet/feature.hdf5",
                     WORKSPACE / "tvr/FeatureData/new_clip_vit_32_tvr_vid_features.hdf5", device, 2)
    actual = a3.score(query, videos)
    feat, mask = normalized_text(query, a3.cfg.q_feat_size, a3.cfg.max_desc_l, a3.device)
    with torch.inference_mode():
        first, fm, second, sm = [x.to(a3.device) for x in a3.contexts[tuple(videos)][0]]
        ns = a3.model.get_pred_from_raw_query(feat, mask, video_feat=first, video_feat_mask=fm)
        nc = a3.model.get_pred_from_raw_query(feat, mask, video_feat=second, video_feat_mask=sm, phrase_action_branch=True)
        native = a3.cfg.fusion_weight * ns + (1 - a3.cfg.fusion_weight) * nc
        np.testing.assert_allclose(actual[:, :2], native.cpu().numpy(), atol=2e-6, rtol=2e-6)
    single = a3.score(query[:1], videos[:1])
    assert single.shape == (1, 1) and np.isfinite(single).all()
    np.testing.assert_allclose(single[0, 0], actual[0, 0], atol=2e-6, rtol=2e-6)
    assert not any(p.requires_grad for p in a3.model.parameters())
    print("PASS: official A3PRVR checkpoint strict load, native two-branch scores, Q=1/V=1, frozen parameters")
    del a3
    holmes = HolmesRetriever(WORKSPACE / "Holmes/tvr-multi-tvr.ckpt", "multisrc", WORKSPACE, device, 2)
    selected = {cid: q for cid, q in rows.items() if cid.split("#")[0] in videos}
    with h5py.File(feature_file, "r") as features:
        raw = [features[cid][...] for cid in selected]
    actual = holmes.score(raw, videos)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        captions = root / "captions.txt"
        write_captions(captions, selected)
        native_holmes_score(SimpleNamespace(device=device, checkpoint=WORKSPACE / "Holmes/tvr-multi-tvr.ckpt",
            backbone="multisrc", data_root=WORKSPACE, captions=captions, video_captions=captions,
            features=feature_file, query_batch=7, context_batch=2, output=root / "native.npz", preflight=False))
        with np.load(root / "native.npz") as native:
            np.testing.assert_allclose(actual, combined(native), atol=2e-6, rtol=2e-6)
    single = holmes.score(raw[:1], videos)
    np.testing.assert_allclose(single, actual[:1], atol=2e-6, rtol=2e-6)
    print("PASS: real Holmes adapter equals existing evaluator; singleton query preserved")
    text = RobertaFeatures(WORKSPACE / "tvr/query_only/roberta-base_tuned_model", device)
    evidence = [dict(start=0, end=5), dict(start=6, end=11)]
    encoded, _ = text.encode(["Penny walks.", "A man walks before sitting."])
    assert all(x.ndim == 2 and x.shape[1] == 768 and len(x) <= 28 for x in encoded)
    vectors = text.spans("Penny walks.", evidence)
    assert vectors.shape == (2, 768) and np.isfinite(vectors).all()
    original_features, _ = text.encode([rows[cid] for cid in ids])
    with h5py.File(feature_file, "r") as feature_cache:
        for cid, feature in zip(ids, original_features):
            np.testing.assert_allclose(feature, feature_cache[cid][...], atol=2e-5, rtol=2e-5)
    print("PASS: real RoBERTa token shapes, contextual spans, and existing feature equality")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu")
    args = parser.parse_args()
    offline_check()
    if args.gpu:
        gpu_check(args.gpu)
