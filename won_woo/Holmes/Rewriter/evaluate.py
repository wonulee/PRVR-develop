"""Score all queries against all videos, using the checkpoint's own configuration."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from utils.holmes_ar import positive, read_captions

ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint, backbone, device):
    import torch
    sys.path.insert(0, str(ROOT / backbone))
    from Models.builder import get_models
    from Opts.optimization import WarmupLinearSchedule
    # Reviewed local arithmetic-only class; no custom deserialization hooks.
    # Keep the restricted unpickler enabled, even for legacy optimizer metadata.
    with torch.serialization.safe_globals([WarmupLinearSchedule]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    model = get_models(cfg)
    model.load_state_dict({k.removeprefix("module."): v for k, v in saved["state_dict"].items()})
    if not np.isfinite(cfg["edl_tau"]) or cfg["edl_tau"] <= 0:
        raise ValueError("Invalid checkpoint edl_tau")
    if not all(np.isfinite(cfg[k]) and cfg[k] >= 0 for k in ("clip_scale_w", "frame_scale_w")):
        raise ValueError("Invalid branch weights")
    if cfg["clip_scale_w"] + cfg["frame_scale_w"] <= 0:
        raise ValueError("Both branch weights are zero")
    return model.to(device).eval(), cfg


def check_features(path, ids, dimension):
    import h5py
    with h5py.File(path, "r") as features:
        for key in ids:
            if key not in features:
                raise ValueError(f"Missing feature {key} in {path}")
            shape = features[key].shape
            if len(shape) != 2 or shape[0] < 1 or shape[1] != dimension:
                raise ValueError(f"Invalid feature {key}: {shape}")


def score(args):
    import torch
    from torch.utils.data import DataLoader
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Holmes context encoder requires CUDA")
    torch.cuda.set_device(device)
    model, cfg = load_model(args.checkpoint, args.backbone, device)
    from Datasets.data_provider import (
        VisDataSet4PRVR, TxtDataSet4PRVR, collate_frame_val, collate_text_val,
    )
    originals = read_captions(args.video_captions)
    videos = list(dict.fromkeys(cid.split("#", 1)[0] for cid in originals))
    rows = read_captions(args.captions) if args.captions.read_text().strip() else {}
    if not set(rows).issubset(originals):
        raise ValueError("Scored query IDs must belong to the original split")
    visual = args.data_root / "tvr/FeatureData" / cfg["visual_feature"] / "feature.hdf5"
    check_features(visual, videos, cfg["visual_feat_dim"])
    if rows:
        check_features(args.features, rows, cfg["q_feat_size"])
    if args.preflight:
        print(json.dumps(dict(queries=len(rows), videos=len(videos), backbone=args.backbone,
                              checkpoint=str(args.checkpoint), edl_tau=cfg["edl_tau"])))
        return

    clip_scores = np.empty((len(rows), len(videos)), dtype=np.float32)
    frame_scores = np.empty_like(clip_scores)
    ordered_ids = list(rows)
    positions = {cid: i for i, cid in enumerate(ordered_ids)}
    contexts = []
    with torch.inference_mode():
        if rows:
            loader = DataLoader(
                VisDataSet4PRVR(str(visual), None, cfg, video_ids=videos),
                batch_size=args.context_batch, collate_fn=collate_frame_val, num_workers=0,
            )
            for batch in loader:
                clips, frames, mask = [x.to(device) for x in batch[:3]]
                encoded_frames, encoded_clips = model.encode_context(clips, frames, mask)
                contexts.append((encoded_frames.cpu(), encoded_clips.cpu(), mask.cpu()))
            loader = DataLoader(
                TxtDataSet4PRVR(str(args.captions), str(args.features), cfg),
                batch_size=args.query_batch, collate_fn=collate_text_val, num_workers=0,
            )
            completed = 0
            for batch in loader:
                feat, mask = [x.to(device) for x in batch[:2]]
                indices = [positions[cid] for cid in batch[-1]]
                # Encode once per query batch, then stream the candidate video blocks.
                query = model.encode_query(feat, mask)
                frame_query = model.encode_frame_query(feat, mask) if args.backbone == "multisrc" else query
                offset = 0
                for frames, clips, video_mask in contexts:
                    frames, clips, video_mask = [x.to(device) for x in (frames, clips, video_mask)]
                    cs = model.get_clip_scale_scores(query, clips)
                    fs = (model.get_frame_scale_scores(frame_query, frames, mask, video_mask)
                          if args.backbone == "multisrc"
                          else model.get_clip_scale_scores(frame_query, frames))
                    count = clips.shape[0]
                    clip_scores[indices, offset:offset + count] = cs.cpu().numpy()
                    frame_scores[indices, offset:offset + count] = fs.cpu().numpy()
                    offset += count
                completed += len(indices)
                print(f"Scored {completed}/{len(rows)} against {len(videos)} videos", flush=True)
    if not np.isfinite(clip_scores).all() or not np.isfinite(frame_scores).all():
        raise ValueError("Nonfinite model scores")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, query_ids=np.array(ordered_ids, dtype=str), video_ids=np.array(videos),
                 clip=clip_scores, frame=frame_scores, tau=cfg["edl_tau"],
                 clip_weight=cfg["clip_scale_w"], frame_weight=cfg["frame_scale_w"])
    temporary.replace(args.output)


def evidence(scores, tau):
    # This is the repository EDL_loss.get_alpha formula, minus one.
    return np.exp(np.tanh(scores.astype(np.float64)) / tau)


def dst_probabilities(first, second):
    """Exact two-opinion Dempster combination with uniform base rate, O(Q*K).

    e_fused = e_first + e_second + e_first*e_second/K follows by substituting
    b=e/S and u=K/S into Dempster's rule; it avoids a Q*K*K conflict tensor.
    """
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] == 0:
        raise ValueError("Evidence matrices must have the same nonempty class axis")
    fused = first + second + first * second / first.shape[1]
    alpha = fused + 1
    return alpha / alpha.sum(axis=1, keepdims=True)


def uncertainty(scores):
    k = len(scores["video_ids"])
    return np.maximum(
        k / (evidence(scores["clip"], float(scores["tau"])) + 1).sum(axis=1),
        k / (evidence(scores["frame"], float(scores["tau"])) + 1).sum(axis=1),
    )


def combined(scores):
    return scores["clip"] * float(scores["clip_weight"]) + scores["frame"] * float(scores["frame_weight"])


def variants(original, rewritten, threshold):
    if not np.array_equal(original["video_ids"], rewritten["video_ids"]):
        raise ValueError("Candidate video order differs")
    for key in ("tau", "clip_weight", "frame_weight"):
        if float(original[key]) != float(rewritten[key]):
            raise ValueError(f"Checkpoint scoring configuration differs: {key}")
    mapping = {cid: i for i, cid in enumerate(original["query_ids"])}
    if len(mapping) != len(original["query_ids"]) or len(set(rewritten["query_ids"])) != len(rewritten["query_ids"]):
        raise ValueError("Duplicate score IDs")
    indices = np.array([mapping[cid] for cid in rewritten["query_ids"]], dtype=int)
    base = combined(original)
    new = combined(rewritten)
    u = uncertainty(original)
    replacement, average, dst, adaptive = [base.copy() for _ in range(4)]
    replacement[indices] = new
    average[indices] = (base[indices] + new) / 2
    fused = np.zeros(new.shape, dtype=np.float64)
    for branch in ("clip", "frame"):
        fused += float(original[branch + "_weight"]) * dst_probabilities(
            evidence(original[branch][indices], float(original["tau"])),
            evidence(rewritten[branch], float(original["tau"])),
        )
    dst[indices] = fused
    active = u[indices] >= threshold
    adaptive[indices[active]] = fused[active]
    return dict(original=base, rewritten=replacement, average=average, dst=dst, adaptive_dst=adaptive), u, indices[active]


def ranks(scores, video_ids, query_ids):
    lookup = {vid: i for i, vid in enumerate(video_ids)}
    gt = np.array([lookup[cid.split("#", 1)[0]] for cid in query_ids])
    # Stable ties follow the fixed candidate order for all methods.
    order = np.argsort(-scores, axis=1, kind="stable")
    result = (order == gt[:, None]).argmax(axis=1) + 1
    return result, np.asarray(video_ids)[order[:, 0]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backbone", choices=("src", "multisrc"), default="multisrc")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--captions", type=Path, required=True)
    p.add_argument("--video-captions", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-batch", type=positive, default=16)
    p.add_argument("--context-batch", type=positive, default=64)
    p.add_argument("--preflight", action="store_true")
    score(p.parse_args())


if __name__ == "__main__":
    main()
