"""Frozen Holmes scores for small query blocks against a fixed full gallery.

Context encodings are computed once and kept on one GPU.  The caller groups
queries by source caption and selects subsets; this module never writes a
query-by-video score matrix or changes the teacher's scoring equations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HOLMES = ROOT / "Holmes"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def caption_video_ids(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(dict.fromkeys(line.split(maxsplit=1)[0].split("#", 1)[0]
                                  for line in stream if line.strip()))


class NativeScorer:
    """Native similarity and evidential uncertainty, preserving input ID order.

    ``score(query_ids)`` returns ``(scores, uncertainty)`` with shapes Q x V
    and Q. Scores use the checkpoint's clip/frame weights; uncertainty is the
    maximum branch uncertainty over the same entire fixed gallery. Input query
    features are normalized and cropped exactly as in TxtDataSet4PRVR.
    """

    def __init__(self, checkpoint, features, data_root, device="cuda:1",
                 backbone="multisrc", query_batch=128, video_batch=512,
                 context_batch=64, cache_path=None, video_ids=None, seed=2026):
        if backbone not in ("src", "multisrc"):
            raise ValueError("backbone must be src or multisrc")
        if min(query_batch, video_batch, context_batch) <= 0:
            raise ValueError("Batch sizes must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The native Holmes context encoder requires CUDA")
        torch.cuda.set_device(self.device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.seed = seed
        self.backbone = backbone
        self.checkpoint = Path(checkpoint).resolve()
        self.feature_path = Path(features).resolve()
        self.data_root = Path(data_root).resolve()
        self.query_batch = query_batch
        self.video_batch = video_batch
        self.context_batch = context_batch
        self.video_ids = list(video_ids) if video_ids is not None else caption_video_ids(
            self.data_root / "tvr/TextData/tvrtrain.caption.txt")
        if not self.video_ids or len(set(self.video_ids)) != len(self.video_ids):
            raise ValueError("The gallery must contain unique, nonempty video IDs")
        sys.path.insert(0, str(HOLMES))
        from Rewriter.evaluate import load_model
        self.model, self.cfg = load_model(self.checkpoint, backbone, self.device)
        from Datasets.data_provider import (
            VisDataSet4PRVR, collate_frame_val, l2_normalize_np_array,
        )
        self._visual_dataset = VisDataSet4PRVR
        self._visual_collate = collate_frame_val
        self._normalize = l2_normalize_np_array
        self.features = h5py.File(self.feature_path, "r")
        self.visual_path = (self.data_root / "tvr/FeatureData" /
                            self.cfg["visual_feature"] / "feature.hdf5").resolve()
        stat = self.visual_path.stat()
        self.manifest = dict(
            format_version=1, checkpoint=str(self.checkpoint),
            checkpoint_sha256=file_sha256(self.checkpoint), backbone=backbone,
            visual_path=str(self.visual_path), visual_size=stat.st_size,
            visual_mtime_ns=stat.st_mtime_ns, cfg=self.cfg,
            video_ids_sha256=hashlib.sha256(
                "\n".join(self.video_ids).encode("utf-8")).hexdigest(),
            seed=seed,
        )
        try:
            self._contexts(Path(cache_path) if cache_path is not None else None)
        except BaseException:
            self.features.close()
            raise

    def _contexts(self, cache_path):
        if cache_path is not None and cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu", weights_only=True)
            if saved.get("manifest") != self.manifest or saved.get("video_ids") != self.video_ids:
                raise ValueError("Context cache provenance or gallery order differs")
            frames, clips, masks = (saved[name] for name in ("frames", "clips", "masks"))
            self._check_context_shapes(frames, clips, masks)
            self.frames, self.clips, self.masks = [x.to(self.device) for x in (frames, clips, masks)]
            return
        from torch.utils.data import DataLoader
        dataset = self._visual_dataset(str(self.visual_path), None, self.cfg,
                                       video_ids=self.video_ids)
        loader = DataLoader(dataset, batch_size=self.context_batch,
                            collate_fn=self._visual_collate, num_workers=0,
                            shuffle=False)
        frame_blocks, clip_blocks, mask_blocks = [], [], []
        completed = 0
        try:
            with torch.inference_mode():
                for batch in loader:
                    clips, frames, mask = [x.to(self.device) for x in batch[:3]]
                    encoded_frames, encoded_clips = self.model.encode_context(clips, frames, mask)
                    if mask.shape[1] != encoded_frames.shape[1]:
                        mask = torch.nn.functional.pad(
                            mask, (0, encoded_frames.shape[1] - mask.shape[1]))
                    frame_blocks.append(encoded_frames)
                    clip_blocks.append(encoded_clips)
                    mask_blocks.append(mask)
                    completed += len(batch[4])
                    if completed % (self.context_batch * 16) == 0 or completed == len(self.video_ids):
                        print(f"Encoded gallery {completed}/{len(self.video_ids)}", flush=True)
            self.frames = torch.cat(frame_blocks)
            self.clips = torch.cat(clip_blocks)
            self.masks = torch.cat(mask_blocks)
        finally:
            dataset.visual_feat.close()
        self._check_context_shapes(self.frames, self.clips, self.masks)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
            try:
                torch.save(dict(manifest=self.manifest, video_ids=self.video_ids,
                                frames=self.frames.cpu(), clips=self.clips.cpu(),
                                masks=self.masks.cpu()), temporary)
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)

    def _check_context_shapes(self, frames, clips, masks):
        n = len(self.video_ids)
        d = self.cfg["hidden_size"]
        if (frames.ndim != 3 or clips.ndim != 3 or masks.ndim != 2 or
                frames.shape[0] != n or clips.shape[0] != n or masks.shape[0] != n or
                frames.shape[1] != masks.shape[1] or
                frames.shape[2] != d or clips.shape[2] != d):
            raise ValueError("Invalid native context cache dimensions")
        if not all(x.dtype == torch.float32 for x in (frames, clips, masks)):
            raise ValueError("Context cache must contain float32 native encodings")
        if not all(torch.isfinite(x).all().item() for x in (frames, clips, masks)):
            raise ValueError("Nonfinite context cache")

    def _query_inputs(self, query_ids):
        arrays = []
        for cid in query_ids:
            if cid not in self.features:
                raise ValueError(f"Missing feature {cid} in {self.feature_path}")
            array = self.features[cid][...]
            if (array.ndim != 2 or array.shape[0] < 1 or
                    array.shape[1] != self.cfg["q_feat_size"] or
                    not np.isfinite(array).all()):
                raise ValueError(f"Invalid query feature {cid}: {array.shape}")
            arrays.append(self._normalize(array)[:self.cfg["max_desc_l"]])
        max_length = max(len(x) for x in arrays)
        feat = torch.zeros(len(arrays), max_length, self.cfg["q_feat_size"], dtype=torch.float32)
        mask = torch.zeros(len(arrays), max_length, dtype=torch.float32)
        for i, array in enumerate(arrays):
            feat[i, :len(array)] = torch.from_numpy(array)
            mask[i, :len(array)] = 1
        return feat.to(self.device), mask.to(self.device)

    def _native_scores(self, feat, mask):
        query = self.model.encode_query(feat, mask)
        frame_query = (self.model.encode_frame_query(feat, mask)
                       if self.backbone == "multisrc" else query)
        n = feat.shape[0]
        k = len(self.video_ids)
        output = np.empty((n, k), dtype=np.float32)
        evidence_sums = np.zeros((2, n), dtype=np.float64)
        tau = float(self.cfg["edl_tau"])
        for start in range(0, k, self.video_batch):
            stop = min(k, start + self.video_batch)
            clip = self.model.get_clip_scale_scores(query, self.clips[start:stop])
            frame = (self.model.get_frame_scale_scores(
                frame_query, self.frames[start:stop], mask, self.masks[start:stop])
                if self.backbone == "multisrc" else
                self.model.get_clip_scale_scores(frame_query, self.frames[start:stop]))
            clip, frame = clip.cpu().numpy(), frame.cpu().numpy()
            output[:, start:stop] = (clip * float(self.cfg["clip_scale_w"]) +
                                    frame * float(self.cfg["frame_scale_w"]))
            for branch, raw in enumerate((clip, frame)):
                evidence_sums[branch] += (np.exp(np.tanh(raw.astype(np.float64)) / tau) + 1).sum(axis=1)
        uncertainty = np.maximum(k / evidence_sums[0], k / evidence_sums[1])
        if not np.isfinite(output).all() or not np.isfinite(uncertainty).all():
            raise ValueError("Nonfinite native query scores")
        return output, uncertainty

    def score(self, query_ids):
        query_ids = list(query_ids)
        output = np.empty((len(query_ids), len(self.video_ids)), dtype=np.float32)
        uncertainty = np.empty(len(query_ids), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(query_ids), self.query_batch):
                stop = min(len(query_ids), start + self.query_batch)
                feat, mask = self._query_inputs(query_ids[start:stop])
                output[start:stop], uncertainty[start:stop] = self._native_scores(feat, mask)
        return output, uncertainty

    def validate_native(self, query_ids):
        """Compare our feature loading/order with the original native loader."""
        query_ids = list(query_ids)
        if not query_ids or len(set(query_ids)) != len(query_ids):
            raise ValueError("Validation requires unique nonempty query IDs")
        from Datasets.data_provider import TxtDataSet4PRVR, collate_text_val
        actual, actual_u = self.score(query_ids)
        with tempfile.TemporaryDirectory(prefix="holmes_native_check_") as directory:
            captions = Path(directory) / "queries.caption.txt"
            captions.write_text("".join(f"{cid} feature verification\n" for cid in query_ids), encoding="utf-8")
            dataset = TxtDataSet4PRVR(str(captions), str(self.feature_path), self.cfg)
            try:
                batch = collate_text_val([dataset[i] for i in range(len(query_ids))])
                feat, mask = [x.to(self.device) for x in batch[:2]]
                with torch.inference_mode():
                    native, native_u = self._native_scores(feat, mask)
                positions = {cid: i for i, cid in enumerate(batch[-1])}
                order = [positions[cid] for cid in query_ids]
                native, native_u = native[order], native_u[order]
            finally:
                dataset.text_feat.close()
        np.testing.assert_allclose(actual, native, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual_u, native_u, rtol=1e-5, atol=1e-8)
        return dict(queries=len(query_ids), videos=len(self.video_ids),
                    maximum_score_difference=float(np.max(np.abs(actual - native))),
                    maximum_uncertainty_difference=float(np.max(np.abs(actual_u - native_u))))

    def close(self):
        self.features.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--video-captions", type=Path, required=True)
    parser.add_argument("--query-id", action="append", required=True)
    parser.add_argument("--video-limit", type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--query-batch", type=int, default=128)
    parser.add_argument("--video-batch", type=int, default=512)
    parser.add_argument("--context-batch", type=int, default=64)
    args = parser.parse_args()
    videos = caption_video_ids(args.video_captions)
    if args.video_limit is not None:
        if args.video_limit <= 0:
            parser.error("--video-limit must be positive")
        videos = videos[:args.video_limit]
    started = time.monotonic()
    with NativeScorer(args.checkpoint, args.features, args.data_root,
                      device=args.device, query_batch=args.query_batch,
                      video_batch=args.video_batch, context_batch=args.context_batch,
                      cache_path=args.cache_path, video_ids=videos) as scorer:
        result = scorer.validate_native(args.query_id)
        result["seconds"] = time.monotonic() - started
        result["context_bytes"] = sum(x.numel() * x.element_size() for x in
                                      (scorer.frames, scorer.clips, scorer.masks))
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
