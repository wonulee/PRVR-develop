"""Frozen native retrieval. A3PRVR source is retained unchanged in third_party."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.nn import functional as F

WORKSPACE = Path(__file__).resolve().parents[2]


def checked_feature(value, dimension, name):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != dimension or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name} feature: {value.shape}; expected [L,{dimension}]")
    return value


def pad_features(values, device):
    from torch.nn.utils.rnn import pad_sequence
    tensors = [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in values]
    padded = pad_sequence(tensors, batch_first=True).to(device)
    lengths = torch.tensor([len(x) for x in values], device=device)
    mask = (torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]).float()
    return padded, mask


def normalized_text(values, dimension, max_length, device):
    normalized = []
    for value in values:
        value = checked_feature(value, dimension, "text")[:max_length]
        # Native data loaders normalize each token with this exact denominator.
        normalized.append(value / (np.linalg.norm(value, axis=-1, keepdims=True) + 1e-5))
    return pad_features(normalized, device)


class RobertaFeatures:
    """Same layer/special-token convention as Holmes' existing feature extractor."""
    def __init__(self, model, device, max_length=30, layer=-2):
        from transformers import RobertaForMaskedLM, RobertaTokenizerFast
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model, do_lower_case=False)
        self.model = RobertaForMaskedLM.from_pretrained(model).to(device).eval()
        self.device, self.max_length, self.layer = device, max_length, layer
        self.full_length = self.model.config.max_position_embeddings - 2
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts, full_context=False):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot encode empty text")
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.full_length if full_context else self.max_length,
                                 return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping").numpy()
        lengths = encoded["attention_mask"].sum(-1).tolist()
        result = self.model.roberta(**{k: v.to(self.device) for k, v in encoded.items()},
                                    output_hidden_states=True, return_dict=True)
        hidden = result.hidden_states[self.layer].float().cpu().numpy()
        features = [hidden[i, 1:n - 1].copy() for i, n in enumerate(lengths)]
        spans = [offsets[i, 1:n - 1].copy() for i, n in enumerate(lengths)]
        if any(not len(x) for x in features):
            raise ValueError("No valid RoBERTa tokens")
        return features, spans

    def spans(self, query, evidence):
        features, offsets = self.encode([query], full_context=True)
        vectors = []
        for row in evidence:
            start, end = row["start"], row["end"]
            selected = (offsets[0][:, 0] < end) & (offsets[0][:, 1] > start)
            if not selected.any() or offsets[0][selected, 0].min() > start or offsets[0][selected, 1].max() < end:
                raise ValueError("Evidence span is empty or truncated by the text encoder")
            vectors.append(features[0][selected].mean(axis=0))
        return np.asarray(vectors)


class HolmesRetriever:
    def __init__(self, checkpoint, backbone, data_root, device, context_batch=64):
        from Rewriter.evaluate import load_model
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The existing Holmes context encoder requires CUDA")
        torch.cuda.set_device(self.device)
        self.model, self.cfg = load_model(checkpoint, backbone, self.device)
        self.model.requires_grad_(False)
        self.backbone, self.data_root, self.context_batch = backbone, Path(data_root), context_batch
        self.contexts = {}

    @torch.inference_mode()
    def prepare(self, video_ids):
        key = tuple(video_ids)
        if not key or len(set(key)) != len(key):
            raise ValueError("Video IDs must be nonempty and unique")
        if key in self.contexts:
            return
        from torch.utils.data import DataLoader
        from Datasets.data_provider import VisDataSet4PRVR, collate_frame_val
        from Rewriter.evaluate import check_features
        visual = self.data_root / "tvr/FeatureData" / self.cfg["visual_feature"] / "feature.hdf5"
        check_features(visual, video_ids, self.cfg["visual_feat_dim"])
        dataset = VisDataSet4PRVR(str(visual), None, self.cfg, video_ids=list(video_ids))
        loader = DataLoader(dataset, batch_size=self.context_batch, collate_fn=collate_frame_val, num_workers=0)
        blocks = []
        for batch in loader:
            clips, frames, mask = [x.to(self.device) for x in batch[:3]]
            encoded_frames, encoded_clips = self.model.encode_context(clips, frames, mask)
            blocks.append((encoded_frames.cpu(), encoded_clips.cpu(), mask.cpu()))
        self.contexts[key] = blocks

    @torch.inference_mode()
    def score(self, raw_features, video_ids):
        self.prepare(video_ids)
        feat, mask = normalized_text(raw_features, self.cfg["q_feat_size"], self.cfg["max_desc_l"], self.device)
        query = self.model.encode_query(feat, mask)
        frame_query = self.model.encode_frame_query(feat, mask) if self.backbone == "multisrc" else query
        blocks = []
        for frames, clips, video_mask in self.contexts[tuple(video_ids)]:
            frames, clips, video_mask = [x.to(self.device) for x in (frames, clips, video_mask)]
            cs = self.model.get_clip_scale_scores(query, clips)
            fs = (self.model.get_frame_scale_scores(frame_query, frames, mask, video_mask)
                  if self.backbone == "multisrc" else self.model.get_clip_scale_scores(frame_query, frames))
            blocks.append((cs * self.cfg["clip_scale_w"] + fs * self.cfg["frame_scale_w"]).cpu().numpy())
        scores = np.concatenate(blocks, axis=1)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite Holmes scores")
        return scores


class A3Retriever:
    """Official TVR two-branch inference; feature files stay separate and read-only."""
    def __init__(self, checkpoint, config, i3d_features, clip_features, device, context_batch=64):
        from easydict import EasyDict
        sys.path.insert(0, str(WORKSPACE / "third_party/A3PRVR"))
        from method.model import MS_SL_Net
        self.cfg = EasyDict(json.loads(Path(config).read_text()))
        if self.cfg.dataset_name != "tvr" or not self.cfg.best_frame_loss:
            raise ValueError("This adapter implements the official TVR best_frame_loss inference")
        if not self.cfg.phrase_action_branch or not self.cfg.flow_feat:
            raise ValueError("Expected the official TVR I3D/CLIP two-branch configuration")
        self.device, self.context_batch = torch.device(device), context_batch
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = MS_SL_Net(self.cfg)
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # The public TVR checkpoint has model + epoch, without model_cfg.
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["model"].items()}, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.i3d, self.clip = Path(i3d_features), Path(clip_features)
        self.contexts = {}

    @torch.inference_mode()
    def encode_context_pair(self, first, first_mask, second, second_mask):
        first = self.model.encode_context(first, first_mask)
        second = self.model.encode_context(second, second_mask, phrase_action_branch=True)
        cfg = self.cfg
        if cfg.cross_branch_fusion:
            if cfg.deformable_attn:
                if cfg.cross_attn_q == "i3d_q":
                    first = self.model.cross_branch_fusion_A1(first.transpose(1, 2), second.transpose(1, 2)).transpose(1, 2)
                elif cfg.cross_attn_q == "clip_q":
                    second = self.model.cross_branch_fusion_A1(second.transpose(1, 2), first.transpose(1, 2)).transpose(1, 2)
                elif cfg.cross_attn_q == "double_q":
                    new_first = self.model.cross_branch_fusion_A1(first.transpose(1, 2), second.transpose(1, 2)).transpose(1, 2)
                    second = self.model.cross_branch_fusion_B1(second.transpose(1, 2), first.transpose(1, 2)).transpose(1, 2)
                    first = new_first
                else:
                    raise ValueError(f"Unknown cross_attn_q: {cfg.cross_attn_q}")
            else:
                first, _ = self.model.cross_branch_fusion_A1(first, second, first_mask, second_mask)
        return first, second

    @torch.inference_mode()
    def prepare(self, video_ids):
        from method.data_provider import uniform_feature_sampling, l2_normalize_np_array
        key = tuple(video_ids)
        if not key or len(set(key)) != len(key):
            raise ValueError("Video IDs must be nonempty and unique")
        if key in self.contexts:
            return
        blocks = []
        with h5py.File(self.i3d, "r") as first_file, h5py.File(self.clip, "r") as second_file:
            for start in range(0, len(key), self.context_batch):
                first, second = [], []
                for vid in key[start:start + self.context_batch]:
                    if vid not in first_file or vid not in second_file:
                        raise ValueError(f"Missing A3PRVR feature for video {vid}")
                    f, s = first_file[vid], second_file[vid]
                    # Also accepts the upstream joint HDF5's i3d_feat / clip_feat groups.
                    f = f["i3d_feat"] if isinstance(f, h5py.Group) else f
                    s = s["clip_feat"] if isinstance(s, h5py.Group) else s
                    f = checked_feature(f[...], self.cfg.visual_feat_dim, "A3 I3D")
                    s = checked_feature(s[...], self.cfg.visual_flow_feat_dim, "A3 CLIP")
                    if self.cfg.deformable_attn and (len(f) < 2 or len(s) < 2):
                        raise ValueError("Upstream Q-DTA requires >=2 temporal positions; no synthetic padding is added")
                    first.append(l2_normalize_np_array(uniform_feature_sampling(f, self.cfg.max_ctx_l)))
                    second.append(l2_normalize_np_array(uniform_feature_sampling(s, self.cfg.max_ctx_l)))
                f, fm = pad_features(first, self.device)
                s, sm = pad_features(second, self.device)
                f, s = self.encode_context_pair(f, fm, s, sm)
                blocks.append((f.cpu(), fm.cpu(), s.cpu(), sm.cpu()))
        self.contexts[key] = blocks

    def branch_scores(self, feat, mask, video_feat, video_mask, second=False):
        query, _ = self.model.encode_query(feat, mask, phrase_action_branch=second)
        query = query.reshape(feat.shape[0], self.cfg.hidden_size)
        mapping = self.model.mapping_linear_phrase_action if second else self.model.mapping_linear
        key = F.normalize(mapping[0](video_feat), dim=-1)
        # Same projected cosine + masked temporal max as native inference.
        # Explicit dimensions avoid upstream squeeze() dropping Q=1 or V=1.
        similarities = torch.einsum("qd,vtd->qvt", F.normalize(query, dim=-1), key)
        return similarities.masked_fill(video_mask.unsqueeze(0).eq(0), -1e9).max(dim=-1).values

    @torch.inference_mode()
    def score(self, raw_features, video_ids):
        self.prepare(video_ids)
        feat, mask = normalized_text(raw_features, self.cfg.q_feat_size, self.cfg.max_desc_l, self.device)
        blocks = []
        for f, fm, s, sm in self.contexts[tuple(video_ids)]:
            f, fm, s, sm = [x.to(self.device) for x in (f, fm, s, sm)]
            i3d = self.branch_scores(feat, mask, f, fm)
            clip = self.branch_scores(feat, mask, s, sm, second=True)
            blocks.append((self.cfg.fusion_weight * i3d + (1 - self.cfg.fusion_weight) * clip).cpu().numpy())
        scores = np.concatenate(blocks, axis=1)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite A3PRVR scores")
        return scores
