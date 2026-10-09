"""Equations from experiments/evidence_risk_rewriting_design.md; no model imports."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Settings:
    risk_weights: list
    selection_weights: list  # Q, S, coverage, preservation, drift penalty
    top_m: int
    neighbors: int
    evidence_temperature: float
    query_temperature: float
    fusion_temperature: float
    risk_threshold: float
    grounding_threshold: float
    discrimination_threshold: float
    preservation_threshold: float
    improvement_margin: float
    rho: float
    candidates: int

    def __post_init__(self):
        values = list(self.__dict__.values())
        flat = [x for v in values for x in (v if isinstance(v, list) else [v])]
        if not np.isfinite(flat).all():
            raise ValueError("Settings must be finite")
        if len(self.risk_weights) != 3 or min(self.risk_weights) < 0 or not np.isclose(sum(self.risk_weights), 1):
            raise ValueError("Three nonnegative risk weights must sum to one")
        if len(self.selection_weights) != 5 or min(self.selection_weights) < 0:
            raise ValueError("Five nonnegative selection weights are required")
        for name in ("top_m", "neighbors", "candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("evidence_temperature", "query_temperature", "fusion_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("risk_threshold", "grounding_threshold", "discrimination_threshold", "preservation_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0 <= self.rho <= 0.5 or self.improvement_margin < 0:
            raise ValueError("rho must be in [0,0.5] and improvement_margin nonnegative")


class InsufficientRoleBank(ValueError):
    """Query must use original scores when its role has no calibrated train bank."""


def finite_array(value, ndim, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected nonempty finite {ndim}-D array")
    return array


def unit_vectors(value):
    value = finite_array(value, 2, "embeddings")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if (norm <= np.finfo(value.dtype).eps).any():
        raise ValueError("Zero-norm evidence embedding")
    return value / norm


def softmax(value, temperature):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid softmax input/temperature")
    logits = (value - value.max(axis=-1, keepdims=True)) / temperature
    if not np.isfinite(logits).all():
        raise ValueError("Temperature produced nonfinite logits")
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def concentration(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 1 or scores.shape[-1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Entropy requires at least two finite candidate scores")
    probabilities = softmax(scores, temperature)
    # x log x has limit zero at x=0; do not add a distribution-altering epsilon.
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    entropy = -(probabilities * logs).sum(axis=-1)
    return np.clip(1 - entropy / np.log(scores.shape[-1]), 0, 1)


def ecdf(reference, value):
    return np.searchsorted(np.sort(reference), value, side="right") / len(reference)


class RiskBank:
    """Train-only reference scores [E,V_ref], contextual embeddings [E,D]."""
    def __init__(self, roles, scores, embeddings, settings):
        self.settings = settings
        self.roles = np.asarray(roles, dtype=str)
        self.scores = finite_array(scores, 2, "reference scores")
        self.embeddings = unit_vectors(embeddings)
        if self.roles.ndim != 1 or len(self.roles) != len(self.scores) or len(self.roles) != len(self.embeddings):
            raise ValueError("Reference roles, scores and embeddings do not align")
        if self.scores.shape[1] < 2 or settings.top_m > self.scores.shape[1]:
            raise ValueError("Reference bank needs >=2 videos and top_m <= bank size")
        self.calibration = {}
        for role in sorted(set(self.roles)):
            indices = np.flatnonzero(self.roles == role)
            if len(indices) <= settings.neighbors:
                continue  # Explicitly unavailable; analyze() returns an auditable error.
            group = self.embeddings[indices]
            distances = []
            for index, embedding in enumerate(group):
                similarities = np.delete(group @ embedding, index)  # leave-one-out
                distances.append(1 - np.sort(similarities)[-settings.neighbors:].mean())
            grounding = np.sort(self.scores[indices], axis=-1)[:, -settings.top_m:].mean(axis=-1)
            self.calibration[role] = (indices, grounding, np.asarray(distances))

    def analyze(self, evidence, scores, embeddings):
        scores = finite_array(scores, 2, "evidence scores")
        embeddings = unit_vectors(embeddings)
        if scores.shape != (len(evidence), self.scores.shape[1]) or len(embeddings) != len(evidence):
            raise ValueError("Query evidence is not aligned with the reference bank")
        if embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Evidence embedding dimension differs from train bank")
        result = []
        cfg = self.settings
        for row, score, embedding in zip(evidence, scores, embeddings):
            role = row["role"]
            if role not in self.calibration:
                raise InsufficientRoleBank(f"No leave-one-out calibration for role {role!r}")
            indices, reference_g, reference_o = self.calibration[role]
            g = float(ecdf(reference_g, np.sort(score)[-cfg.top_m:].mean()))
            d = float(concentration(score, cfg.evidence_temperature))
            nearest = np.sort(self.embeddings[indices] @ embedding)[-cfg.neighbors:]
            o = float(ecdf(reference_o, 1 - nearest.mean()))
            risk = float(np.dot(cfg.risk_weights, [1 - g, 1 - d, o]))
            protected = (bool(row.get("protected", False)) or row["role"] in {"actor", "action", "object", "negation", "quantity", "temporal"}
                         or g >= cfg.grounding_threshold or d >= cfg.discrimination_threshold)
            action = ("SIMPLIFY" if risk >= cfg.risk_threshold else
                      "EMPHASIZE" if g >= cfg.grounding_threshold and d >= cfg.discrimination_threshold else "KEEP")
            result.append(dict(row, G=g, D=d, OOD=o, risk=risk, protected=protected,
                               action=action, down_weight=g < cfg.grounding_threshold and d < cfg.discrimination_threshold))
        return result


def utility(scores, semantic, coverage, preservation, drift, settings):
    components = np.asarray([semantic, coverage, preservation, drift], dtype=float)
    if not np.isfinite(components).all() or ((components < 0) | (components > 1)).any():
        raise ValueError("Candidate semantic components must be in [0,1]")
    quality = float(concentration(scores, settings.query_temperature))
    value = float(np.dot(settings.selection_weights, [quality, semantic, coverage, preservation, -drift]))
    return value, quality


def candidate_utility(scores, semantic, preserved, drift, hard_violation, evidence, settings):
    preserved = np.asarray(preserved, dtype=float)
    if preserved.shape != (len(evidence),) or not len(evidence):
        raise ValueError("Candidate evidence correspondence does not align")
    if not np.isfinite(preserved).all() or ((preserved < 0) | (preserved > 1)).any():
        raise ValueError("Invalid evidence preservation values")
    accepted = not hard_violation and all(
        not row["protected"] or score >= settings.preservation_threshold
        for row, score in zip(evidence, preserved))
    weights = np.finfo(np.float64).eps + np.asarray([row["G"] + row["D"] for row in evidence])
    coverage = float(preserved.mean())
    preservation = float(np.dot(weights, preserved) / weights.sum())
    value, quality = utility(scores, semantic, coverage, preservation, drift, settings)
    return dict(accepted=accepted, utility=value, quality=quality, semantic=float(semantic),
                coverage=coverage, preservation=preservation, drift=float(drift))


def final_scores(original, candidates, utilities, original_utility, settings, *, enabled=True, gate=True, mode="fusion"):
    """Returns original unchanged on OFF/fallback; candidates [K,N], utilities [K]."""
    original = np.asarray(original)
    if original.ndim != 1 or len(original) < 2 or not np.isfinite(original).all():
        raise ValueError("Expected finite original scores for >=2 videos")
    if mode not in {"selection", "fusion"}:
        raise ValueError("Unknown combination mode")
    if not enabled or not gate or len(candidates) == 0:
        return original.copy(), []
    candidates = finite_array(candidates, 2, "candidate scores")
    utilities = finite_array(utilities, 1, "candidate utilities")
    if candidates.shape != (len(utilities), len(original)) or not np.isfinite(original_utility):
        raise ValueError("Candidate score/utility alignment differs")
    # Strict improvement: ties always retain the original, including zero margin.
    chosen = np.flatnonzero(utilities > original_utility + settings.improvement_margin)
    if not len(chosen):
        return original.copy(), []
    if mode == "selection":
        best = int(chosen[np.argmax(utilities[chosen])])
        return candidates[best].astype(original.dtype), [best]
    weights = softmax(utilities[chosen], settings.fusion_temperature)
    fused = (1 - settings.rho) * original + settings.rho * (weights @ candidates[chosen])
    return fused.astype(original.dtype), chosen.tolist()
