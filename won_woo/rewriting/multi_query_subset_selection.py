"""Label-aware Train selector for original + zero, one or two valid rewrites.

This implements the proposed MQVR-inspired experiment, not an inference-time
selector: ``positive_index`` must identify the known positive in a fixed gallery.
Semantic/temporal validity must be checked before passing candidates here.
Scores are averaged per video *before* finding the hardest negative. No single
candidate is discarded before evaluating every pair.
"""

from itertools import combinations
import math

import numpy as np


def select_subset(
    original_scores,
    candidate_scores,
    positive_index,
    query_ids,
    video_ids,
    *,
    original_query_id=None,
    lambda_penalty=0.0,
    epsilon=1e-6,
    require_duplicate_gain=True,
):
    """Return a JSON-serializable choice and the audit of all compared subsets.

    ``original_scores`` is [videos]; ``candidate_scores`` is [candidates, videos].
    ``query_ids`` gives candidate IDs in row order and ``video_ids`` gives the
    shared gallery column order. The optional original ID is retained separately.
    Lambda defaults to no count penalty; epsilon defaults to numerical tolerance,
    not a tuned research threshold. Near-ties choose fewer additional queries.

    Pairs must improve on both constituent single-candidate subsets. By default
    they must also improve on both original+A+A and original+B+B. These duplicate
    controls test 2/3 candidate weighting; they do not establish independence.
    """
    original = np.asarray(original_scores, dtype=np.float64)
    candidates = np.asarray(candidate_scores, dtype=np.float64)
    query_ids = list(query_ids)
    video_ids = list(video_ids)
    if original.ndim != 1 or original.size < 2:
        raise ValueError("original_scores must be a vector over at least two videos")
    if candidates.size == 0 and not query_ids:
        candidates = np.empty((0, original.size), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape != (len(query_ids), original.size):
        raise ValueError("candidate_scores must have shape (len(query_ids), videos)")
    if len(video_ids) != original.size or len(set(video_ids)) != len(video_ids):
        raise ValueError("video_ids must uniquely identify every gallery column")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must uniquely identify candidate rows")
    if original_query_id is not None and original_query_id in query_ids:
        raise ValueError("the original query must not also be a candidate")
    if (
        isinstance(positive_index, (bool, np.bool_))
        or not isinstance(positive_index, (int, np.integer))
        or not 0 <= positive_index < original.size
    ):
        raise ValueError("positive_index must be a valid integer gallery index")
    positive_index = int(positive_index)
    lambda_penalty, epsilon = float(lambda_penalty), float(epsilon)
    if not math.isfinite(lambda_penalty) or lambda_penalty < 0:
        raise ValueError("lambda_penalty must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not np.isfinite(original).all() or not np.isfinite(candidates).all():
        raise ValueError("all retrieval scores must be finite")

    negative_indices = np.concatenate(
        (np.arange(positive_index), np.arange(positive_index + 1, original.size))
    )

    def retrieval_audit(scores):
        hardest = int(negative_indices[np.argmax(scores[negative_indices])])
        positive_score, negative_score = float(scores[positive_index]), float(scores[hardest])
        return {
            "margin": positive_score - negative_score,
            "positive_score": positive_score,
            "positive_video_id": video_ids[positive_index],
            "hardest_negative_score": negative_score,
            "hardest_negative_video_id": video_ids[hardest],
            "hardest_negative_index": hardest,
        }

    baseline = retrieval_audit(original)
    evaluations = []

    def subset_audit(indices, scores):
        row = retrieval_audit(scores)
        delta = row["margin"] - baseline["margin"]
        row.update(
            candidate_indices=list(indices),
            candidate_ids=[query_ids[i] for i in indices],
            candidate_count=len(indices),
            margin_gain=delta,
            objective=delta - lambda_penalty * len(indices),
            eligible=True,
            rejection_reasons=[],
        )
        return row

    empty = subset_audit((), original)
    evaluations.append(empty)
    singles = []
    for i, scores in enumerate(candidates):
        row = subset_audit((i,), (original + scores) / 2.0)
        singles.append(row)
        evaluations.append(row)

    # ponytail: exhaustive pairs are O(candidates^2 * videos); block callers by
    # source caption instead of pruning individually poor but complementary queries.
    for i, j in combinations(range(len(query_ids)), 2):
        row = subset_audit((i, j), (original + candidates[i] + candidates[j]) / 3.0)
        duplicate_i = retrieval_audit((original + 2.0 * candidates[i]) / 3.0)
        duplicate_j = retrieval_audit((original + 2.0 * candidates[j]) / 3.0)
        removal_gains = [
            row["margin"] - singles[j]["margin"],
            row["margin"] - singles[i]["margin"],
        ]
        duplicate_gains = [
            row["margin"] - duplicate_i["margin"],
            row["margin"] - duplicate_j["margin"],
        ]
        row.update(
            # Position k means removing candidate_indices[k] from the pair.
            removal_margin_gains=removal_gains,
            best_constituent_single_margin=max(singles[i]["margin"], singles[j]["margin"]),
            duplicate_controls=[
                {"repeated_candidate_id": query_ids[i], **duplicate_i},
                {"repeated_candidate_id": query_ids[j], **duplicate_j},
            ],
            duplicate_margin_gains=duplicate_gains,
            complementary_margin_gain=min(duplicate_gains),
        )
        if min(removal_gains) <= epsilon:
            row["rejection_reasons"].append("second_candidate_has_no_deletion_gain")
        if require_duplicate_gain and min(duplicate_gains) <= epsilon:
            row["rejection_reasons"].append("pair_has_no_gain_over_both_duplicate_controls")
        row["eligible"] = not row["rejection_reasons"]
        evaluations.append(row)

    # Find the global best objective first: sequential approximate comparisons can
    # otherwise drift by epsilon and depend on input order. Select the smallest
    # subset within epsilon of that optimum, then the best objective of that size.
    eligible = [row for row in evaluations if row["eligible"]]
    optimum = max(row["objective"] for row in eligible)
    near_optimal = [row for row in eligible if optimum - row["objective"] <= epsilon]
    smallest = min(row["candidate_count"] for row in near_optimal)
    chosen = max(
        (row for row in near_optimal if row["candidate_count"] == smallest),
        key=lambda row: row["objective"],
    )
    for row in evaluations:
        row["selected"] = row is chosen

    count = chosen["candidate_count"]
    reason = (
        "no_improvement_after_penalty_and_tolerance"
        if count == 0
        else "single_candidate_improves_objective"
        if count == 1
        else "pair_improves_objective_and_passes_complementarity_controls"
        if require_duplicate_gain
        else "pair_improves_objective_and_passes_deletion_controls"
    )
    return {
        "original_query_id": original_query_id,
        "original_always_included": True,
        "selected_candidate_ids": list(chosen["candidate_ids"]),
        "selected_candidate_indices": list(chosen["candidate_indices"]),
        "selected_query_ids": (
            [original_query_id] + list(chosen["candidate_ids"])
            if original_query_id is not None
            else None
        ),
        "selected_candidate_count": count,
        "selection_reason": reason,
        "baseline": baseline,
        "selected": chosen,
        "evaluations": evaluations,
        "lambda_penalty": lambda_penalty,
        "epsilon": epsilon,
        "require_duplicate_gain": bool(require_duplicate_gain),
        "gallery_size": original.size,
        "positive_index": positive_index,
        "label_aware_train_selection": True,
        "scoring": "mean_query_scores_then_max_negative",
        "evaluated_single_count": len(query_ids),
        "evaluated_pair_count": len(query_ids) * (len(query_ids) - 1) // 2,
    }
