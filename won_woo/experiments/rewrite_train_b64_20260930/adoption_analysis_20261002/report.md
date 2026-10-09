# Train rewrite adoption analysis (2026-10-02)

## Data and verification

Read all 87,174 audit records; 11,064 accepted (12.69%). Rechecked every accepted query with the local RoBERTa tokenizer, candidate_check, verdict_check and per-video accepted-candidate duplicate guard. All 11,064 passed. This replays stored verdicts; it does not re-run the LLM or provide independent semantic verification.

Audit SHA256: fd42e0f67dfdd1f6f5ed690616e4b0b42c85ebc84216c2640518b1a2761c7b91
Final run source SHA256: bf617d935402746c8173977f63c28f52e01ef031aad87ea5362950f8ba0e4a50
Current source SHA256: bf617d935402746c8173977f63c28f52e01ef031aad87ea5362950f8ba0e4a50

## Actual adoption conditions

- Changed, nonempty, single-line candidate.
- At most 22 whitespace-separated words; word_min=16 is soft (strict_word_budget=false).
- At most 29 RoBERTa tokens including special tokens.
- Identical detected temporal marker sequences in original and candidate; this is a lexical guard, not a temporal semantics guarantee.
- Grounding list type valid; every supplied item cites a same-video source caption containing quote and text. Empty lists are allowed.
- Stored verifier has all event/roles/negation/time Booleans true, sibling_event_imported=false, unsupported_visual_details=[], valid schema and string reason.
- No normalized duplicate among already accepted rewrites in the same video.
- No requirement that any of the four content transformations was actually applied. No retrieval-score improvement test.

## Original lexical flags

| Flag | All queries | Accepted | Acceptance rate |
|---|---:|---:|---:|
| pronoun | 49,601 | 6,311 | 12.72% |
| negation | 2,462 | 573 | 23.27% |
| mental_state | 1,952 | 334 | 17.11% |
| any_flag | 50,350 | 6,458 | 12.83% |
| no_flag | 36,824 | 4,606 | 12.51% |

Flag categories overlap. No-flag excludes only pronoun/negation/mental-state keyword flags; object grounding has no flag and these heuristics are not annotated labels.

## Accepted lengths and grounding

Word counts: 4–22, mean 10.71; original mean 12.31. Shorter 7,482, same 2,332, longer 1,250.
Below recommended minimum 16 words: 10,241. Within 14–20 words: 1,629; the run did not enforce a 14–20 hard budget.
RoBERTa token counts: 8–29, mean 15.44.

| Grounding scope (exclusive query categories) | Count |
|---|---:|
| Target-only citations | 10,679 |
| Includes sibling citation | 373 |
| Empty grounding | 12 |

Queries with at least one grounding.text not literally in the candidate: 10,032. This is permitted by candidate_check for faithful paraphrases; it means the literal-source check is not a check of each newly added candidate detail.

## Lexical flags after rewriting

| Flag | Present in both | Disappeared | Newly added |
|---|---:|---:|---:|
| pronoun | 5,045 | 1,266 | 322 |
| negation | 302 | 271 | 57 |
| mental_state | 145 | 189 | 107 |

Disappearance is not proof of faithful negation conversion, coreference resolution, or visual grounding. It may also result from deletion or another paraphrase.

## Representative audit evidence

- Line 57: leaves -> exits; generic paraphrase accepted.
- Line 40: him -> Howard; accepted named-reference rewrite.
- Line 3853: won't close -> remains open; affirmative-state rewrite accepted.
- Line 1480: visibly upset retained with order change; emotion-to-action not mandatory.
- Line 1: picks up a drink -> grabs a drink rejected because grounding.text was not literal source text.
- Line 2: beside her -> by her side rejected with negation_equivalent=false although explanation says target matched and original contains no negation.
- Line 15827: without wearing pants -> barefoot accepted despite non-equivalent meanings.
- Line 333: Montgomery holding gun -> Beckett holding gun accepted despite role change.

## Terminal reasons

| Recorded result/reason | Count |
|---|---:|---:|
| unsupported_summary_span | 30,769 |
| temporal_change | 22,900 |
| negation_equivalent | 16,435 |
| accepted | 11,064 |
| invalid_json | 2,456 |
| unsupported_visual_details | 1,057 |
| unchanged | 625 |
| temporal_relationships_preserved | 608 |
| word_budget | 520 |
| unsupported_quote | 299 |
| invalid_verification_schema | 272 |
| token_budget | 113 |
| target_collapse | 20 |
| target_event_preserved | 18 |
| participant_roles_preserved | 17 |
| sibling_event_imported | 1 |

One terminal reason is stored per query. Some invalid_json records originate from generation or verification; absence of verdict does not uniquely identify an earlier pipeline stage.

negation_equivalent rejections without original negation keyword flag: 16,387 / 16,435. This is evidence to audit the verifier, not an automatic count of erroneous rejections.

## Files

- summary.json: counts, acceptance settings, provenance and caveats.
- accepted.csv: all 11,064 accepted original/candidate pairs, verifier explanations, flags, lengths and submitted grounding.

The dataset is context-assisted/oracle: sibling captions are provided using known same-video grouping. Accepted count is not a retrieval-improvement count or a human-verified transformation count.
