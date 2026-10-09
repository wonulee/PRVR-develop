# TVR linguistic synonym candidates

`tvrtrain_synonym_candidates.csv` retains every original OOD caption and maps its original words to synonym expressions. `tvrtrain_synonym_pairs.csv` has one row per caption/source expression/synonym. `ood_word_synonyms.csv` aggregates surface-word alternatives across captions; consult the per-caption result for word sense.

Only different lexical expressions sharing a WordNet sense are candidates. Both the original word's focused context and an actual Val occurrence must select that sense, with cosine at least 0.20. This is an uncalibrated heuristic, not a guarantee of equivalence. Morphology-only and common spelling variants, broader terms, embedding-only neighbors and direct antonyms are excluded. No nearest-neighbor fallback fills missing synonyms.

Verb/preposition candidates exclude verbs with direct/indirect objects, so touching a person on the arm does not become the idiom `touch on`. Raising hands/arms with `throw up` is excluded from the vomiting synonym.

Single-word candidates have a head in the original matching Val pool. Multiword candidates must be observed with that head in a Val caption. Output inflects that head to the original POS; thus the suggested inflection can differ from the observed Val form, recorded separately as `val_expression`. `requires_review=True` throughout; check meaning, grammar and timestamp-specific action before rewriting. Future full-caption rewrites must follow object emphasis, pronoun resolution, visible affirmative states, visible emotion expressions, action preservation and 14–20 words with fewer than 30 subword tokens.

Caption status: `ALL_TERMS_HAVE_VAL_SYNONYMS` means each OOD word has at least one candidate; `PARTIAL_VAL_SYNONYMS` means some words have candidates; `NO_VAL_SYNONYM` means none do under this method. An empty list means no candidate was found by the current corpus and heuristic, not proof that English has no synonym.

Reproduce:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/viglab2/miniconda3/envs/holmes/bin/python rewriting/build_val_synonyms.py
```
