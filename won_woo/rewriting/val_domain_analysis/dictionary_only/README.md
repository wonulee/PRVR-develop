# TVR validation-domain analysis

Train: 87,174; validation: 10,895. Val action lemmas: 879; object lemmas: 1,997.

KEEP: 74,126; REWRITE: 13,031 (14.9483%); REVIEW: 17.

`tvrtrain_rewriting_candidates.csv` contains all REWRITE rows, in original train order.
`tvrtrain_domain_analysis.csv` contains every train caption, including KEEP/REVIEW.
`tvrval_action_object_analysis.csv` records validation extraction and lemma provenance.
`val_action_pool.csv` / `val_object_pool.csv` record validation-only lemmas, caption counts, forms and example IDs.
`train_ood_lemma_frequency.csv` ranks pool-external lemmas by affected train caption count.
`summary.json` records methodology, counts, input hashes and model version.

The supplied dictionaries currently cover train IDs only. Val tokens are retained only when their exact surface AND Penn POS already occur in the same supplied aspect dictionary. No case/inflection expansion, new vocabulary or Object/Action transfer is allowed. POS tagging only filters this whitelist; it cannot add a term. Both splits use the same POS-conditioned rule lemmatizer. Pool membership is compared separately for actions and objects, after case normalization. This restricted Val vocabulary cannot measure vocabulary outside the supplied Train dictionaries.

`ood_actions` and `ood_objects` identify missing lemmas; `ood_*_surfaces` map original words to those lemmas. Coverage is the fraction of distinct extracted lemmas covered by the matching pool; empty categories have blank coverage. Array/dict cells are JSON. `caption_id` is the full ID, while `video_id` and `enc` preserve its components. `needs_manual_review=1` marks empty categories or unsupported source POS. Unsupported POS remains in raw columns but is excluded from membership; both-empty or invalid-POS rows with no observed OOD go to REVIEW.

OOD here means lexical pool mismatch, not proven semantic novelty. POS errors, typos, proper names mislabeled as nouns, ambiguous lemmas and lost phrasal-verb context can affect the list. No LLM rewriting or semantic replacement has been executed. Future rewrites must preserve the timestamp-specific action, retain visible objects/interactions, resolve pronouns without inventing identities, convert negation to visible states and emotions to visible expressions, and use 14–20 words with fewer than 30 subword tokens. If no faithful in-pool expression exists, flag it for review.

Reproduce from the workspace:

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/analyze_val_domain.py
```
