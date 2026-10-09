"""Apply caption-specific review decisions without modifying source data."""

import csv
import json
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, dictionary_sources, validate_dictionary_terms
from export_B_synonym_pairs import expression_observed, lemma_issue


def load_reviews(path):
    reviews = {}
    if path is None or not Path(path).exists():
        return reviews
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["aspect"], row["split"], row["caption_id"]
            assert row["decision"] in {"KEEP", "CORRECT", "MOVE", "EXCLUDE"}
            reviews.setdefault(key, {})[row["source_expression"]] = row
    return reviews


def reviewed_terms(raw, kind, split, caption_id, caption, reviews):
    terms = {surface: {"tag": tag, "original_tag": tag, "original_aspect": kind,
                       "review": None, "excluded": ""} for surface, tag in raw.items()}
    for surface, row in reviews.get((kind, split, caption_id), {}).items():
        assert row["caption"] == caption and surface in raw and raw[surface] == row["source_pos"]
        if row["decision"] == "MOVE" or (row["decision"] in {"KEEP", "CORRECT"} and
                (row["corrected_aspect"] != kind or row["corrected_pos"] != raw[surface])):
            raise ValueError(f"Review cannot change supplied dictionary aspect/POS: {caption_id}: {surface}")
        info = terms[surface]
        info["review"] = row
        if row["decision"] in {"MOVE", "EXCLUDE"}:
            info["excluded"] = "MOVED_TO_ACTION" if row["decision"] == "MOVE" else "CONTEXT_EXCLUDED"
        else:
            info["tag"] = row["corrected_pos"]
    return terms


def index_sources(extraction, normalize, tokenizer, wn, reviews):
    dictionaries, vocabularies = dictionary_sources(extraction)
    representatives, senses, witness_by_sense, paths = {}, {}, {}, []
    allowed = {"object": OBJECT_TAGS, "action": ACTION_TAGS}
    for split, filename in [("train", "tvrtrain_domain_analysis.csv"), ("validation", "tvrval_action_object_analysis.csv")]:
        path = extraction / filename
        paths.append(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for kind in allowed:
                    raw = json.loads(row[f"{kind}s_raw"])
                    validate_dictionary_terms(raw, kind, split, row["caption_id"], dictionaries[kind], vocabularies[kind])
                    for surface, info in reviewed_terms(raw, kind, split, row["caption_id"], row["caption"], reviews).items():
                        tag, review = info["tag"], info["review"]
                        if info["excluded"] or tag not in allowed[kind] or not expression_observed(surface, row["caption"], tokenizer):
                            continue
                        term = review["corrected_lemma"] if review else normalize(surface, tag)
                        if lemma_issue(kind, surface, term, wn):
                            continue
                        pos = wn.NOUN if kind == "object" else wn.VERB
                        full_names = {sense.name() for sense in wn.synsets(term.replace(" ", "_"), pos=pos)}
                        names = set(review["context_synsets"].split(" | ")) - {""} if review else full_names
                        assert names <= full_names, (row["caption_id"], surface, term, names)
                        key = kind, split, term
                        witness = {"expression": surface, "caption_id": row["caption_id"], "caption": row["caption"], "synsets": names, "tag": tag}
                        representatives.setdefault(key, witness)
                        senses.setdefault(key, set()).update(names)
                        for name in names:
                            witness_by_sense.setdefault(key + (name,), witness)
        print(f"Indexed context-reviewed {split} terms", flush=True)
    return representatives, senses, witness_by_sense, paths
