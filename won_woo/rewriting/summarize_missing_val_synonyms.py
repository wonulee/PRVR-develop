"""Export captions with no Val synonyms, with reproducible source-side failure reasons."""

import argparse
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from apply_val_synonyms import digest, write_csv
from build_val_synonyms import focused_context
from suggest_val_replacements import lexical_info, load_wordnet, read_csv


REASONS = {
    "no_wordnet_sense": "정규화 조회어·품사로 조회 가능한 WordNet 의미 없음",
    "weak_source_sense": "원문 문맥의 의미 선택 점수가 기준 미만",
    "no_distinct_synonym_in_matching_val_usage": "동일 의미·Val 문맥·다른 표현·형태 조건을 통과한 후보 없음",
}


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def main():
    import numpy as np
    import spacy
    import torch
    from sentence_transformers import SentenceTransformer

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "rewriting/val_domain_analysis")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    synonym_dir = args.input_dir / "linguistic_synonyms"
    out = args.output_dir or synonym_dir / "no_val_synonym_analysis"
    inputs = [args.input_dir / "tvrtrain_rewriting_candidates.csv",
              synonym_dir / "tvrtrain_synonym_candidates.csv", synonym_dir / "summary.json"]
    train = read_csv(inputs[0])
    candidates = {row["caption_id"]: row for row in read_csv(inputs[1])}
    prior = json.loads(inputs[2].read_text())
    threshold, seed = prior["method"]["minimum_sense_cosine"], prior["method"]["seed"]
    selected = [row for row in train if candidates[row["caption_id"]]["synonym_status"] == "NO_VAL_SYNONYM"]
    selected_ids = {row["caption_id"] for row in selected}
    if len(selected_ids) != prior["caption_status"]["NO_VAL_SYNONYM"]:
        raise ValueError("No-synonym caption count does not match the source summary")

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    spacy.util.fix_random_seed(seed)
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    if spacy.__version__ != prior["method"]["spacy_version"] or nlp.meta["version"] != prior["method"]["spacy_model_version"]:
        raise ValueError("spaCy versions differ from the original synonym analysis")
    wn = load_wordnet(workspace / "rewriting/nlp_resources")
    if wn.get_version() != prior["method"]["wordnet_version"]:
        raise ValueError("WordNet version differs from the original synonym analysis")

    @lru_cache(None)
    def senses(word, kind):
        return lexical_info(wn, word, kind)["senses"]

    queries = []
    print(f"Reconstructing missing-expression contexts from {len(train):,} captions", flush=True)
    for index, (row, doc) in enumerate(zip(train, nlp.pipe(
            (row["caption"] for row in train), batch_size=128, n_process=1)), 1):
        candidate = candidates[row["caption_id"]]
        for kind in ["action", "object"]:
            missing = set(json.loads(candidate[f"{kind}s_without_synonym"]))
            raw_tags = json.loads(row[f"{kind}s_raw"])
            for surface, word in json.loads(row[f"ood_{kind}_surfaces"]).items():
                if surface not in missing:
                    continue
                tokens = [token for token in doc if token.text == surface]
                flags = []
                if not re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", row["caption"]):
                    flags.append("source_not_standalone_token")
                if len(surface) == 1 and surface.isalpha():
                    flags.append("single_letter_expression")
                if surface.isalpha() and surface.isupper() and len(surface) > 1:
                    flags.append("uppercase_expression")
                if tokens and all(token.tag_ != raw_tags[surface] for token in tokens):
                    flags.append("source_pos_differs_from_spacy")
                queries.append({**{key: row[key] for key in ["caption_id", "video_id", "enc", "caption"]},
                                "term_type": kind, "source_expression": surface, "word": word,
                                "source_pos": raw_tags[surface],
                                "source_context": focused_context(tokens[0]) if tokens else surface,
                                "review_flags": flags})
        if index % 2500 == 0:
            print(f"Contexts: {index:,}/{len(train):,}", flush=True)
    definitions = {name: sense.definition() for query in queries
                   for name, sense in senses(query["word"], query["term_type"]).items()}
    texts = sorted(set(definitions.values()) | {query["source_context"] for query in queries})
    model_path = Path("/home/viglab2/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
                      "snapshots") / prior["method"]["embedding_model_revision"]
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    vectors = {}
    for start in range(0, len(texts), 1000):
        chunk = texts[start:start + 1000]
        vectors.update(zip(chunk, model.encode(chunk, batch_size=128, normalize_embeddings=True,
                                               show_progress_bar=False)))
        print(f"Context/meaning embeddings: {min(start + 1000, len(texts)):,}/{len(texts):,}", flush=True)

    terms, all_missing_reasons = [], Counter()
    for query in queries:
        choices = senses(query["word"], query["term_type"])
        name, score = "", -1.0
        if choices:
            name = min(choices, key=lambda key: (-float(
                vectors[query["source_context"]] @ vectors[definitions[key]]), key))
            score = float(vectors[query["source_context"]] @ vectors[definitions[name]])
        reason = "no_wordnet_sense" if not name else (
            "weak_source_sense" if score < threshold else "no_distinct_synonym_in_matching_val_usage")
        all_missing_reasons[reason] += 1
        if query["caption_id"] in selected_ids:
            surface_sense_count = len(senses(query["source_expression"].casefold(), query["term_type"]))
            if not choices and surface_sense_count:
                query["review_flags"].append("raw_surface_has_sense_but_lookup_does_not")
            terms.append({**{key: value for key, value in query.items() if key != "word"},
                          "lookup_expression": query["word"], "wordnet_sense_count": len(choices),
                          "raw_surface_wordnet_sense_count": surface_sense_count, "selected_sense": name,
                          "selected_meaning": definitions[name] if name else "",
                          "source_sense_score": round(score, 6) if name else "",
                          "failure_reason_code": reason, "failure_reason_ko": REASONS[reason]})
    if dict(all_missing_reasons) != prior["missing_reasons"]:
        raise ValueError(f"Failure reason reconstruction differs: {dict(all_missing_reasons)} vs {prior['missing_reasons']}")

    by_caption, by_word = defaultdict(list), defaultdict(list)
    for term in terms:
        by_caption[term["caption_id"]].append(term)
        by_word[(term["term_type"], term["source_expression"])].append(term)
    caption_rows = []
    for row in selected:
        details = by_caption[row["caption_id"]]
        candidate = candidates[row["caption_id"]]
        for kind in ["action", "object"]:
            actual = {term["source_expression"] for term in details if term["term_type"] == kind}
            if actual != set(json.loads(candidate[f"ood_{kind}_words"])):
                raise ValueError(f"Missing term coverage mismatch: {row['caption_id']}, {kind}")
        caption_rows.append({**{key: row[key] for key in ["caption_id", "video_id", "enc", "caption", "rewrite_reason"]},
                             **{f"ood_{kind}_words": candidate[f"ood_{kind}_words"] for kind in ["action", "object"]},
                             "failure_reason_codes": encoded(sorted({term["failure_reason_code"] for term in details})),
                             "review_flags": encoded(sorted({flag for term in details for flag in term["review_flags"]})),
                             "missing_expression_details": encoded([{key: term[key] for key in [
                                 "term_type", "source_expression", "lookup_expression", "source_pos", "source_context",
                                 "wordnet_sense_count", "raw_surface_wordnet_sense_count", "selected_sense",
                                 "selected_meaning", "source_sense_score", "failure_reason_code", "failure_reason_ko"]}
                                 for term in details]), "synonym_status": "NO_VAL_SYNONYM"})
    word_rows = []
    for (kind, surface), entries in sorted(by_word.items(), key=lambda item: (-len(item[1]), item[0])):
        examples = entries[:3]
        word_rows.append({"term_type": kind, "source_expression": surface,
                          "caption_count": len({term["caption_id"] for term in entries}),
                          "failure_reason_counts": encoded(dict(Counter(term["failure_reason_code"] for term in entries))),
                          "review_caption_count": sum(bool(term["review_flags"]) for term in entries),
                          "example_caption_ids": encoded([term["caption_id"] for term in examples]),
                          "example_captions": encoded([term["caption"] for term in examples])})
    reason_rows = []
    for code, label in REASONS.items():
        entries = [term for term in terms if term["failure_reason_code"] == code]
        reason_rows.append({"failure_reason_code": code, "failure_reason_ko": label,
                            "expression_occurrence_count": len(entries),
                            "caption_count": len({term["caption_id"] for term in entries}),
                            "unique_expression_count": len({(term["term_type"], term["source_expression"]) for term in entries}),
                            "action_occurrence_count": sum(term["term_type"] == "action" for term in entries),
                            "object_occurrence_count": sum(term["term_type"] == "object" for term in entries)})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "tvrtrain_no_val_synonym_captions.csv", list(caption_rows[0]), caption_rows)
    write_csv(out / "tvrtrain_no_val_synonym_terms.csv", [
        "caption_id", "video_id", "enc", "caption", "term_type", "source_expression", "lookup_expression", "source_pos",
        "source_context", "wordnet_sense_count", "raw_surface_wordnet_sense_count", "selected_sense", "selected_meaning", "source_sense_score",
        "failure_reason_code", "failure_reason_ko", "review_flags"],
        [{**term, "review_flags": encoded(term["review_flags"])} for term in terms])
    write_csv(out / "no_val_synonym_word_frequency.csv", list(word_rows[0]), word_rows)
    write_csv(out / "failure_reason_summary.csv", list(reason_rows[0]), reason_rows)
    summary = {"caption_count": len(caption_rows), "expression_occurrence_count": len(terms),
               "unique_expression_count": len(word_rows),
               "rewrite_reason_counts": dict(Counter(row["rewrite_reason"] for row in selected)),
               "term_type_counts": dict(Counter(term["term_type"] for term in terms)),
               "failure_reasons": reason_rows,
               "review_flags": {flag: {"expression_occurrence_count": sum(flag in term["review_flags"] for term in terms),
                                       "caption_count": len({term["caption_id"] for term in terms if flag in term["review_flags"]})}
                                for flag in sorted({flag for term in terms for flag in term["review_flags"]})},
               "source_side_reconstruction_verified_against_original_summary": True,
               "all_missing_expression_reasons_including_partial_captions": dict(all_missing_reasons),
               "minimum_sense_score": threshold, "seed": seed,
               "lookup_gap_examples": [{"term_type": kind, "source_expression": surface,
                                         "lookup_expression": lookup, "expression_occurrence_count": count}
                                        for (kind, surface, lookup), count in Counter(
                                            (term["term_type"], term["source_expression"], term["lookup_expression"])
                                            for term in terms if "raw_surface_has_sense_but_lookup_does_not" in term["review_flags"]
                                        ).most_common(15)],
               "scope": "NO_VAL_SYNONYM captions only; partial-synonym captions excluded",
               "reason_method": "Reconstruct source sense/score; residual no-candidate reason uses the existing empty synonym result.",
               "limitations": "Residual reason combines Val occurrence, same-word/spelling/antonym guards and inflection constraints; not proof that no English synonym exists.",
               "inputs": {path.name: {"path": str(path.resolve()), "sha256": digest(path.read_bytes())} for path in inputs}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
