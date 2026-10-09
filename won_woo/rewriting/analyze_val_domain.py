"""Compare supplied TVR train terms with a validation-only lemma vocabulary."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ACTION_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
OBJECT_TAGS = {"NN", "NNS"}


def dictionary_vocabulary(source):
    """Index exact supplied surfaces/POS, retaining a dictionary ID as evidence."""
    vocabulary = {}
    for caption_id, terms in source.items():
        for surface, tag in terms.items():
            vocabulary.setdefault(surface, {}).setdefault(tag, caption_id)
    return vocabulary


def extract_dictionary_terms(doc, source, vocabulary, caption_id, allowed_tags):
    if caption_id in source:
        return dict(source[caption_id])
    # POS can only select an existing surface/POS pair; it cannot add vocabulary.
    return {token.text: token.tag_ for token in doc
            if token.tag_ in allowed_tags and token.tag_ in vocabulary.get(token.text, {})}


def validate_dictionary_terms(raw, kind, split, caption_id, source, vocabulary):
    if split == "train" or caption_id in source:
        if raw != source.get(caption_id):
            raise ValueError(f"{kind} terms differ from supplied ID annotation: {caption_id}")
    for surface, tag in raw.items():
        if tag not in vocabulary.get(surface, {}):
            raise ValueError(f"Term outside supplied {kind} dictionary: {caption_id}: {surface!r}/{tag}")


def dictionary_sources(extraction):
    summary = json.loads((extraction / "summary.json").read_text())
    if not summary["method"].get("dictionary_only"):
        raise ValueError("Rebuild extraction with dictionary-only vocabulary before exporting")
    sources = {}
    for kind in ("object", "action"):
        info = summary["inputs"][f"tvr_id_{kind}_dict.json"]
        data = Path(info["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise ValueError(f"Supplied {kind} dictionary changed; rebuild extraction")
        sources[kind] = json.loads(data)
    return sources, {kind: dictionary_vocabulary(source) for kind, source in sources.items()}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_captions(path):
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            caption_id, caption = line.strip().split(maxsplit=1)
            video_id, enc = caption_id.rsplit("#enc#", 1)
            if not caption.strip() or not enc.isdigit() or caption_id in seen:
                raise ValueError("empty caption, invalid enc, or duplicate ID")
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        seen.add(caption_id)
        rows.append({"caption_id": caption_id, "video_id": video_id,
                     "enc": int(enc), "caption": caption})
    if not rows:
        raise ValueError(f"No captions in {path}")
    return rows


def write_csv(path, fields, rows):
    # UTF-8 BOM keeps English captions and future Korean notes readable in Excel.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_normalizer(nlp, wordnet=None, inflector=None):
    from spacy.symbols import NOUN, VERB
    from spacy.tokens import Doc

    lemmatizer = nlp.get_pipe("lemmatizer")

    @lru_cache(maxsize=None)
    def normalize(surface, tag):
        word = surface.strip().casefold()
        if tag not in ACTION_TAGS | OBJECT_TAGS:
            raise ValueError(f"Unsupported POS: {surface!r}: {tag}")
        if wordnet is not None:
            pos = wordnet.VERB if tag in ACTION_TAGS else wordnet.NOUN
            roots = wordnet._morphy(word.replace(" ", "_"), pos)
            if len(roots) == 1:
                return roots[0].replace("_", " ")
            if inflector is not None and len(roots) > 1:
                matching = [root for root in roots if word in
                            {form.casefold() for form in inflector(root.replace("_", " "), tag)}]
                if len(matching) == 1:
                    return matching[0].replace("_", " ")
        doc = Doc(nlp.vocab, words=[word])
        doc[0].pos = VERB if tag in ACTION_TAGS else NOUN
        doc[0].tag_ = tag
        if tag == "NN":
            doc[0].set_morph("Number=Sing")
        lemmatizer(doc)
        lemma = doc[0].lemma_.casefold()
        return lemma.replace("_", " ") if wordnet is not None else lemma

    return normalize


def normalize_terms(terms, allowed_tags, normalize):
    lemmas = {}
    invalid = {}
    for surface, tag in terms.items():
        if tag not in allowed_tags:
            invalid[surface] = tag
        else:
            lemmas[surface] = normalize(surface, tag)
    return lemmas, invalid


def classify(action_lemmas, object_lemmas, action_pool, object_pool, invalid=False):
    ood_actions = set(action_lemmas) - action_pool
    ood_objects = set(object_lemmas) - object_pool
    if ood_actions or ood_objects:
        reason = "action_and_object_ood" if ood_actions and ood_objects else (
            "action_ood" if ood_actions else "object_ood")
        return "out_of_domain", "REWRITE", reason
    if invalid or not (action_lemmas or object_lemmas):
        return "undetermined", "REVIEW", "invalid_pos_or_no_extracted_terms"
    return "in_domain", "KEEP", "all_observed_lemmas_in_val_pool"


def self_check(normalize):
    # Morphological variants must not become artificial OOD candidates.
    assert normalize("walks", "VBZ") == normalize("walking", "VBG") == "walk"
    assert normalize("sat", "VBD") == "sit"
    assert normalize("held", "VBD") == "hold"
    assert normalize("hands", "NNS") == normalize("hand", "NN") == "hand"
    assert normalize("House", "NN") == "house"
    ap, op = {"walk"}, {"hand"}
    assert classify({"walk"}, {"hand"}, ap, op)[1] == "KEEP"
    assert classify({"jump"}, {"hand"}, ap, op)[2] == "action_ood"
    assert classify({"walk"}, {"helmet"}, ap, op)[2] == "object_ood"
    assert classify({"jump"}, {"helmet"}, ap, op)[2] == "action_and_object_ood"
    assert classify(set(), set(), ap, op)[1] == "REVIEW"
    assert classify({"walk"}, set(), ap, op, invalid=True)[1] == "REVIEW"


def main():
    import spacy

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=workspace / "tvr/TextData")
    parser.add_argument("--train-captions", type=Path,
                        help="Original caption snapshot associated with the supplied ID dictionaries")
    parser.add_argument("--output-dir", type=Path,
                        default=workspace / "rewriting/val_domain_analysis/dictionary_only")
    parser.add_argument("--model", default="en_core_web_md")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    spacy.util.fix_random_seed(args.seed)
    nlp = spacy.load(args.model, disable=["parser", "ner"])
    normalize = make_normalizer(nlp)
    self_check(normalize)

    inputs = [args.data_dir / name for name in (
        "tvrtrain.caption.txt", "tvrval.caption.txt",
        "tvr_id_action_dict.json", "tvr_id_object_dict.json")]
    if args.train_captions:
        inputs[0] = args.train_captions
    train, val = map(read_captions, inputs[:2])
    actions, objects = [json.loads(path.read_text(encoding="utf-8")) for path in inputs[2:]]
    vocabularies = {"action": dictionary_vocabulary(actions), "object": dictionary_vocabulary(objects)}
    train_ids = {row["caption_id"] for row in train}
    val_ids = {row["caption_id"] for row in val}
    if train_ids & val_ids:
        raise ValueError("Train and validation caption IDs overlap")
    for name, source in [("action", actions), ("object", objects)]:
        if not isinstance(source, dict) or not train_ids.issubset(source):
            raise ValueError(f"{name} dictionary must cover every train caption ID")
        for caption_id, terms in source.items():
            if not isinstance(terms, dict) or any(
                    not isinstance(word, str) or not word.strip() or not isinstance(tag, str)
                    for word, tag in terms.items()):
                raise ValueError(f"Invalid {name} dictionary entry: {caption_id}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pools = {kind: defaultdict(lambda: {"caption_count": 0, "surfaces": set(), "examples": []})
             for kind in ["action", "object"]}
    val_rows = []
    print(f"Extracting validation terms from {len(val):,} captions ({args.model})", flush=True)
    for index, (row, doc) in enumerate(zip(val, nlp.pipe(
            (r["caption"] for r in val), batch_size=128, n_process=1)), 1):
        result = dict(row)
        for kind, tags, source in [("action", ACTION_TAGS, actions),
                                   ("object", OBJECT_TAGS, objects)]:
            # Supplied validation entries are preferred if added in a future dataset.
            raw = extract_dictionary_terms(doc, source, vocabularies[kind], row["caption_id"], tags)
            origin = "supplied_dictionary" if row["caption_id"] in source else "dictionary_vocabulary_exact_surface_and_pos"
            validate_dictionary_terms(raw, kind, "validation", row["caption_id"], source, vocabularies[kind])
            mapped, invalid = normalize_terms(raw, tags, normalize)
            result[f"{kind}_source"] = origin
            result[f"{kind}s_raw"] = encoded(raw)
            result[f"{kind}_lemmas"] = encoded(sorted(set(mapped.values())))
            result[f"invalid_{kind}_pos"] = encoded(invalid)
            for lemma in sorted(set(mapped.values())):
                entry = pools[kind][lemma]
                entry["caption_count"] += 1
                entry["surfaces"].update(word for word, value in mapped.items() if value == lemma)
                if len(entry["examples"]) < 3:
                    entry["examples"].append(row["caption_id"])
        val_rows.append(result)
        if index % 2500 == 0:
            print(f"Validation: {index:,}/{len(val):,}", flush=True)
    if len(val_rows) != len(val):
        raise RuntimeError("Validation extraction did not cover all captions")
    write_csv(out / "tvrval_action_object_analysis.csv", list(val_rows[0]), val_rows)
    for kind, pool in pools.items():
        write_csv(out / f"val_{kind}_pool.csv",
                  ["lemma", "val_caption_count", "surface_forms", "example_caption_ids"],
                  ({"lemma": lemma, "val_caption_count": entry["caption_count"],
                    "surface_forms": encoded(sorted(entry["surfaces"])),
                    "example_caption_ids": encoded(entry["examples"])}
                   for lemma, entry in sorted(pool.items())))

    action_pool, object_pool = set(pools["action"]), set(pools["object"])
    counts = Counter()
    reasons = Counter()
    flags_count = Counter()
    ood_frequency = defaultdict(lambda: {"caption_count": 0, "surfaces": set(), "examples": []})
    fields = ["caption_id", "video_id", "enc", "caption", "domain", "decision", "rewrite_reason",
              "actions_raw", "objects_raw", "action_lemmas", "object_lemmas",
              "in_domain_actions", "in_domain_objects", "ood_actions", "ood_objects",
              "ood_action_surfaces", "ood_object_surfaces", "action_coverage", "object_coverage",
              "ood_action_count", "ood_object_count", "ood_total_count",
              "extraction_flags", "invalid_action_pos", "invalid_object_pos", "needs_manual_review"]
    print(f"Comparing {len(train):,} train captions with validation-only pools", flush=True)
    with (out / "tvrtrain_domain_analysis.csv").open("w", encoding="utf-8-sig", newline="") as all_file, \
            (out / "tvrtrain_rewriting_candidates.csv").open("w", encoding="utf-8-sig", newline="") as candidate_file:
        all_writer = csv.DictWriter(all_file, fieldnames=fields)
        candidate_writer = csv.DictWriter(candidate_file, fieldnames=fields)
        all_writer.writeheader()
        candidate_writer.writeheader()
        for row in train:
            caption_id = row["caption_id"]
            result = dict(row)
            flags = []
            term_sets = {}
            invalid_sets = {}
            for kind, source, tags, pool in [("action", actions, ACTION_TAGS, action_pool),
                                            ("object", objects, OBJECT_TAGS, object_pool)]:
                raw = source[caption_id]
                mapped, invalid = normalize_terms(raw, tags, normalize)
                lemmas = set(mapped.values())
                inside, outside = lemmas & pool, lemmas - pool
                term_sets[kind] = lemmas
                invalid_sets[kind] = invalid
                if not lemmas:
                    flags.append(f"empty_{kind}")
                if invalid:
                    flags.append(f"invalid_{kind}_pos")
                result[f"{kind}s_raw"] = encoded(raw)
                result[f"{kind}_lemmas"] = encoded(sorted(lemmas))
                result[f"in_domain_{kind}s"] = encoded(sorted(inside))
                result[f"ood_{kind}s"] = encoded(sorted(outside))
                result[f"ood_{kind}_surfaces"] = encoded(
                    {word: lemma for word, lemma in mapped.items() if lemma in outside})
                result[f"{kind}_coverage"] = round(len(inside) / len(lemmas), 6) if lemmas else ""
                result[f"ood_{kind}_count"] = len(outside)
                result[f"invalid_{kind}_pos"] = encoded(invalid)
                for lemma in sorted(outside):
                    entry = ood_frequency[(kind, lemma)]
                    entry["caption_count"] += 1
                    entry["surfaces"].update(word for word, value in mapped.items() if value == lemma)
                    if len(entry["examples"]) < 3:
                        entry["examples"].append(caption_id)
            domain, decision, reason = classify(term_sets["action"], term_sets["object"],
                                                action_pool, object_pool, any(invalid_sets.values()))
            result.update(domain=domain, decision=decision, rewrite_reason=reason,
                          ood_total_count=result["ood_action_count"] + result["ood_object_count"],
                          extraction_flags=encoded(flags), needs_manual_review=int(bool(flags)))
            all_writer.writerow(result)
            counts[decision] += 1
            flags_count.update(flags)
            if decision == "REWRITE":
                candidate_writer.writerow(result)
                reasons[reason] += 1
    write_csv(out / "train_ood_lemma_frequency.csv",
              ["term_type", "lemma", "train_caption_count", "surface_forms", "example_caption_ids"],
              ({"term_type": kind, "lemma": lemma, "train_caption_count": entry["caption_count"],
                "surface_forms": encoded(sorted(entry["surfaces"])),
                "example_caption_ids": encoded(entry["examples"])}
               for (kind, lemma), entry in sorted(ood_frequency.items(),
                                                  key=lambda item: (-item[1]["caption_count"], item[0]))))
    assert sum(counts.values()) == len(train)
    assert sum(reasons.values()) == counts["REWRITE"]
    summary = {
        "train_captions": len(train), "validation_captions": len(val),
        "val_action_pool_size": len(action_pool), "val_object_pool_size": len(object_pool),
        "decisions": {key: counts[key] for key in ["KEEP", "REWRITE", "REVIEW"]},
        "rewrite_reasons": dict(sorted(reasons.items())),
        "rewriting_percentage": round(100 * counts["REWRITE"] / len(train), 4),
        "unique_ood_actions": sum(kind == "action" for kind, _ in ood_frequency),
        "unique_ood_objects": sum(kind == "object" for kind, _ in ood_frequency),
        "extraction_flags": dict(sorted(flags_count.items())),
        "validation_dictionary_coverage": {
            "action": len(val_ids & set(actions)), "object": len(val_ids & set(objects))},
        "method": {
            "dictionary_only": True,
            "llm_used": False,
            "train_terms": "Supplied dictionaries; original term/POS mappings preserved",
            "validation_terms": "Supplied entries when available; otherwise exact surface AND Penn POS pairs already in the same supplied aspect dictionary; no new words, forms or aspect transfers",
            "action_tags": sorted(ACTION_TAGS), "object_tags": sorted(OBJECT_TAGS),
            "validation_stopword_filter": False,
            "normalization": "Unicode casefold + spaCy rule lemmatizer, forced VERB/NOUN from supplied POS; NN uses Number=Sing",
            "reference_domain": "Validation lemmas only, separately for actions and objects",
            "candidate_rule": "Any normalized action OR object outside its validation pool",
            "empty_rule": "Empty categories are flagged, never counted as OOD; both empty => REVIEW",
            "invalid_pos_rule": "Preserve and flag unsupported POS; exclude from lemma membership; REVIEW unless other terms are OOD",
            "semantic_scope": "Lexical vocabulary coverage, not a semantic OOD or rewrite-feasibility guarantee",
            "limitations": [
                "Dictionaries cover Train IDs only; restricted Val vocabulary cannot measure Val words outside the supplied dictionaries.",
                "POS and rule-based lemmas can be wrong or ambiguous; raw forms remain in every row.",
                "Individual terms do not encode phrasal verbs, action-object relations, or temporal alignment.",
                "No automatic replacement or LLM rewrite has been performed."],
            "spacy_version": spacy.__version__, "model": args.model,
            "model_version": nlp.meta["version"], "seed": args.seed,
            "n_process": 1, "output_order": "Original caption order; pools alphabetical; frequency descending",
        },
        "inputs": {name: {"path": str(path.resolve()),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in zip(("tvrtrain.caption.txt", "tvrval.caption.txt", "tvr_id_action_dict.json", "tvr_id_object_dict.json"), inputs)},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    (out / "README.md").write_text(
        f"# TVR validation-domain analysis\n\n"
        f"Train: {len(train):,}; validation: {len(val):,}. "
        f"Val action lemmas: {len(action_pool):,}; object lemmas: {len(object_pool):,}.\n\n"
        f"KEEP: {counts['KEEP']:,}; REWRITE: {counts['REWRITE']:,} "
        f"({summary['rewriting_percentage']}%); REVIEW: {counts['REVIEW']:,}.\n\n"
        "`tvrtrain_rewriting_candidates.csv` contains all REWRITE rows, in original train order.\n"
        "`tvrtrain_domain_analysis.csv` contains every train caption, including KEEP/REVIEW.\n"
        "`tvrval_action_object_analysis.csv` records validation extraction and lemma provenance.\n"
        "`val_action_pool.csv` / `val_object_pool.csv` record validation-only lemmas, caption counts, forms and example IDs.\n"
        "`train_ood_lemma_frequency.csv` ranks pool-external lemmas by affected train caption count.\n"
        "`summary.json` records methodology, counts, input hashes and model version.\n\n"
        "The supplied dictionaries currently cover train IDs only. Val tokens are retained only when their exact surface "
        "AND Penn POS already occur in the same supplied aspect dictionary. No case/inflection expansion, new vocabulary "
        "or Object/Action transfer is allowed. POS tagging only filters this whitelist; it cannot add a term. "
        "Both splits use the same POS-conditioned rule lemmatizer. Pool membership is compared separately "
        "for actions and objects, after case normalization. This restricted Val vocabulary cannot measure vocabulary "
        "outside the supplied Train dictionaries.\n\n"
        "`ood_actions` and `ood_objects` identify missing lemmas; `ood_*_surfaces` map original words to those lemmas. "
        "Coverage is the fraction of distinct extracted lemmas covered by the matching pool; empty categories have blank coverage. "
        "Array/dict cells are JSON. `caption_id` is the full ID, while `video_id` and `enc` preserve its components. "
        "`needs_manual_review=1` marks empty categories or unsupported source POS. Unsupported POS remains in raw columns "
        "but is excluded from membership; both-empty or invalid-POS rows with no observed OOD go to REVIEW.\n\n"
        "OOD here means lexical pool mismatch, not proven semantic novelty. POS errors, typos, proper names mislabeled "
        "as nouns, ambiguous lemmas and lost phrasal-verb context can affect the list. "
        "No LLM rewriting or semantic replacement has been executed. "
        "Future rewrites must preserve the timestamp-specific action, retain visible objects/interactions, resolve pronouns "
        "without inventing identities, convert negation to visible states and emotions to visible expressions, "
        "and use 14–20 words with fewer than 30 subword tokens. If no faithful in-pool expression exists, flag it for review.\n\n"
        "Reproduce from the workspace:\n\n```bash\n"
        "/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/analyze_val_domain.py\n```\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"method", "inputs"}},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"CSV artifacts saved to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
