"""Build synonym-only TVR candidates using original words and Val usage."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, OBJECT_TAGS, encoded, make_normalizer
from suggest_val_replacements import load_wordnet, lexical_info, opposed, read_csv


def spelling_key(word):
    variants = {"colour": "color", "flavour": "flavor", "neighbour": "neighbor",
                "harbour": "harbor", "favour": "favor", "honour": "honor", "labour": "labor",
                "rumour": "rumor", "centre": "center", "metre": "meter", "theatre": "theater",
                "grey": "gray", "mould": "mold", "plough": "plow", "tyre": "tire", "kerb": "curb"}
    return " ".join(re.sub(r"ise$", "ize", variants.get(w, w)) for w in word.casefold().split())


def same_word(source, target):
    if source["base"] and source["base"] == target["base"]:
        return True
    return spelling_key(source["word"]) == spelling_key(target["word"])


def focused_context(token):
    root = token
    if token.pos_ in {"NOUN", "PROPN"} and token.dep_ in {"compound", "amod"}:
        root = token.head
    selected = []
    for child in root.subtree:
        head, path = child, []
        while head.i != root.i and len(path) < len(token.doc):
            path.append(head.dep_)
            head = head.head
        if head.i == root.i and not any(dep in {"conj", "advcl", "relcl", "punct"} for dep in path):
            selected.append(child)
    return " ".join(t.text for t in sorted(selected, key=lambda t: t.i)) or token.text


def raises_limbs(token):
    # ponytail: known ambiguous phrase guard; annotated word senses are needed for exhaustive disambiguation.
    arguments = [c for c in token.children if c.dep_ in {"dobj", "obj", "nsubjpass"}]
    if token.dep_ == "acl":
        arguments.append(token.head)
    return any(t.text.casefold() in {"hand", "hands", "arm", "arms"}
               for argument in arguments for t in argument.subtree)


def main():
    import numpy as np
    import spacy
    import torch
    from sentence_transformers import SentenceTransformer

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "rewriting/val_domain_analysis")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-sense-score", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.top_k < 1 or not -1 <= args.min_sense_score <= 1:
        parser.error("top-k must be positive and min-sense-score must be in [-1, 1]")
    sys.path.insert(0, str(workspace / "rewriting/nlp_resources/python"))
    from lemminflect import getInflection
    wn = load_wordnet(workspace / "rewriting/nlp_resources")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    spacy.util.fix_random_seed(args.seed)
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    normalize = make_normalizer(nlp)

    @lru_cache(None)
    def lex(word, kind):
        return dict(lexical_info(wn, word, kind), word=word)

    assert same_word(lex("thumbs", "object"), lex("thumb", "object"))
    assert same_word(lex("realise", "action"), lex("realize", "action"))
    assert not same_word(lex("handcuff", "action"), lex("cuff", "action"))
    assert "vomit.v.01" in lex("throw up", "action")["senses"]
    assert getInflection("cuff", "VBN")[0] == "cuffed"
    for sentence, expected in [("Ross throws up his hands.", True),
                               ("Ross throws up in the bathroom.", False)]:
        test_doc = nlp(sentence)
        assert raises_limbs(next(t for t in test_doc if t.text == "throws")) == expected

    paths = [args.input_dir / name for name in ["tvrtrain_rewriting_candidates.csv",
             "tvrval_action_object_analysis.csv", "val_action_pool.csv", "val_object_pool.csv"]]
    train, validation = map(read_csv, paths[:2])
    pools = {kind: {row["lemma"] for row in read_csv(path)}
             for kind, path in zip(["action", "object"], paths[2:])}
    source_words = {kind: {word for row in train for word in json.loads(row[f"ood_{kind}_surfaces"]).values()}
                    for kind in pools}
    source_senses = {kind: {name for word in words for name in lex(word, kind)["senses"]}
                     for kind, words in source_words.items()}
    source_query_map = defaultdict(list)
    queries = []
    print(f"Extracting focused source contexts: {len(train):,} captions", flush=True)
    for index, (row, doc) in enumerate(zip(train, nlp.pipe(
            (r["caption"] for r in train), batch_size=128, n_process=1)), 1):
        for kind in pools:
            raw_tags = json.loads(row[f"{kind}s_raw"])
            for surface, word in json.loads(row[f"ood_{kind}_surfaces"]).items():
                tokens = [t for t in doc if t.text == surface]
                context = focused_context(tokens[0]) if tokens else surface
                query = {"caption_id": row["caption_id"], "term_type": kind,
                         "source_expression": surface, "word": word,
                         "source_pos": raw_tags[surface], "source_context": context}
                source_query_map[row["caption_id"]].append(len(queries))
                queries.append(query)
        if index % 2500 == 0:
            print(f"Source contexts: {index:,}/{len(train):,}", flush=True)

    targets = {}
    occurrences = []

    def add_target(kind, expression, head_word, head_index, observed, tag, row, context):
        info = lex(expression, kind)
        if not (set(info["senses"]) & source_senses[kind]):
            return
        key = kind, expression
        target = targets.setdefault(key, {"word": expression, "head": head_word,
                                          "head_index": head_index, "occurrences": []})
        target["occurrences"].append(len(occurrences))
        occurrences.append({"target_key": key, "val_expression": observed, "pos": tag,
                            "val_caption_id": row["caption_id"], "val_caption": row["caption"],
                            "context": context})

    print(f"Extracting validation expressions: {len(validation):,} captions", flush=True)
    for index, (row, doc) in enumerate(zip(validation, nlp.pipe(
            (r["caption"] for r in validation), batch_size=128, n_process=1)), 1):
        observed = set()
        for token in doc:
            kind = "action" if token.tag_ in ACTION_TAGS else "object" if token.tag_ in OBJECT_TAGS else None
            if kind is None:
                continue
            head = normalize(token.text, token.tag_)
            if head not in pools[kind]:
                continue
            context = focused_context(token)
            proposals = [(head, 0, token.text)]
            if kind == "action":
                particles = sorted((c for c in token.children if c.dep_ == "prt"), key=lambda t: t.i)
                if particles:
                    phrase = " ".join([head] + [c.text.casefold() for c in particles])
                    if not (phrase == "throw up" and raises_limbs(token)):
                        proposals.append((phrase, 0, " ".join([token.text] + [c.text for c in particles])))
                for prep in token.children:
                    # "touch Monica on the arm" is not the idiom "touch on a topic".
                    if prep.dep_ == "prep" and not any(c.dep_ in {"dobj", "obj", "iobj", "dative"} for c in token.children):
                        proposals.append((head + " " + prep.text.casefold(), 0, token.text + " " + prep.text))
            else:
                modifiers = sorted((c for c in token.children if c.dep_ in {"compound", "amod"}), key=lambda t: t.i)
                if modifiers:
                    words = [normalize(c.text, c.tag_) if c.tag_ in OBJECT_TAGS else c.text.casefold() for c in modifiers]
                    proposals.append((" ".join(words + [head]), len(words),
                                      " ".join([c.text for c in modifiers] + [token.text])))
            for expression, head_index, form in proposals:
                key = kind, expression
                if key in observed:
                    continue
                observed.add(key)
                add_target(kind, expression, head, head_index, form, token.tag_, row, context)
        if index % 2500 == 0:
            print(f"Validation contexts: {index:,}/{len(validation):,}", flush=True)

    definitions = {}
    for kind, words in source_words.items():
        for word in words:
            for name, sense in lex(word, kind)["senses"].items():
                definitions[name] = sense.definition()
    for kind, word in targets:
        for name, sense in lex(word, kind)["senses"].items():
            definitions[name] = sense.definition()
    texts = sorted(set(definitions.values()) | {q["source_context"] for q in queries} |
                   {r["context"] for r in occurrences})
    model_path = Path("/home/viglab2/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
                      "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    vectors = {}
    for start in range(0, len(texts), 1000):
        chunk = texts[start:start + 1000]
        vectors.update(zip(chunk, model.encode(chunk, batch_size=128, normalize_embeddings=True,
                                               show_progress_bar=False)))
        print(f"Meaning/context embeddings: {min(start + 1000, len(texts)):,}/{len(texts):,}", flush=True)

    def choose_sense(word, kind, context):
        info = lex(word, kind)
        if not info["senses"]:
            return "", -1.0
        best = min(info["senses"], key=lambda name: (-float(vectors[context] @ vectors[definitions[name]]), name))
        return best, float(vectors[context] @ vectors[definitions[best]])

    by_sense = defaultdict(lambda: defaultdict(list))
    for index, occurrence in enumerate(occurrences):
        kind, expression = occurrence["target_key"]
        sense, support = choose_sense(expression, kind, occurrence["context"])
        if support >= args.min_sense_score:
            occurrence["sense_support"] = support
            by_sense[(kind, sense)][(kind, expression)].append(index)

    out = args.output_dir or args.input_dir / "linguistic_synonyms"
    out.mkdir(parents=True, exist_ok=True)
    caption_fields = ["caption_id", "video_id", "enc", "caption", "ood_action_words", "ood_object_words",
                      "action_synonyms", "object_synonyms", "actions_without_synonym", "objects_without_synonym", "synonym_status"]
    pair_fields = ["caption_id", "video_id", "enc", "caption", "term_type", "source_expression",
                   "synonym_expression", "synonym_rank", "meaning", "val_expression", "val_caption_id", "val_caption", "requires_review"]
    counts = Counter()
    missing_reasons = Counter()
    dictionary = defaultdict(lambda: {"synonyms": set(), "captions": set()})
    pair_count = 0
    with (out / "tvrtrain_synonym_candidates.csv").open("w", encoding="utf-8-sig", newline="") as cf, \
            (out / "tvrtrain_synonym_pairs.csv").open("w", encoding="utf-8-sig", newline="") as pf:
        cw, pw = csv.DictWriter(cf, fieldnames=caption_fields), csv.DictWriter(pf, fieldnames=pair_fields)
        cw.writeheader()
        pw.writeheader()
        for index, row in enumerate(train, 1):
            result = {key: row[key] for key in ["caption_id", "video_id", "enc", "caption"]}
            proposals, missing = {k: {} for k in pools}, {k: [] for k in pools}
            for query_index in source_query_map[row["caption_id"]]:
                query = queries[query_index]
                kind, word, surface = query["term_type"], query["word"], query["source_expression"]
                sense, source_support = choose_sense(word, kind, query["source_context"])
                options = []
                if source_support >= args.min_sense_score:
                    for target_key, occurrence_ids in by_sense.get((kind, sense), {}).items():
                        target = targets[target_key]
                        target_info = lex(target["word"], kind)
                        if same_word(lex(word, kind), target_info) or opposed(lex(word, kind), target_info):
                            continue
                        best_id = min(occurrence_ids, key=lambda i: (
                            -(0.6 * float(vectors[query["source_context"]] @ vectors[occurrences[i]["context"]]) +
                              0.4 * occurrences[i]["sense_support"]), occurrences[i]["val_caption_id"]))
                        example = occurrences[best_id]
                        words = target["word"].split()
                        inflections = getInflection(target["head"], query["source_pos"])
                        if not inflections:
                            continue
                        words[target["head_index"]] = inflections[0]
                        expression = " ".join(words)
                        if surface[:1].isupper() and surface[1:].islower():
                            expression = expression[:1].upper() + expression[1:]
                        score = 0.6 * float(vectors[query["source_context"]] @ vectors[example["context"]]) + 0.4 * example["sense_support"]
                        options.append({"expression": expression, "meaning": definitions[sense],
                                        "val_expression": example["val_expression"], "val_caption_id": example["val_caption_id"],
                                        "val_caption": example["val_caption"], "requires_review": True, "score": score})
                options.sort(key=lambda item: (-item["score"], item["expression"], item["val_caption_id"]))
                distinct = {}
                for option in options:
                    distinct.setdefault(option["expression"], option)
                options = list(distinct.values())[:args.top_k]
                for rank, option in enumerate(options, 1):
                    option.pop("score")
                    option["rank"] = rank
                    pair = {key: row[key] for key in ["caption_id", "video_id", "enc", "caption"]}
                    pair.update(term_type=kind, source_expression=surface, synonym_expression=option["expression"],
                                synonym_rank=rank, **{key: option[key] for key in ["meaning", "val_expression", "val_caption_id", "val_caption", "requires_review"]})
                    pw.writerow(pair)
                    pair_count += 1
                proposals[kind][surface] = options
                entry = dictionary[(kind, surface)]
                entry["captions"].add(row["caption_id"])
                entry["synonyms"].update(option["expression"] for option in options)
                if not options:
                    missing[kind].append(surface)
                    missing_reasons["no_wordnet_sense" if not sense else "weak_source_sense" if source_support < args.min_sense_score else "no_distinct_synonym_in_matching_val_usage"] += 1
            for kind in pools:
                result[f"ood_{kind}_words"] = encoded(list(proposals[kind]))
                result[f"{kind}_synonyms"] = encoded(proposals[kind])
                result[f"{kind}s_without_synonym"] = encoded(missing[kind])
            has_any = any(options for mapping in proposals.values() for options in mapping.values())
            state = "NO_VAL_SYNONYM" if not has_any else "PARTIAL_VAL_SYNONYMS" if any(missing.values()) else "ALL_TERMS_HAVE_VAL_SYNONYMS"
            result["synonym_status"] = state
            counts[state] += 1
            cw.writerow(result)
            if index % 2500 == 0:
                print(f"Synonym captions: {index:,}/{len(train):,}", flush=True)
    with (out / "ood_word_synonyms.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["term_type", "source_expression", "synonym_expressions", "train_caption_count"])
        writer.writeheader()
        for (kind, surface), entry in sorted(dictionary.items()):
            writer.writerow({"term_type": kind, "source_expression": surface,
                             "synonym_expressions": encoded(sorted(entry["synonyms"])), "train_caption_count": len(entry["captions"])})
    assert sum(counts.values()) == len(train)
    summary = {"captions": len(train), "synonym_pairs": pair_count, "caption_status": dict(counts),
               "source_word_entries": len(dictionary), "missing_reasons": dict(missing_reasons),
               "method": {"candidates": "Different lexical expressions sharing the selected WordNet sense, supported by a Val occurrence selecting that same sense",
                          "excluded": "Same-word morphology, UK/US spelling variants, hypernyms, embedding-only neighbors and direct antonyms",
                          "source_context": "Target word dependency neighborhood; coordinating, adverbial and relative clauses excluded",
                          "val_scope": "Original action/object pool heads; multiword expressions observed with that head in Val captions",
                          "phrase_guards": "Verb/preposition phrases exclude verbs with direct/indirect objects; throw up with arm/hand arguments is excluded from the vomiting synonym",
                          "surface_forms": "Original OOD words shown; synonym head inflected to the original Penn POS with lemminflect 0.2.3",
                          "minimum_sense_cosine": args.min_sense_score, "top_k_maximum": args.top_k,
                          "review": "Sense selection is heuristic; all candidates require semantic and grammar review before a rewrite",
                          "seed": args.seed, "wordnet_version": wn.get_version(), "spacy_version": spacy.__version__,
                          "spacy_model_version": nlp.meta["version"], "embedding_model_revision": model_path.name},
               "inputs": {path.name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# TVR linguistic synonym candidates\n\n"
        "`tvrtrain_synonym_candidates.csv` retains every original OOD caption and maps its original words to synonym expressions. "
        "`tvrtrain_synonym_pairs.csv` has one row per caption/source expression/synonym. "
        "`ood_word_synonyms.csv` aggregates surface-word alternatives across captions; consult the per-caption result for word sense.\n\n"
        "Only different lexical expressions sharing a WordNet sense are candidates. Both the original word's focused context "
        "and an actual Val occurrence must select that sense, with cosine at least 0.20. This is an uncalibrated heuristic, "
        "not a guarantee of equivalence. Morphology-only and common spelling variants, broader terms, embedding-only neighbors "
        "and direct antonyms are excluded. No nearest-neighbor fallback fills missing synonyms.\n\n"
        "Verb/preposition candidates exclude verbs with direct/indirect objects, so touching a person on the arm "
        "does not become the idiom `touch on`. Raising hands/arms with `throw up` is excluded from the vomiting synonym.\n\n"
        "Single-word candidates have a head in the original matching Val pool. Multiword candidates must be observed "
        "with that head in a Val caption. Output inflects that head to the original POS; thus the suggested inflection can "
        "differ from the observed Val form, recorded separately as `val_expression`. "
        "`requires_review=True` throughout; check meaning, grammar and timestamp-specific action before rewriting. "
        "Future full-caption rewrites must follow object emphasis, pronoun resolution, visible affirmative states, "
        "visible emotion expressions, action preservation and 14–20 words with fewer than 30 subword tokens.\n\n"
        "Caption status: `ALL_TERMS_HAVE_VAL_SYNONYMS` means each OOD word has at least one candidate; "
        "`PARTIAL_VAL_SYNONYMS` means some words have candidates; `NO_VAL_SYNONYM` means none do under this method. "
        "An empty list means no candidate was found by the current corpus and heuristic, not proof that English has no synonym.\n\n"
        "Reproduce:\n\n```bash\nHF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/build_val_synonyms.py\n```\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"method", "inputs"}}, indent=2), flush=True)
    print(f"Saved to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
