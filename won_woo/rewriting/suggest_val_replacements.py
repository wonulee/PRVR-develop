"""Retrieve reviewable, context-aware OOD replacements from the matching Val pool."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze_val_domain import ACTION_TAGS, encoded, make_normalizer


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_wordnet(resource_dir):
    import nltk
    nltk.data.path.insert(0, str(resource_dir.resolve()))
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
    return wordnet


def lexical_info(wn, word, kind):
    pos = wn.VERB if kind == "action" else wn.NOUN
    lookup = word.replace(" ", "_")
    senses = wn.synsets(lookup, pos=pos)
    return {
        "base": wn.morphy(lookup, pos),
        "senses": {sense.name(): sense for sense in senses},
        "hypernyms": {sense.name(): {h.name() for h in sense.hypernyms()} for sense in senses},
        "antonyms": {ant.synset().name() for sense in senses for lemma in sense.lemmas()
                     for ant in lemma.antonyms()},
    }


def opposed(source, target):
    return bool(source["antonyms"] & set(target["senses"]) or
                target["antonyms"] & set(source["senses"]))


def relation(source, target, chosen_sense):
    if source["base"] and source["base"] == target["base"]:
        return "morphological_variant"
    target_senses = set(target["senses"])
    if chosen_sense in target_senses:
        return "synonym_in_selected_sense"
    if source["hypernyms"].get(chosen_sense, set()) & target_senses:
        return "broader_hypernym_in_selected_sense"
    if (set(source["senses"]) & target_senses or any(
            names & target_senses for names in source["hypernyms"].values())):
        return "other_source_sense"
    return "embedding_related_only"


def status_for(rel, word_similarity, threshold):
    if rel in {"morphological_variant", "synonym_in_selected_sense"}:
        return "LEXICAL_RELATION_REVIEW"
    if rel == "broader_hypernym_in_selected_sense":
        return "BROADER_MEANING_REVIEW"
    return "SIMILARITY_ONLY_REVIEW" if word_similarity >= threshold else "LOW_SIMILARITY_REVIEW"


def semantic_tier(rel):
    if rel in {"morphological_variant", "synonym_in_selected_sense"}:
        return 0
    return 1 if rel == "broader_hypernym_in_selected_sense" else 2


def check_lexical_rules(wn):
    assert opposed(lexical_info(wn, "zip", "action"), lexical_info(wn, "unzip", "action"))
    source = lexical_info(wn, "zip", "action")
    fast_sense = next(name for name, sense in source["senses"].items() if "hurry" in sense.lemma_names())
    closure_sense = next(name for name, sense in source["senses"].items() if "zip_up" in sense.lemma_names())
    target = lexical_info(wn, "hurry", "action")
    assert relation(source, target, fast_sense) == "synonym_in_selected_sense"
    assert relation(source, target, closure_sense) == "other_source_sense"
    assert relation(lexical_info(wn, "thumbs", "object"), lexical_info(wn, "thumb", "object"), "") == "morphological_variant"
    vomit = lexical_info(wn, "vomit", "action")
    assert relation(vomit, lexical_info(wn, "throw up", "action"), "vomit.v.01") == "synonym_in_selected_sense"
    assert semantic_tier("synonym_in_selected_sense") < semantic_tier("embedding_related_only")


def main():
    import nltk
    import numpy as np
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "rewriting/val_domain_analysis")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(
        "/home/viglab2/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
        "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"))
    parser.add_argument("--wordnet-dir", type=Path, default=workspace / "rewriting/nlp_resources")
    parser.add_argument("--spacy-model", default="en_core_web_md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.top_k < 1 or not 0 <= args.similarity_threshold <= 1:
        parser.error("top-k must be positive and similarity-threshold must be in [0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    wn = load_wordnet(args.wordnet_dir)
    check_lexical_rules(wn)
    inputs = [args.input_dir / name for name in ["tvrtrain_rewriting_candidates.csv",
              "train_ood_lemma_frequency.csv", "tvrval_action_object_analysis.csv",
              "val_action_pool.csv", "val_object_pool.csv"]]
    captions = read_csv(inputs[0])
    ood_rows = read_csv(inputs[1])
    validation = {row["caption_id"]: row for row in read_csv(inputs[2])}
    pools = {kind: {row["lemma"]: row for row in read_csv(path)}
             for kind, path in zip(["action", "object"], inputs[3:])}
    sources = {(row["term_type"], row["lemma"]): row for row in ood_rows}
    if not captions or not sources or any(not pool for pool in pools.values()):
        raise ValueError("Nonempty train OOD captions, frequency rows and Val pools are required")
    assert all(row["decision"] == "REWRITE" for row in captions)
    observed = Counter((kind, word) for row in captions for kind in pools
                       for word in json.loads(row[f"ood_{kind}s"]))
    assert observed == Counter({key: int(row["train_caption_count"]) for key, row in sources.items()})
    assert all(word not in pools[kind] for kind, word in sources)
    for kind, pool in pools.items():
        for word, row in pool.items():
            row["example_ids"] = json.loads(row["example_caption_ids"])
            if not row["example_ids"]:
                raise ValueError(f"Missing validation evidence: {kind}:{word}")
            assert all(word in json.loads(validation[key][f"{kind}_lemmas"])
                       for key in row["example_ids"])

    # Keep the head inside the original Action Pool while retaining observed particles.
    # This prevents suggesting bare "throw" for the distinct action "throw up".
    import spacy
    spacy.util.fix_random_seed(args.seed)
    nlp = spacy.load(args.spacy_model, disable=["ner"])
    normalize = make_normalizer(nlp)
    phrases = {}
    print("Extracting observed validation phrasal verbs", flush=True)
    for row, doc in zip(validation.values(), nlp.pipe(
            (row["caption"] for row in validation.values()), batch_size=128, n_process=1)):
        in_caption = set()
        for token in doc:
            if token.tag_ not in ACTION_TAGS:
                continue
            particles = sorted((child for child in token.children if child.dep_ == "prt"), key=lambda t: t.i)
            if not particles:
                continue
            head = normalize(token.text, token.tag_)
            expression = " ".join([head] + [part.text.casefold() for part in particles])
            if head not in pools["action"] or head not in json.loads(row["action_lemmas"]):
                continue
            info = phrases.setdefault(expression, {"lemma": head, "val_caption_count": 0,
                                                    "forms": set(), "example_ids": []})
            info["forms"].add(" ".join([token.text] + [part.text for part in particles]))
            if expression not in in_caption:
                info["val_caption_count"] += 1
                in_caption.add(expression)
                # Phrases are polysemous: early "throw up" rows may only raise hands.
                # Keep all occurrences so a later vomiting example remains retrievable.
                info["example_ids"].append(row["caption_id"])
    targets = {kind: dict(pool) for kind, pool in pools.items()}
    for expression, info in sorted(phrases.items()):
        targets["action"][expression] = {"lemma": info["lemma"],
                                         "val_caption_count": info["val_caption_count"],
                                         "surface_forms": encoded(sorted(info["forms"])),
                                         "example_ids": info["example_ids"]}
    print(f"Observed validation phrasal verbs: {len(phrases):,}", flush=True)

    print("Loading local MiniLM and WordNet; inference runs on CPU", flush=True)
    model = SentenceTransformer(str(args.model_path), device="cpu", local_files_only=True)
    words = sorted({word for _, word in sources} | {word for pool in targets.values() for word in pool})

    def embed(texts, label):
        parts = []
        for start in range(0, len(texts), 1000):
            parts.append(model.encode(texts[start:start + 1000], batch_size=128,
                                      normalize_embeddings=True, show_progress_bar=False))
            print(f"{label}: {min(start + 1000, len(texts)):,}/{len(texts):,}", flush=True)
        return np.concatenate(parts) if parts else np.empty((0, model.get_sentence_embedding_dimension()))

    word_vectors = dict(zip(words, embed(words, "Term embeddings")))
    lexical = {kind: {} for kind in pools}
    sense_definitions = {}
    for kind in pools:
        for word in sorted(set(targets[kind]) | {w for k, w in sources if k == kind}):
            info = lexical_info(wn, word, kind)
            lexical[kind][word] = info
            if (kind, word) in sources:
                for name, sense in info["senses"].items():
                    sense_definitions[name] = sense.definition()
    sense_names = sorted(sense_definitions)
    sense_vectors = dict(zip(sense_names, embed(
        [sense_definitions[name] for name in sense_names], "WordNet sense embeddings")))
    example_ids = sorted({key for pool in targets.values() for row in pool.values() for key in row["example_ids"]})
    val_vectors = dict(zip(example_ids, embed(
        [validation[key]["caption"] for key in example_ids], "Validation context embeddings")))
    train_vectors = embed([row["caption"] for row in captions], "Train context embeddings")
    train_indices = {row["caption_id"]: index for index, row in enumerate(captions)}

    def select_sense(info, context):
        names = sorted(info["senses"])
        if not names:
            return "", "", ""
        scored = sorted(((float(context @ sense_vectors[name]), name) for name in names),
                        key=lambda item: (-item[0], item[1]))
        margin = scored[0][0] - scored[1][0] if len(scored) > 1 else ""
        return scored[0][1], round(scored[0][0], 6), round(margin, 6) if margin != "" else ""

    shortlists = {}
    short_size = max(30, args.top_k * 4)
    for kind, pool in targets.items():
        target_words = sorted(pool)
        matrix = np.stack([word_vectors[word] for word in target_words])
        by_sense = {}
        for word in target_words:
            for name in lexical[kind][word]["senses"]:
                by_sense.setdefault(name, set()).add(word)
        for key in sorted(key for key in sources if key[0] == kind):
            _, source_word = key
            info = lexical[kind][source_word]
            similarities = matrix @ word_vectors[source_word]
            nearest = np.argsort(-similarities, kind="stable")[:short_size]
            eligible = {target_words[index] for index in nearest}
            for name in info["senses"]:
                eligible.update(by_sense.get(name, set()))
                for hypernym in info["hypernyms"][name]:
                    eligible.update(by_sense.get(hypernym, set()))
            if info["base"] in pool:
                eligible.add(info["base"])
            shortlist = []
            for target_word in sorted(eligible):
                target = lexical[kind][target_word]
                if opposed(info, target):
                    continue
                shortlist.append({"lemma": target_word,
                                  "word_similarity": float(word_vectors[source_word] @ word_vectors[target_word])})
            shortlists[key] = shortlist

    def rank_candidates(key, context):
        kind, source_word = key
        info = lexical[kind][source_word]
        chosen, sense_similarity, sense_margin = select_sense(info, context)
        ranked = []
        for item in shortlists[key]:
            word = item["lemma"]
            rel = relation(info, lexical[kind][word], chosen)
            # WordNet links belonging exclusively to another source sense are not proposals.
            if rel == "other_source_sense":
                continue
            examples = targets[kind][word]["example_ids"]
            def example_score(example):
                contextual = float(context @ val_vectors[example])
                if chosen and rel == "synonym_in_selected_sense":
                    return 0.7 * contextual + 0.3 * float(sense_vectors[chosen] @ val_vectors[example])
                return contextual
            best_id = min(examples, key=lambda example: (-example_score(example), example))
            context_similarity = float(context @ val_vectors[best_id])
            bonus = 0.12 if rel in {"morphological_variant", "synonym_in_selected_sense"} else (
                0.04 if rel == "broader_hypernym_in_selected_sense" else 0.0)
            score = 0.8 * item["word_similarity"] + 0.2 * context_similarity + bonus
            ranked.append({"candidate_lemma": targets[kind][word]["lemma"],
                           "candidate_expression": word,
                           "semantic_tier": semantic_tier(rel),
                           "relation": rel,
                           "word_similarity": round(item["word_similarity"], 6),
                           "context_similarity": round(context_similarity, 6),
                           "ranking_score": round(score, 6),
                           "candidate_status": status_for(rel, item["word_similarity"], args.similarity_threshold),
                           "val_caption_count": int(targets[kind][word]["val_caption_count"]),
                           "val_surface_forms": json.loads(targets[kind][word]["surface_forms"]),
                           "val_caption_id": best_id,
                           "val_caption": validation[best_id]["caption"],
                           "source_sense": chosen,
                           "source_sense_definition": sense_definitions.get(chosen, ""),
                           "source_sense_similarity": sense_similarity,
                           "source_sense_margin": sense_margin,
                           "requires_semantic_review": True,
                           "safe_to_auto_substitute": False})
        ranked.sort(key=lambda item: (item["semantic_tier"], -item["ranking_score"],
                                      -item["val_caption_count"], item["candidate_expression"]))
        return [dict(item, candidate_rank=index) for index, item in enumerate(ranked[:args.top_k], 1)]

    def term_status(candidates):
        if any(c["candidate_status"] == "LEXICAL_RELATION_REVIEW" for c in candidates):
            return "LEXICAL_CANDIDATE_AVAILABLE_REVIEW"
        if any(c["word_similarity"] >= args.similarity_threshold for c in candidates):
            return "SIMILARITY_CANDIDATE_AVAILABLE_REVIEW"
        return "NO_STRONG_CANDIDATE_REVIEW"

    out = args.output_dir or args.input_dir / "semantic_candidates"
    out.mkdir(parents=True, exist_ok=True)
    example_candidate = next((rank_candidates(key, train_vectors[train_indices[
        json.loads(row["example_caption_ids"])[0]]])[0]
        for key, row in sources.items() if shortlists[key]), None)
    if example_candidate is None:
        raise ValueError("No candidates after semantic filtering")
    recommendation_fields = list(example_candidate)
    term_fields = ["term_type", "ood_lemma", "ood_surface_forms", "train_caption_count",
                   "source_has_wordnet_sense", "term_status"] + recommendation_fields
    global_counts = Counter()
    with (out / "ood_lemma_val_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=term_fields)
        writer.writeheader()
        for key, row in sources.items():
            ids = json.loads(row["example_caption_ids"])
            context = np.mean([train_vectors[train_indices[key_id]] for key_id in ids], axis=0)
            context /= max(float(np.linalg.norm(context)), 1e-12)
            candidates = rank_candidates(key, context)
            term_state = term_status(candidates)
            global_counts[term_state] += 1
            for candidate in candidates:
                result = {"term_type": key[0], "ood_lemma": key[1],
                          "ood_surface_forms": row["surface_forms"],
                          "train_caption_count": row["train_caption_count"],
                          "source_has_wordnet_sense": int(bool(lexical[key[0]][key[1]]["senses"])),
                          "term_status": term_state, **candidate}
                result["val_surface_forms"] = encoded(result["val_surface_forms"])
                writer.writerow(result)

    caption_fields = list(captions[0]) + ["action_replacement_candidates", "object_replacement_candidates",
                                         "terms_without_strong_candidate", "replacement_review_status"]
    pair_fields = ["caption_id", "video_id", "enc", "caption", "term_type", "ood_lemma",
                   "ood_surface_forms", "source_has_wordnet_sense", "term_status"] + recommendation_fields
    pair_count = 0
    pair_relations = Counter()
    caption_counts = Counter()
    term_occurrences = Counter()
    with (out / "tvrtrain_rewriting_with_val_candidates.csv").open("w", encoding="utf-8-sig", newline="") as caption_file, \
            (out / "tvrtrain_val_replacement_pairs.csv").open("w", encoding="utf-8-sig", newline="") as pair_file:
        caption_writer = csv.DictWriter(caption_file, fieldnames=caption_fields)
        pair_writer = csv.DictWriter(pair_file, fieldnames=pair_fields)
        caption_writer.writeheader()
        pair_writer.writeheader()
        for index, (row, context) in enumerate(zip(captions, train_vectors), 1):
            result = dict(row)
            unresolved = []
            for kind in pools:
                proposals = {}
                original = json.loads(row[f"ood_{kind}_surfaces"])
                for source_word in json.loads(row[f"ood_{kind}s"]):
                    key = kind, source_word
                    candidates = rank_candidates(key, context)
                    term_state = term_status(candidates)
                    term_occurrences[term_state] += 1
                    proposals[source_word] = candidates
                    if term_state == "NO_STRONG_CANDIDATE_REVIEW":
                        unresolved.append({"term_type": kind, "ood_lemma": source_word})
                    for candidate in candidates:
                        pair = {field: row[field] for field in ["caption_id", "video_id", "enc", "caption"]}
                        pair.update(term_type=kind, ood_lemma=source_word,
                                    ood_surface_forms=encoded([word for word, lemma in original.items() if lemma == source_word]),
                                    source_has_wordnet_sense=int(bool(lexical[kind][source_word]["senses"])),
                                    term_status=term_state, **candidate)
                        pair["val_surface_forms"] = encoded(pair["val_surface_forms"])
                        pair_writer.writerow(pair)
                        pair_count += 1
                        pair_relations[candidate["relation"]] += 1
                result[f"{kind}_replacement_candidates"] = encoded(proposals)
            review_status = "HAS_UNRESOLVED_TERMS_REVIEW" if unresolved else "CANDIDATES_AVAILABLE_REVIEW"
            result["terms_without_strong_candidate"] = encoded(unresolved)
            result["replacement_review_status"] = review_status
            caption_writer.writerow(result)
            caption_counts[review_status] += 1
            if index % 2500 == 0:
                print(f"Candidate captions: {index:,}/{len(captions):,}", flush=True)

    summary = {
        "train_ood_captions": len(captions), "unique_ood_lemmas": len(sources),
        "observed_val_phrasal_verbs": len(phrases),
        "top_k": args.top_k, "caption_candidate_pairs": pair_count,
        "caption_review_status": dict(caption_counts), "global_term_status": dict(global_counts),
        "term_occurrence_status": dict(term_occurrences), "pair_relations": dict(pair_relations),
        "method": {
            "pool_constraint": "Action heads only from Val Action Pool; objects only from Val Object Pool; multiword actions add only observed verb particles",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "local_model_path": str(args.model_path.resolve()),
            "model_revision": args.model_path.name,
            "retrieval": f"Nearest {short_size} lemmas + WordNet same-synset/hypernym/morphology matches",
            "ranking_score": "0.8 * word cosine + 0.2 * best Val example cosine + relation bonus (0.12 synonym/morphology; 0.04 broader term)",
            "ranking_priority": "First selected-sense synonyms/morphology (tier 0), then broader hypernyms (tier 1), then embedding-only results (tier 2); score ranks within a tier",
            "multiword_actions": "Only verb + dependency-prt particles observed in an actual validation caption; normalized head is in original Val Action Pool",
            "spacy_model": args.spacy_model, "spacy_model_version": nlp.meta["version"],
            "context": "Original train caption; up to three stored pool examples for single lemmas; all observed occurrences for phrasal verbs",
            "val_example_selection": "Highest train-caption cosine; for selected-sense synonyms, 0.7 train-caption cosine + 0.3 source-sense-definition cosine",
            "sense_selection": "Highest cosine between original caption embedding and source WordNet sense definition embedding",
            "sense_filter": "Exclude lexical matches supported exclusively by an unselected source sense",
            "antonym_filter": "Exclude candidates whose WordNet senses are direct antonyms of any source sense, in either direction",
            "threshold": args.similarity_threshold,
            "threshold_scope": "Uncalibrated word-cosine review flag, never an acceptance probability",
            "global_vs_caption": "Global term rows use up to three original train example captions; per-caption rankings use that exact caption",
            "all_candidates_require_review": True, "automatic_substitution": False,
            "limitations": ["WordNet sense selection is heuristic and can be wrong.",
                           "Embedding proximity can reflect related or opposing concepts; WordNet does not cover every antonym.",
                           "Hypernyms lose specificity and are not interchangeable synonyms.",
                           "Source typos, wrong POS, abbreviations and proper names can produce weak candidates.",
                           "Pool membership does not prove pixel grounding, action preservation or timestamp equivalence."],
            "seed": args.seed, "device": "cpu", "torch_threads": 4,
            "torch_version": torch.__version__, "sentence_transformers_version": sentence_transformers.__version__,
            "nltk_version": nltk.__version__, "wordnet_version": wn.get_version(),
        },
        "inputs": {path.name: {"path": str(path.resolve()),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Validation-pool semantic candidates\n\n"
        "`tvrtrain_rewriting_with_val_candidates.csv`: all original OOD captions, with JSON action/object candidate lists per OOD lemma.\n"
        "`tvrtrain_val_replacement_pairs.csv`: one row per caption, OOD lemma and candidate; easier to filter in Excel.\n"
        "`ood_lemma_val_candidates.csv`: shared lemma-level candidates, using the original train examples as context.\n"
        "`summary.json`: counts, ranking method, model/corpus versions, input hashes and limitations.\n\n"
        f"Candidates are the top {args.top_k} after same-type Val pool membership, direct WordNet antonym exclusion "
        "and heuristic source-sense filtering. Each includes term cosine, validation context cosine, relation, "
        "source sense definition and an actual validation caption containing its lemma/expression. All observed "
        "phrasal-verb occurrences are eligible as evidence; synonym example selection also uses the selected sense definition. "
        "Caption rankings may differ "
        "from global lemma rankings because the source caption can change the selected sense.\n\n"
        "`candidate_expression` is the expression to consider; `candidate_lemma` is its pool member/head. "
        "Actions may include verb particles only when the phrasal expression is observed in a validation caption, "
        "e.g. `throw up` retains the `throw` head in the Action Pool. Objects remain original Object Pool lemmas. "
        "Tier 0 (morphology/same-sense synonyms) always precedes tier 1 (broader hypernyms), "
        "which precedes tier 2 (embedding-only). Scores order candidates within each tier. "
        "Relations distinguish morphology, same-sense synonym, broader hypernym and embedding-only proximity. "
        "All candidates require semantic review; `safe_to_auto_substitute=False` throughout. "
        "A broader term loses detail; a nearby vector can be a co-occurring object or action rather than a synonym. "
        "WordNet can miss antonyms and the embedding model can select the wrong sense. "
        f"The word-cosine cutoff {args.similarity_threshold} only marks weak retrieval results, not equivalence. "
        "`terms_without_strong_candidate` identifies OOD terms without a selected lexical/morphological match "
        "or sufficiently similar embedding result. Those terms still retain their nearest retrieval candidates for inspection.\n\n"
        "For a later LLM rewrite, use the original caption and supported candidate meanings, preserve the unique "
        "timestamp-specific action and visible objects/interactions, resolve pronouns without inventing identity, "
        "convert negation to visible affirmative states and emotions to visible expressions, and enforce "
        "14–20 words and fewer than 30 subword tokens. If the object/action cannot be faithfully expressed using "
        "the pool, retain it for manual review. No rewritten caption is generated by this script.\n\n"
        "Reproduce from the workspace:\n\n```bash\n"
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/viglab2/miniconda3/envs/holmes/bin/python "
        "rewriting/suggest_val_replacements.py\n```\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"method", "inputs"}}, indent=2), flush=True)
    print(f"Saved to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
