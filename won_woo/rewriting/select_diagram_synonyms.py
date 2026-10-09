"""Select conservative Train -> Val synonyms with lexical/syntax rules; no LLM.

Run apply_val_synonyms.py with the resulting approved_pairs.csv and
--enforce-length-budget --apply to preserve IDs and back up the input.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from apply_val_synonyms import digest, write_csv


# These families only shortlist conventional lexical equivalents. Each actual
# occurrence still needs matching POS and explicit syntax checks in BOTH splits.
FAMILIES = {
    'object': [
        'sofa couch',
        'phone telephone',
        'photo photograph',
        'drink beverage',
        'car auto automobile motorcar',
        'shop store',
        'hallway corridor hall',
        'tie necktie',
        'purse handbag pocketbook',
        'tv television',
        'trash garbage rubbish',
        'elevator lift',
        'truck lorry',
        'napkin serviette',
        'bike bicycle',
        'cab taxi taxicab',
        'gun firearm',
        'airplane aeroplane plane',
        'lab laboratory',
        'doctor physician',
        'cop policeman',
        'lawyer attorney',
        'gift present',
        'baby infant',
        'dorm dormitory',
        'corpse cadaver',
        'hat chapeau',
        'tux tuxedo',
        'motorcycle motorbike',
    ],
    'action': [
        'talk speak',
        'close shut',
        'put place',
        'start begin',
    ],
}
BLOCKED = re.compile(
    r"\b(?:he|she|it|they|him|her|his|their|them|its|himself|herself|themselves|"
    r"you|your|we|our|us|I|me|my|mine|who|someone|something|another|"
    r"not|never|no|without|upset|angry|happy|sad|afraid|worried|concerned|"
    r"confused|surprised|embarrassed|jealous|thinks?|knows?|believes?|realizes?|"
    r"wants?|needs?|feels?|remembers?|hopes?|wishes?|decision|thought|idea|fear|anger|sadness)\b|\b\w+['’]t\b", re.I)


def read_captions(path):
    data = path.read_bytes()
    captions = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"(\S+#enc#\d+)[ \t]+(.*)", line)
        if not match or match[1] in captions:
            raise ValueError(f"Invalid/duplicate caption ID in {path}: {line[:100]}")
        captions[match[1]] = match[2]
    return data, captions


NONVISUAL = {"think", "know", "believe", "realize", "want", "need", "feel", "remember", "hope", "wish", "expect", "imagine", "understand", "consider", "suppose", "decide", "plan", "love", "hate", "like", "prefer", "suspect", "wonder", "try"}
SPEECH = {"say", "tell", "talk", "speak", "ask", "explain", "describe", "discuss", "mention"}
LOCATIONS = {"on", "in", "into", "at", "under", "over", "by", "onto", "behind", "beside", "inside", "across"}


def syntax_status(tokens, kind, lemma):
    for token in tokens:
        if token.dep_ == "compound" or any(c.dep_ == "compound" for c in token.children):
            return "compound_preserved"
        if kind == "object" and lemma in {"elevator", "lift"} and not (
                token.dep_ == "pobj" and token.head.lower_ in LOCATIONS | {"to", "from", "near", "toward", "towards"}):
            return "ambiguous_lift_context"
        if kind == "object" and lemma == "drink" and (
                token.head.lemma_ in {"take", "swallow", "sip"} or
                (token.i > 0 and token.doc[token.i - 1].lemma_ == "be")):
            return "drink_action_not_beverage"
        if kind == "object" and lemma in {"bike", "bicycle"} and any(
                t.lower_ in {"stationary", "exercise", "motor", "motorcycle", "motorbike"} for t in token.sent):
            return "ambiguous_bike_context"
        if kind == "object" and lemma == "bike" and not any(t.lemma_ == "pedal" for t in token.sent):
            return "ambiguous_bike_context"
        if kind == "action":
            children = list(token.children)
            if lemma in {"put", "place"} and not (any(c.dep_ == "dobj" for c in children) and any(c.dep_ == "prep" and c.lower_ in LOCATIONS for c in children)):
                return "no_physical_placement_context"
            if lemma in {"start", "begin"} and not any(c.dep_ == "xcomp" and c.pos_ == "VERB" and c.lemma_ not in NONVISUAL for c in children):
                return "no_visible_inchoative_action"
            if lemma in {"close", "shut"} and not any(c.dep_ == "dobj" for c in children):
                return "no_object_to_close"
            if lemma in {"talk", "speak"} and any(c.dep_ in {"dobj", "ccomp"} or c.lower_ in {"about", "of"} for c in children):
                return "speech_content_or_argument_change"
        elif token.head.pos_ == "VERB" and token.head.lemma_ in {"order", "request", "mention", "describe", "discuss"}:
            return "dialogue_only_object"
        elif any(t.lemma_ in SPEECH and any(c.dep_ == "ccomp" or c.lower_ == "about" for c in t.children) for t in token.ancestors):
            return "dialogue_only_object"
        elif lemma in {"hall", "hallway", "corridor"} and not (token.dep_ == "pobj" and token.head.lower_ in LOCATIONS | {"through", "down", "along"}):
            return "ambiguous_hall_context"
        elif lemma in {"car", "auto", "automobile", "motorcar"} and any(t.lower_ in {"railway", "railroad", "train", "cable", "subway"} for t in token.sent):
            return "ambiguous_vehicle_context"
    return ""


def main():
    import spacy
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, default=root / "diagram_synonym_pairs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--captions", type=Path, default=root / "tvr/TextData/tvrtrain.caption.txt")
    parser.add_argument("--val-captions", type=Path, default=root / "tvr/TextData/tvrval.caption.txt")
    args = parser.parse_args()
    metadata = json.loads((args.pairs_dir / "summary.json").read_text())
    if not metadata["method"].get("dictionary_only"):
        raise ValueError("Only dictionary-only diagram pairs may be selected; rebuild the diagram first")
    dictionaries = {kind: json.loads((root / f"tvr/TextData/tvr_id_{kind}_dict.json").read_text())
                    for kind in FAMILIES}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "rewriting/nlp_resources/python"))
    from lemminflect import getAllLemmas

    spacy.util.fix_random_seed(2026)
    train_path = args.captions
    val_path = args.val_captions
    train_data, train = read_captions(train_path)
    val_data, val = read_captions(val_path)
    eligible = {cid for cid, caption in train.items()
                if 14 <= len(caption.split()) <= 20 and not BLOCKED.search(caption)}
    print(f"Eligible original captions: {len(eligible):,}", flush=True)
    allowed = defaultdict(set)
    for kind, families in FAMILIES.items():
        for family in families:
            words = {lemma for word in family.split() for lemma in
                     getAllLemmas(word).get("NOUN" if kind == "object" else "VERB", (word,))}
            for word in words:
                allowed[kind, word].update(words - {word})
    counts, grouped, inputs = Counter(), defaultdict(list), {}
    lemma_cache = {}
    for kind in FAMILIES:
        path = args.pairs_dir / f"{kind}_B_train.csv"
        inputs[str(path)] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), 1):
                counts[f"{kind}_rows"] += 1
                cid = row["caption_id"]
                if cid not in train or row["caption"].strip() != train[cid].strip():
                    raise ValueError(f"CSV/train mismatch: {cid}")
                if not row["synonym_expression"]:
                    counts["no_synonym"] += 1
                    continue
                tag = dictionaries[kind].get(cid, {}).get(row["source_expression"])
                if tag != row["source_pos"] or dictionaries[kind].get(
                        row["synonym_dictionary_caption_id"], {}).get(row["synonym_expression"]) != tag:
                    raise ValueError(f"Source or replacement lacks supplied dictionary evidence: {cid}")
                if dictionaries[kind].get(row["target_dictionary_caption_id"], {}).get(
                        row["val_expression"]) != row["target_source_pos"]:
                    raise ValueError(f"Val witness lacks supplied dictionary evidence: {cid}")
                if cid not in eligible:
                    counts["original_length_or_grounding_filter"] += 1
                    continue
                if row["term_region"] != "B" or row["review_decision"] == "EXCLUDE":
                    raise ValueError("Excluded row in B file")
                target = row["synonym_expression"].casefold()
                key = kind, target
                if key not in lemma_cache:
                    lemma_cache[key] = getAllLemmas(target).get("NOUN" if kind == "object" else "VERB", ())
                if not allowed[kind, row["source_lemma"]].intersection(lemma_cache[key]):
                    counts["outside_conservative_families"] += 1
                    continue
                vid = row["val_caption_id"]
                if vid not in val or row["val_caption"].strip() != val[vid].strip():
                    raise ValueError(f"CSV/val mismatch: {vid}")
                row.update(csv_file=path.name, data_row_number=index, original_synonym_rank=row["synonym_rank"])
                grouped[cid].append(row)
        print(f"Read {path.name}; shortlisted {sum(map(len, grouped.values())):,} pairs", flush=True)

    nlp = spacy.load("en_core_web_md", disable=["ner"])
    texts = sorted({train[cid] for cid in grouped} | {
        row["val_caption"] for rows in grouped.values() for row in rows})
    docs = dict(zip(texts, nlp.pipe(texts, batch_size=128)))
    token_path = Path("/home/viglab2/.cache/huggingface/hub/models--roberta-base/snapshots/"
                      "e2da8e2f811d1448a5b465c236feacd80ffbac7b")
    budget_tokenizer = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
    requests, rejected = [], []
    for cid, rows in grouped.items():
        candidates = []
        for row in rows:
            source = row["source_expression"]
            source_tokens = [t for t in docs[train[cid]] if t.text == source]
            witness_tokens = [t for t in docs[row["val_caption"]] if t.text == row["val_expression"]]
            pos = "NOUN" if row["term_type"] == "object" else "VERB"
            pattern = re.compile(r"(?<!\w)" + re.escape(source) + r"(?!\w)")
            rewritten = pattern.sub(row["synonym_expression"], train[cid])
            status = ""
            if not source_tokens or not witness_tokens or not pattern.search(train[cid]):
                status = "source_or_witness_not_token"
            elif any(t.pos_ != pos or t.tag_ != row["source_pos"] for t in source_tokens) or any(
                    t.pos_ != pos for t in witness_tokens):
                status = "context_pos_mismatch"
            elif pos == "VERB" and any(child.dep_ == "prt" for t in source_tokens + witness_tokens for child in t.children):
                status = "phrasal_verb_preserved"
            elif not (14 <= len(rewritten.split()) <= 20 and len(budget_tokenizer.encode(rewritten)) < 30):
                status = "length_budget_exceeded"
            if not status:
                status = syntax_status(source_tokens, kind=row["term_type"], lemma=row["source_lemma"])
            if not status:
                status = syntax_status(witness_tokens, kind=row["term_type"], lemma=witness_tokens[0].lemma_)
            doc = docs[train[cid]]
            if not status and any(t.pos_ == "VERB" and t.lemma_ in NONVISUAL for t in doc):
                status = "nonvisual_caption_preserved"
            if not status and any(t.lemma_ in SPEECH and any(c.dep_ == "ccomp" for c in t.children) for t in doc):
                status = "dialogue_content_preserved"
            if not status and any(t.i > 0 and doc[t.i - 1].lower_ in {"a", "an"} and
                    ((doc[t.i - 1].lower_ == "an") != (row["synonym_expression"][0].lower() in "aeiou")) for t in source_tokens):
                status = "article_mismatch"
            if status:
                rejected.append(dict(row, status=status))
            else:
                candidates.append(row)
        if candidates:
            requests.append({"caption_id": cid, "caption": train[cid], "rows": candidates})
    selected, audit = [], rejected
    for request in requests:
        used = set()
        for row in sorted(request["rows"], key=lambda r: (r["term_type"], r["source_expression"], int(r["synonym_rank"]))):
            status = "alternative_for_same_source" if row["source_expression"] in used else "rule_selected"
            if status == "rule_selected":
                used.add(row["source_expression"])
                selected.append(dict(row, synonym_rank=1))
            audit.append(dict(row, status=status))
    fields = list(next(iter(grouped.values()))[0])
    write_csv(out / "approved_pairs.csv", fields, selected)
    write_csv(out / "rule_pair_audit.csv", fields + ["status"], audit)
    if train_path.read_bytes() != train_data or val_path.read_bytes() != val_data:
        raise ValueError("Caption inputs changed during selection")
    for name, info in inputs.items():
        stat = Path(name).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != info:
            raise ValueError(f"Pair file changed: {name}")
    summary = {"train_file": str(train_path), "train_sha256": digest(train_data),
        "val_file": str(val_path), "val_sha256": digest(val_data), "pair_inputs": inputs,
        "scan_counts": dict(counts), "eligible_original_captions": len(eligible),
        "shortlisted_pairs": len(audit), "pair_status_counts": dict(Counter(r["status"] for r in audit)),
        "selected_pairs": len(selected), "selected_captions": len({r["caption_id"] for r in selected}),
        "families": FAMILIES, "blocked_pattern": BLOCKED.pattern, "seed": 2026,
        "llm_used_for_selection": False, "video_grounding_verified": False,
        "script_sha256": digest(Path(__file__).read_bytes()), "tokenizer": str(token_path),
        "policy": "Conservative lexical families, observed Train/Val expressions, contextual POS and compound/phrasal/argument/article rules. 14-20 words and <30 RoBERTa tokens including special tokens; combined budget rechecked at apply time."}
    (out / "rule_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k in {"selected_pairs","selected_captions","pair_status_counts"}}, indent=2))


if __name__ == "__main__":
    main()
