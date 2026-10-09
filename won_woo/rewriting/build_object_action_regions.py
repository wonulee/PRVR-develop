"""Export TVR query coverage regions separately for objects and actions.

Coverage is identical normalized lemma OR a shared same-POS WordNet synset.
WordNet matches are candidates, not context-validated semantic equivalence.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from analyze_val_domain import encoded, read_captions, write_csv, dictionary_sources, validate_dictionary_terms
from build_val_synonyms import spelling_key
from suggest_val_replacements import load_wordnet, read_csv


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_region(split, terms, uncovered):
    if not terms:
        return "EMPTY"
    return ("A" if split == "train" else "C") if uncovered else "B"


def self_check():
    assert query_region("train", [], []) == "EMPTY"
    assert query_region("validation", [], []) == "EMPTY"
    assert query_region("train", ["cup"], []) == "B"
    assert query_region("validation", ["cup"], ["cup"]) == "C"
    assert query_region("train", ["cup", "helmet"], ["helmet"]) == "A"


def distinct_expression(wn, source, target, pos):
    """Exclude the source itself, morphology-only changes, and spelling variants."""
    if spelling_key(source.replace("_", " ")) == spelling_key(target.replace("_", " ")):
        return False
    source_base = wn.morphy(source.replace(" ", "_"), pos)
    target_base = wn.morphy(target.replace(" ", "_"), pos)
    return not (source_base and source_base == target_base)


def synonym_options(wn, source, pos, senses, other_index, other_senses, examples, captions):
    by_sense, words = [], set()
    for name in sorted(senses[source]):
        sense = wn.synset(name)
        alternatives = sorted({word for word in other_senses if name in other_senses[word]
                               and distinct_expression(wn, source, word, pos)})
        words.update(alternatives)
        by_sense.append({"synset": name, "meaning": sense.definition(), "synonyms": alternatives})
    observed = sorted({target for name in senses[source] for target in other_index[name]
                       if distinct_expression(wn, source, target, pos)})
    evidence = [{"other_lemma": target, "shared_synsets": sorted(senses[source] & other_senses[target]),
                 "example_caption_ids": examples[target],
                 "example_captions": [captions[key] for key in examples[target]]} for target in observed]
    return {"wordnet": sorted(words), "other_split": observed, "by_sense": by_sense, "evidence": evidence}


def plot_regions(out, counts, totals, pools, unit="query", empty_counts=None, unclassified_counts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    train_color, val_color = "#153b56", "#e84839"
    units = {"query": "queries", "lemma": "lemmas", "mapping": "mappings"}[unit]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Object / Action: TVR {unit} coverage regions", fontsize=23, y=0.97, weight="bold")
    fig.text(0.5, 0.918, "Covered = identical normalized lemma OR shared same-POS WordNet synset candidate",
             ha="center", fontsize=12, color="#334155")
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=color,
                      markeredgecolor=color, markersize=9, label=label)
               for color, label in [(train_color, f"Training {unit} origin"), (val_color, f"Validation {unit} origin")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False, fontsize=12)
    for ax, kind, pos in zip(axes, ["object", "action"], ["NOUN", "VERB"]):
        panel_totals = totals if unit == "query" else totals[kind]
        ax.set_xlim(-2, 102)
        ax.set_ylim(-9, 105)
        ax.axis("off")
        ax.set_title(f"{kind.title()} ({pos})", fontsize=20, weight="bold", pad=8)
        for center, color, fill in [(35, train_color, "#eaf1f7"), (65, val_color, "#fff0ed")]:
            ax.add_patch(Ellipse((center, 47), 67, 75, facecolor=fill, edgecolor=color, linewidth=2, alpha=0.64))
        ax.text(23, 89, "Train-derived coverage", ha="center", fontsize=11, color=train_color)
        ax.text(78, 89, "Val-derived coverage", ha="center", fontsize=11, color=val_color)
        for x, region, split, color in [(18, "A", "train", train_color), (83, "C", "validation", val_color)]:
            n = counts[(kind, split, region)]
            ax.scatter([x], [59], s=65, c=color, edgecolors="white", zorder=4)
            ax.text(x, 49, f"{region} | {'Train' if split == 'train' else 'Val'}", ha="center", fontsize=16, weight="bold", color=color)
            ax.text(x, 37, f"{n:,} {units}\n{100 * n / panel_totals[split]:.2f}%", ha="center", fontsize=13, linespacing=1.6)
            detail = f">=1 uncovered\n{kind} lemma" if unit == "query" else "No identical lemma\nor shared synset"
            ax.text(x, 20, detail, ha="center", fontsize=10, color="#475569")
        ax.text(50, 69, "B | Covered", ha="center", fontsize=16, weight="bold", color="#334155")
        for y, split, color, label in [(53, "train", train_color, "Train"), (36, "validation", val_color, "Val")]:
            n = counts[(kind, split, "B")]
            ax.scatter([39], [y], s=60, c=color, edgecolors="white", zorder=4)
            ax.text(42, y, f"{label}: {n:,}\n({100 * n / panel_totals[split]:.2f}%)", va="center", fontsize=11, color=color, linespacing=1.5)
        ax.annotate("Interest", xy=(83, 63), xytext=(95, 76), ha="center", fontsize=11, color="#a44109",
                    arrowprops={"arrowstyle": "->", "color": "#a44109", "lw": 1.5})
        empty = counts if empty_counts is None else empty_counts
        empty_label = "EMPTY" if unit == "query" else "EMPTY queries"
        ax.text(50, 3, f"{empty_label}: Train {empty[(kind, 'train', 'EMPTY')]:,} | Val {empty[(kind, 'validation', 'EMPTY')]:,}",
                ha="center", fontsize=11, bbox={"boxstyle": "round,pad=0.5", "fc": "#f1f5f9", "ec": "#cbd5e1"})
        pool_label = "Observed mappings" if unit == "mapping" else "Observed lemmas"
        ax.text(50, -6, f"{pool_label}: Train {len(pools[kind]['train']):,} | Val {len(pools[kind]['validation']):,}",
                ha="center", fontsize=10, color="#64748b")
        if unclassified_counts is not None:
            ax.text(50, -11, f"UNCLASSIFIED source occurrences: Train {unclassified_counts[(kind, 'train', 'UNCLASSIFIED')]:,} | Val {unclassified_counts[(kind, 'validation', 'UNCLASSIFIED')]:,}",
                    ha="center", fontsize=9, color="#64748b")
    denominator_note = (f"Percent denominators: all {totals['train']:,} train / {totals['validation']:,} val queries, including EMPTY."
                        if unit == "query" else
                        "Counts/percentages: distinct directed mapping rows per panel/source split. B can have multiple targets per source."
                        if unit == "mapping" else
                        "Counts/percentages: distinct source lemmas in each panel/split. EMPTY counts are queries.")
    fig.text(0.5, 0.07, denominator_note,
             ha="center", fontsize=11, color="#334155")
    fig.text(0.5, 0.039, "Word senses are not fully reviewed. Ellipse areas are schematic. EMPTY = no extracted terms.",
             ha="center", fontsize=10, color="#64748b")
    fig.text(0.5, 0.018, "Dictionary-only scope: Val terms outside supplied Train Object/Action vocabularies are excluded.",
             ha="center", fontsize=10, color="#64748b")
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.035, right=0.965, wspace=0.08)
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(out / f"object_action_regions.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    import csv

    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=workspace / "rewriting/val_domain_analysis/dictionary_only")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wordnet-dir", type=Path, default=workspace / "rewriting/nlp_resources")
    args = parser.parse_args()
    self_check()
    out = args.output_dir or args.input_dir / "object_action_regions"
    out.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((args.input_dir / "summary.json").read_text())
    dictionaries, vocabularies = dictionary_sources(args.input_dir)
    for name, item in source_summary["inputs"].items():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Extraction input has changed: {name}; refresh the upstream extraction first")
    sources = {"train": args.input_dir / "tvrtrain_domain_analysis.csv",
               "validation": args.input_dir / "tvrval_action_object_analysis.csv"}
    rows = {split: read_csv(path) for split, path in sources.items()}
    totals = {split: len(values) for split, values in rows.items()}
    for split, values in rows.items():
        caption_name = "tvrtrain.caption.txt" if split == "train" else "tvrval.caption.txt"
        originals = read_captions(Path(source_summary["inputs"][caption_name]["path"]))
        if [(r["caption_id"], r["caption"]) for r in values] != [(r["caption_id"], r["caption"]) for r in originals]:
            raise ValueError(f"Extraction CSV does not match original caption order/content: {split}")
        for row in values:
            for kind in ["object", "action"]:
                validate_dictionary_terms(json.loads(row[f"{kind}s_raw"]), kind, split, row["caption_id"], dictionaries[kind], vocabularies[kind])
                row[f"{kind}_terms"] = sorted(set(json.loads(row[f"{kind}_lemmas"])))
        print(f"Validated {split}: {len(values):,} original captions", flush=True)

    wn = load_wordnet(args.wordnet_dir)
    assert {s.name() for s in wn.synsets("bicycle", pos=wn.NOUN)} & {s.name() for s in wn.synsets("bike", pos=wn.NOUN)}
    assert not ({s.name() for s in wn.synsets("cup", pos=wn.NOUN)} & {s.name() for s in wn.synsets("cup", pos=wn.VERB)})
    assert distinct_expression(wn, "bicycle", "bike", wn.NOUN)
    assert not distinct_expression(wn, "walk", "walking", wn.VERB)
    assert not distinct_expression(wn, "color", "colour", wn.NOUN)
    assert not distinct_expression(wn, "cup", "cup", wn.NOUN)
    pools, term_info, term_rows = {}, {}, []
    captions = {split: {row["caption_id"]: row["caption"] for row in values} for split, values in rows.items()}
    for kind, pos in [("object", wn.NOUN), ("action", wn.VERB)]:
        frequencies, examples = {}, {}
        for split, values in rows.items():
            frequencies[split] = Counter(term for row in values for term in row[f"{kind}_terms"])
            examples[split] = defaultdict(list)
            for row in values:
                for term in row[f"{kind}_terms"]:
                    if len(examples[split][term]) < 3:
                        examples[split][term].append(row["caption_id"])
        pools[kind] = {split: set(freq) for split, freq in frequencies.items()}
        senses = {term: {s.name() for s in wn.synsets(term.replace(" ", "_"), pos=pos)}
                  for term in sorted(set.union(*pools[kind].values()))}
        index = {}
        for split, terms in pools[kind].items():
            index[split] = defaultdict(set)
            for term in sorted(terms):
                for sense in sorted(senses[term]):
                    index[split][sense].add(term)
        for split, terms in pools[kind].items():
            other = "validation" if split == "train" else "train"
            for term in sorted(terms):
                matches = {term} if term in pools[kind][other] else {
                    target for sense in senses[term] for target in index[other][sense]}
                status = "DIRECT_LEMMA" if term in pools[kind][other] else (
                    "WORDNET_CANDIDATE" if matches else "UNCOVERED")
                evidence = [{"other_lemma": target, "shared_synsets": sorted(senses[term] & senses[target]),
                             "other_example_caption_ids": examples[other][target]} for target in sorted(matches)]
                options = synonym_options(wn, term, pos, senses, index[other], senses,
                                          examples[other], captions[other])
                info = {"status": status, "has_wordnet_sense": bool(senses[term]), "synonyms": options}
                term_info[(kind, split, term)] = info
                term_rows.append({"aspect": kind, "split": split, "other_split": other, "lemma": term,
                                  "term_region": query_region(split, [term], [] if matches else [term]),
                                  "coverage_status": status, "source_caption_count": frequencies[split][term],
                                  "source_example_caption_ids": encoded(examples[split][term]),
                                  "source_synsets": encoded(sorted(senses[term])), "no_wordnet_sense": int(not senses[term]),
                                  "matched_other_lemmas": encoded(sorted(matches)), "candidate_evidence": encoded(evidence),
                                  "requires_sense_review": int(status == "WORDNET_CANDIDATE")})
        print(f"Indexed {kind}: train {len(pools[kind]['train']):,} / val {len(pools[kind]['validation']):,} lemmas", flush=True)
    write_csv(out / "term_coverage.csv", list(term_rows[0]), term_rows)

    fields = ["aspect", "split", "caption_id", "video_id", "enc", "caption", "region", "lemma_only_region",
              "coverage_status", "coverage_fraction", "term_count", "direct_lemmas", "wordnet_candidate_lemmas",
              "uncovered_lemmas", "no_wordnet_sense_lemmas", "original_terms_raw", "invalid_pos_raw",
              "requires_sense_review", "extraction_source", "upstream_needs_manual_review"]
    synonym_fields = ["synonym_analysis_status", "wordnet_synonym_candidates", "other_split_synonym_candidates",
                      "terms_without_wordnet_synonym", "terms_without_other_split_synonym",
                      "has_any_wordnet_synonym", "all_terms_have_wordnet_synonym",
                      "has_any_other_split_synonym", "all_terms_have_other_split_synonym", "requires_synonym_review"]
    fields += synonym_fields
    counts, coverage_counts, pair_counts = Counter(), Counter(), Counter()
    synonym_counts, b_term_counts, b_term_examples = Counter(), Counter(), defaultdict(list)
    pair_rows = []
    with ExitStack() as stack:
        writers = {}
        for name in ["query_regions"] + [f"{kind}_{region}" for kind in ["object", "action"] for region in ["A", "B", "C", "EMPTY"]]:
            handle = stack.enter_context((out / f"{name}.csv").open("w", encoding="utf-8-sig", newline=""))
            writers[name] = csv.DictWriter(handle, fieldnames=fields)
            writers[name].writeheader()
        for split, values in rows.items():
            for row in values:
                pair = {key: row[key] for key in ["caption_id", "video_id", "enc"]}
                pair["split"] = split
                for kind in ["object", "action"]:
                    terms = row[f"{kind}_terms"]
                    direct, candidates, uncovered, no_sense = [], [], [], []
                    for term in terms:
                        info = term_info[(kind, split, term)]
                        {"DIRECT_LEMMA": direct, "WORDNET_CANDIDATE": candidates, "UNCOVERED": uncovered}[info["status"]].append(term)
                        if not info["has_wordnet_sense"]:
                            no_sense.append(term)
                    region = query_region(split, terms, uncovered)
                    status = "EMPTY" if not terms else "FULL" if not uncovered else "PARTIAL" if direct or candidates else "NONE"
                    result = {key: row[key] for key in ["caption_id", "video_id", "enc", "caption"]}
                    result.update({"aspect": kind, "split": split, "region": region,
                                   "lemma_only_region": query_region(split, terms, candidates + uncovered),
                                   "coverage_status": status, "coverage_fraction": (len(direct) + len(candidates)) / len(terms) if terms else "",
                                   "term_count": len(terms), "direct_lemmas": encoded(direct),
                                   "wordnet_candidate_lemmas": encoded(candidates), "uncovered_lemmas": encoded(uncovered),
                                   "no_wordnet_sense_lemmas": encoded(no_sense), "original_terms_raw": row[f"{kind}s_raw"],
                                   "invalid_pos_raw": row[f"invalid_{kind}_pos"], "requires_sense_review": int(bool(candidates)),
                                   "extraction_source": row.get(f"{kind}_source", "supplied_train_dictionary"),
                                   "upstream_needs_manual_review": row.get("needs_manual_review", "")})
                    result.update({key: "" for key in synonym_fields})
                    result["synonym_analysis_status"] = "NOT_EVALUATED_NON_B"
                    if region == "B":
                        wordnet_candidates = {term: term_info[(kind, split, term)]["synonyms"]["wordnet"] for term in terms}
                        other_candidates = {term: term_info[(kind, split, term)]["synonyms"]["other_split"] for term in terms}
                        flags = {"has_any_wordnet_synonym": int(any(wordnet_candidates.values())),
                                 "all_terms_have_wordnet_synonym": int(all(wordnet_candidates.values())),
                                 "has_any_other_split_synonym": int(any(other_candidates.values())),
                                 "all_terms_have_other_split_synonym": int(all(other_candidates.values()))}
                        result.update(flags)
                        result.update({"synonym_analysis_status": "LEXICAL_CANDIDATES_REQUIRE_CONTEXT_REVIEW",
                                       "wordnet_synonym_candidates": encoded(wordnet_candidates),
                                       "other_split_synonym_candidates": encoded(other_candidates),
                                       "terms_without_wordnet_synonym": encoded([term for term in terms if not wordnet_candidates[term]]),
                                       "terms_without_other_split_synonym": encoded([term for term in terms if not other_candidates[term]]),
                                       "requires_synonym_review": flags["has_any_wordnet_synonym"] or flags["has_any_other_split_synonym"]})
                        for key, value in flags.items():
                            synonym_counts[(kind, split, key)] += value
                        for term in terms:
                            key = kind, split, term
                            b_term_counts[key] += 1
                            if len(b_term_examples[key]) < 3:
                                b_term_examples[key].append(row["caption_id"])
                    assert len(direct) + len(candidates) + len(uncovered) == len(terms)
                    assert not terms or (region == "B") == (not uncovered)
                    writers["query_regions"].writerow(result)
                    writers[f"{kind}_{region}"].writerow(result)
                    counts[(kind, split, region)] += 1
                    coverage_counts[(kind, split, status)] += 1
                    pair[f"{kind}_region"] = region
                pair_rows.append(pair)
                pair_counts[(split, pair["object_region"], pair["action_region"])] += 1
    write_csv(out / "query_region_pairs.csv", list(pair_rows[0]), pair_rows)

    for kind in ["object", "action"]:
        synonym_terms = []
        for (aspect, split, term), n in sorted(b_term_counts.items()):
            if aspect != kind:
                continue
            info = term_info[(kind, split, term)]
            options = info["synonyms"]
            synonym_terms.append({"aspect": kind, "split": split,
                                  "other_split": "validation" if split == "train" else "train",
                                  "source_lemma": term, "B_query_count": n,
                                  "source_B_example_caption_ids": encoded(b_term_examples[(kind, split, term)]),
                                  "has_wordnet_entry": int(info["has_wordnet_sense"]),
                                  "has_wordnet_synonym": int(bool(options["wordnet"])),
                                  "has_other_split_synonym": int(bool(options["other_split"])),
                                  "wordnet_synonym_count": len(options["wordnet"]),
                                  "other_split_synonym_count": len(options["other_split"]),
                                  "wordnet_synonyms": encoded(options["wordnet"]),
                                  "other_split_synonyms": encoded(options["other_split"]),
                                  "wordnet_synonyms_by_sense": encoded(options["by_sense"]),
                                  "other_split_evidence": encoded(options["evidence"]),
                                  "requires_synonym_review": int(bool(options["wordnet"] or options["other_split"]))})
        write_csv(out / f"{kind}_B_synonym_terms.csv", list(synonym_terms[0]), synonym_terms)
    synonym_summary = []
    for kind in ["object", "action"]:
        for split in rows:
            denominator = counts[(kind, split, "B")]
            for criterion in ["has_any_wordnet_synonym", "all_terms_have_wordnet_synonym",
                              "has_any_other_split_synonym", "all_terms_have_other_split_synonym"]:
                n = synonym_counts[(kind, split, criterion)]
                synonym_summary.append({"aspect": kind, "split": split, "criterion": criterion,
                                        "B_query_count": n, "denominator_B_queries": denominator,
                                        "percentage_of_B": round(100 * n / denominator, 6)})
    write_csv(out / "B_synonym_summary.csv", list(synonym_summary[0]), synonym_summary)

    summary_rows = []
    for kind in ["object", "action"]:
        for split in rows:
            for region in ["A", "B", "C", "EMPTY"]:
                n = counts[(kind, split, region)]
                summary_rows.append({"aspect": kind, "split": split, "region": region, "query_count": n,
                                     "denominator_all_split_queries": totals[split], "percentage": round(100 * n / totals[split], 6),
                                     "csv_file": f"{kind}_{region}.csv"})
            assert sum(counts[(kind, split, r)] for r in ["A", "B", "C", "EMPTY"]) == totals[split]
    assert counts[("object", "validation", "A")] == counts[("action", "validation", "A")] == 0
    assert counts[("object", "train", "C")] == counts[("action", "train", "C")] == 0
    write_csv(out / "region_summary.csv", list(summary_rows[0]), summary_rows)
    plot_regions(out, counts, totals, pools)
    metadata = {"source": "Original TVR captions; existing object/action extraction reused without changes",
                "totals": totals, "wordnet_version": wn.get_version(), "randomness": "None; original row order and sorted lexical evidence",
                "coverage_rule": "Same normalized lemma OR at least one shared same-POS WordNet synset; all senses considered",
                "query_rule": "All extracted aspect lemmas must be covered for B; any uncovered lemma => train A / validation C; no terms => EMPTY",
                "wordnet_scope": "Noun synsets for objects; verb synsets for actions; no hypernyms, embeddings, or context sense disambiguation",
                "direct_lemma_rule": "Identical observed lemma is covered even if no WordNet sense exists",
                "synonym_rule": "B query terms: different expressions in same-POS WordNet synsets; all senses considered; morphology and spelling variants excluded",
                "synonym_other_split_rule": "Different observed lemma in the opposite original split sharing a same-POS synset; source itself/morphology/spelling variants excluded",
                "synonym_inflection": "Candidates are lemma/base expressions, not automatically inflected replacements",
                "B_synonym_summary": synonym_summary,
                "limitations": source_summary["method"]["limitations"] + [
                    "WordNet coverage is a lexical candidate/upper bound on context-valid coverage, not proven semantic equivalence.",
                    "Object means extracted NN/NNS terms, including abstract/nonvisual nouns; Action means extracted verb terms, including nonvisual verbs.",
                    "No visual grounding, phrasal-verb recovery, or action-object/temporal relation validation is performed.",
                    "C is unsupported under this rule, not proven semantic OOD or model ignorance.",
                    "Object and action C query lists overlap; do not add their counts to obtain unique C queries.",
                    "The supplied diagram's 1.57% is not used; all percentages are computed from these source CSVs."],
                "upstream_method": source_summary["method"], "regions": summary_rows,
                "pair_counts": [{"split": s, "object_region": o, "action_region": a, "query_count": n}
                                for (s, o, a), n in sorted(pair_counts.items())],
                "inputs": {path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
                           for path in list(sources.values()) + [args.input_dir / "summary.json", args.wordnet_dir / "corpora/wordnet.zip"]},
                "script_sha256": sha256(Path(__file__))}
    (out / "methodology.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text("""# TVR Object / Action 영역 분류

원본 train 87,174개와 validation 10,895개 쿼리의 기존 추출 결과를 재사용합니다.
Object는 NN/NNS 기반 명사, Action은 동사입니다. 시각적으로 검증된 객체/동작만을 뜻하지 않습니다.

## 다이어그램과 영역

점의 색은 데이터 출처, 큰 원은 해당 split에서 만든 lemma + WordNet 후보 커버 범위입니다.
반대 split에 동일 정규화 lemma가 있거나, 같은 품사에서 하나 이상의 WordNet synset을 공유하면 커버 후보입니다.
WordNet은 모든 의미를 고려합니다. 문맥상 의미 일치가 검증된 결과가 아닙니다.

- A: train 쿼리에서 해당 aspect 용어가 하나 이상 반대 split에 미커버.
- B: 해당 aspect 용어가 하나 이상 있고 모두 반대 split에 커버됨. Train/validation은 split 열로 구분.
- C: validation 쿼리에서 해당 aspect 용어가 하나 이상 train에 미커버. 관심 대상.
- EMPTY: 해당 aspect 추출 용어 없음. B로 취급하지 않음.

일부 용어만 미커버인 쿼리도 A/C입니다. coverage_status의 PARTIAL/NONE으로 구분합니다.
no_wordnet_sense_lemmas는 별도 진단이며 A/C와 같지 않습니다. 동일 lemma가 반대 split에 있으면 WordNet에 없어도 커버됩니다.
Object C와 Action C는 서로 겹칠 수 있습니다. Query 전체의 새로움, 검색 난도 또는 불확실성을 직접 증명하지 않습니다.

## 파일

- object_action_regions.png / .svg / .pdf: Object와 Action을 분리한 개념 다이어그램. 원의 면적은 비율을 의미하지 않음.
- object_A/B/C/EMPTY.csv, action_A/B/C/EMPTY.csv: 각 aspect와 영역의 실제 쿼리. 원문/ID 보존.
- query_regions.csv: 전체 결과. 한 행은 쿼리 × aspect이며 총 196,138행.
- query_region_pairs.csv: 한 쿼리의 Object/Action 영역을 함께 기록. CSV 간 중복 식별에 사용.
- term_coverage.csv: split별 lemma 단위 판정, 공유 synset, 반대 split의 근거 쿼리 ID.
- object_B_synonym_terms.csv / action_B_synonym_terms.csv: 실제 B 쿼리의 용어별 유의어 여부, 후보, synset·뜻·반대 split 근거 캡션.
- B_synonym_summary.csv: B 쿼리에서 유의어 후보가 있는 비율. 하나 이상의 용어/모든 용어를 구분.
- region_summary.csv: 각 영역의 쿼리 수와 전체 해당 split 쿼리 수 기준 비율. EMPTY 포함 분모.
- methodology.json: 입력 해시, WordNet 버전, 추출 방법과 한계, 교차 영역 집계.

CSV는 Excel 호환 UTF-8 BOM입니다. 배열/사전 셀은 JSON입니다.
lemma_only_region은 WordNet 확장 없이 동일 lemma만 비교한 영역입니다.
requires_sense_review=1은 WordNet 후보에 의존하는 쿼리/용어입니다.
원본 그림의 1.57%는 검증된 기준이 없으므로 재사용하지 않았습니다.

## B 유의어 후보

B는 커버리지 분류입니다. 동일 lemma로 커버되거나 WordNet에 원본만 있는 경우도 있으므로 교체 가능성을 보장하지 않습니다.
B CSV의 wordnet_synonym_candidates는 용어별 전체 WordNet 유의어 후보입니다.
other_split_synonym_candidates는 반대 split에서 관측된 다른 lemma 중 동일 synset을 공유하는 후보입니다.
원본 자체·단순 굴절·UK/US 철자 변형은 유의어 후보에서 제외합니다. 동사·명사 의미는 섞지 않습니다.
has_any_*는 해당 aspect 용어 중 하나 이상에 후보가 있음, all_terms_have_*는 모든 용어에 후보가 있음을 뜻합니다.
용어별 CSV의 has_wordnet_entry / has_wordnet_synonym / has_other_split_synonym으로 등재·다른 유의어·반대 split 후보를 따로 구분합니다.
wordnet_synonyms_by_sense는 의미별 후보와 뜻, other_split_evidence는 실제 관측 캡션을 제공합니다.
NOT_EVALUATED_NON_B 행의 후보 열은 비어 있으며 유의어가 없다는 뜻이 아닙니다.
후보는 기본형 표현입니다. 문맥·시각적 동작·구동사·문법·동작 시점은 검증하지 않았으므로 자동 치환하지 않습니다.
실제 쿼리 재작성에는 객체 강조·대명사 해소·가시적 상태·감정의 동작화·14–20단어/30 미만 subword·시점별 동작 보존 규칙을 적용해야 합니다.

기존 train 사전과 validation spaCy POS 추출은 추출 방식이 다릅니다.
다의어, 추상 명사/비시각적 동사, POS·lemma 오류, 구동사 손실이 결과에 영향을 줄 수 있습니다.
rewriting이나 모델/데이터 split 변경은 수행하지 않습니다.

## 재현

```bash
/home/viglab2/miniconda3/envs/holmes/bin/python rewriting/build_object_action_regions.py
```

무작위 샘플링/모델 초기화 없이 원본 순서와 정렬된 근거를 사용합니다.
스크립트는 입력 해시·원문 일치·영역 분할 합계와 핵심 빈 용어/부분 커버 사례를 검증합니다.
""", encoding="utf-8")
    archive = out.parent / "object_action_regions.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            if path.is_file():
                bundle.write(path, arcname=f"object_action_regions/{path.name}")
        bundle.write(Path(__file__), arcname="object_action_regions/build_object_action_regions.py")
    print(json.dumps({"regions": summary_rows, "B_synonym_summary": synonym_summary,
                      "validation_both_C": pair_counts[("validation", "C", "C")],
                      "archive": str(archive)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
