"""Conservative, local text checks for one-position Object/Action rewrites.

``eligible`` means suitable for a provisional retrieval experiment, not verified
visual or temporal equivalence. No LLM, downloads, or hard length filter is used.
"""

import re
from functools import lru_cache
from pathlib import Path

from select_diagram_synonyms import FAMILIES, NONVISUAL, syntax_status
from suggest_val_replacements import load_wordnet


@lru_cache(maxsize=1)
def _wordnet():
    return load_wordnet(Path(__file__).resolve().parent / "nlp_resources")


@lru_cache(maxsize=16384)
def _senses(lemma, kind):
    return tuple(_wordnet().synsets(lemma.replace(" ", "_"),
                                  pos="n" if kind == "object" else "v"))


@lru_cache(maxsize=16384)
def _normalized_word(word):
    wn = _wordnet()
    return {word} | {base for pos in ("n", "v", "a", "r")
                     for base in [wn.morphy(word, pos)] if base}


@lru_cache(maxsize=32768)
def _sense_words(sense):
    # ponytail: gloss overlap is a conservative ambiguity signal, not certified
    # WSD; unresolved senses stay pending for manual/segment-level verification.
    text = " ".join([sense.definition(), *sense.examples(), *sense.lemma_names()])
    return set().union(*(_normalized_word(word) for word in
                        re.findall(r"[a-z]+", text.lower())))


def _context_senses(token, senses):
    context = {word.lemma_.lower() for word in token.sent
               if word.i != token.i and not word.is_stop and not word.is_punct}
    scores = {sense: len(context & _sense_words(sense)) for sense in senses}
    highest = max(scores.values(), default=0)
    return {sense for sense, score in scores.items() if score == highest}, highest


def _span(doc, start, end):
    return doc.char_span(start, end, alignment_mode="strict")


def _length_metadata(text, tokenizer):
    words = len(text.split())
    result = {"word_count": words, "within_14_20_words": 14 <= words <= 20,
              "subword_count_including_special_tokens": "",
              "under_30_subword_tokens": "", "encoder_max_length": 30,
              "encoder_truncated": "unknown", "length_filter_applied": False}
    if tokenizer is not None:
        count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        result.update(subword_count_including_special_tokens=count,
                      under_30_subword_tokens=count < 30,
                      encoder_truncated=count > 30)
    return result


def validate_candidate(source_doc, candidate_doc, witness_doc, *, term_type,
                       source_span_start, source_span_end, source_expression,
                       val_expression, source_pos="", val_pos="",
                       timestamp_status="", tokenizer=None):
    """Validate already parsed source/candidate/Val witness spaCy documents.

    The supplied candidate must be the exact single-span replacement with the
    already selected rank-one Val surface. Witness text confirms that surface
    and aspect only; it does not resolve the source event's word sense.
    """
    if term_type not in {"object", "action"}:
        raise ValueError("term_type must be object or action")
    start, end = int(source_span_start), int(source_span_end)
    original, candidate = source_doc.text, candidate_doc.text
    result = {"text_validation_status": "review_required", "validation_status": "review_required",
              "validation_reason": "", "lexical_status": "unchecked",
              "source_context_pos": "", "source_context_tag": "", "source_context_lemma": "",
              "target_context_pos": "", "target_context_tag": "", "target_context_lemma": "",
              "shared_wordnet_synsets": "", "source_sense_context_score": "",
              "target_sense_context_score": "", "visual_grounding_status": "not_verified",
              "temporal_status": "timestamp_available" if timestamp_status.startswith("resolved") else "unresolved",
              "event_preservation_status": "not_verified",
              "semantic_preservation_certified": False,
              "semantic_check_basis": "POS/morphology/dependencies/WordNet/local_context",
              **{"source_" + key: value for key, value in _length_metadata(original, tokenizer).items()},
              **{"candidate_" + key: value for key, value in _length_metadata(candidate, tokenizer).items()}}

    def finish(status, reason, lexical_status=None):
        result.update(text_validation_status=status, validation_status=status,
                      validation_reason=reason)
        if lexical_status is not None:
            result["lexical_status"] = lexical_status
        return result

    if not (0 <= start < end <= len(original)) or original[start:end] != source_expression:
        return finish("rejected", "source_span_mismatch")
    if not val_expression or candidate != original[:start] + val_expression + original[end:]:
        return finish("rejected", "not_exact_single_span_val_replacement")
    if candidate == original:
        return finish("rejected", "unchanged_caption")
    source_span = _span(source_doc, start, end)
    target_span = _span(candidate_doc, start, start + len(val_expression))
    if source_span is None or target_span is None:
        return finish("rejected", "replacement_not_aligned_to_tokens")
    if len(source_span) != 1 or len(target_span) != 1:
        return finish("review_required", "multiword_context_requires_review")
    src, dst = source_span[0], target_span[0]
    result.update(source_context_pos=src.pos_, source_context_tag=src.tag_,
                  source_context_lemma=src.lemma_, target_context_pos=dst.pos_,
                  target_context_tag=dst.tag_, target_context_lemma=dst.lemma_)
    expected = "NOUN" if term_type == "object" else "VERB"
    if src.pos_ != expected or dst.pos_ != expected:
        return finish("rejected", "actual_context_pos_mismatch")
    if source_pos and src.tag_ != source_pos:
        return finish("rejected", "source_dictionary_pos_mismatch")
    if src.tag_ != dst.tag_ or src.morph != dst.morph:
        return finish("rejected", "tense_or_number_not_preserved")
    witnesses = [token for token in witness_doc if token.text == val_expression]
    if not witnesses:
        return finish("rejected", "val_surface_absent_from_witness")
    if not any(token.pos_ == expected and (not val_pos or token.tag_ == val_pos)
               and token.lemma_.lower() == dst.lemma_.lower() for token in witnesses):
        return finish("rejected", "val_witness_context_pos_or_lemma_mismatch")
    if term_type == "action":
        if any(child.dep_ == "prt" for token in (src, dst) for child in token.children):
            return finish("rejected", "phrasal_verb_component_preserved")
        if src.lemma_.lower() in NONVISUAL or dst.lemma_.lower() in NONVISUAL:
            return finish("review_required", "nonvisual_action_requires_review")
        for child in src.children:
            if child.dep_ == "prep" and _senses(src.lemma_ + "_" + child.lower_, "action"):
                return finish("review_required", "prepositional_verb_requires_review")
    if len(source_doc) != len(candidate_doc) or src.i != dst.i:
        return finish("review_required", "tokenization_changed_outside_replacement")
    # Compare the full dependency frame, including unchanged subjects, negation,
    # objects and temporal modifiers. Parser disagreement is kept for review.
    for left, right in zip(source_doc, candidate_doc):
        if (left.dep_, left.head.i) != (right.dep_, right.head.i):
            return finish("review_required", "dependency_frame_changed")
        if left.i != src.i and (left.text, left.lemma_, left.pos_, left.tag_) != (
                right.text, right.lemma_, right.pos_, right.tag_):
            return finish("review_required", "unchanged_token_analysis_changed")
    for token in (src, dst):
        reason = syntax_status([token], term_type, token.lemma_.lower())
        if reason:
            return finish("review_required", reason)
    source_senses = _senses(src.lemma_.lower(), term_type)
    target_senses = _senses(dst.lemma_.lower(), term_type)
    shared = set(source_senses) & set(target_senses)
    result["shared_wordnet_synsets"] = " | ".join(sorted(sense.name() for sense in shared))
    if not shared:
        return finish("rejected", "no_shared_same_aspect_wordnet_sense", "not_synonymous")
    best_source, source_score = _context_senses(src, source_senses)
    best_target, target_score = _context_senses(dst, target_senses)
    result.update(source_sense_context_score=source_score, target_sense_context_score=target_score)
    if (source_score and not best_source & shared) or (target_score and not best_target & shared):
        return finish("review_required", "shared_sense_not_supported_by_context", "ambiguous")
    conventional = any({src.lemma_.lower(), dst.lemma_.lower()} <= set(family.split())
                       for family in FAMILIES[term_type])
    if not conventional and (not source_score or not target_score or
                             not best_source <= shared or not best_target <= shared):
        return finish("review_required", "wordnet_context_sense_ambiguous", "ambiguous")
    return finish("eligible", "conservative_text_checks_passed_video_unverified",
                  "conventional_synonym" if conventional else "context_supported_shared_sense")


def _initial_sound_hint(word):
    """Return a spelling heuristic, never a certified pronunciation decision."""
    low = word.lower()
    if not re.fullmatch(r"[a-z]+", low) or word.isupper() or low.startswith(("herb", "x")):
        return "unknown"
    if low.startswith(("honest", "honor", "honour", "hour", "heir")):
        return "vowel_hint"
    if low.startswith(("university", "unicorn", "uniform", "union", "unit", "usual",
                       "user", "use", "urine", "european")) or low in {"one", "once"}:
        return "consonant_hint"
    if low.startswith(("u", "eu")):
        return "unknown"
    return "vowel_hint" if low[0] in "aeio" else "consonant_hint"


def strict_post_review(source_doc, candidate_doc, *, source_span_start,
                       source_span_end, val_expression, term_type, validation):
    """Independently review initially eligible candidates without editing text.

    Pronunciation spelling hints and the existing conventional synonym families
    are conservative experimental gates. They do not certify semantic validity.
    Nonconventional candidates need *all* senses in both words to be shared.
    """
    result = dict(validation)
    result.update(initial_text_validation_status=validation["text_validation_status"],
                  initial_validation_reason=validation["validation_reason"],
                  strict_review_applied=False, strict_review_status="not_needed",
                  strict_review_policy="article_hint_and_conventional_or_all_senses_shared_v1",
                  strict_review_reason="", strict_article_status="unchecked",
                  strict_article_sound_hint="", strict_article_pronunciation_certified=False,
                  strict_wordnet_status="unchecked", strict_source_sense_count="",
                  strict_target_sense_count="", strict_shared_sense_count="",
                  semantic_preservation_certified=False)
    if validation["text_validation_status"] != "eligible":
        return result
    result["strict_review_applied"] = True
    start, end = int(source_span_start), int(source_span_end)
    src_span = _span(source_doc, start, end)
    dst_span = _span(candidate_doc, start, start + len(val_expression))
    reasons = []
    if src_span is None or dst_span is None or len(src_span) != 1 or len(dst_span) != 1:
        reasons.append("strict_review_span_requires_review")
    else:
        src, dst = src_span[0], dst_span[0]
        result["strict_article_status"] = "not_directly_before_replacement"
        for article in dst.children:
            if article.dep_ != "det" or article.lower_ not in {"a", "an"} or article.i >= dst.i:
                continue
            if any(not token.is_space and not token.is_punct
                   for token in candidate_doc[article.i + 1:dst.i]):
                # "a red car" -> "a red auto": the first sound remains "red".
                result["strict_article_status"] = "unchanged_intervening_word"
                continue
            hint = _initial_sound_hint(dst.text)
            result["strict_article_sound_hint"] = hint
            if hint == "unknown":
                result["strict_article_status"] = "pronunciation_requires_review"
                reasons.append("strict_article_pronunciation_uncertain")
            elif (article.lower_ == "a" and hint == "vowel_hint") or (
                    article.lower_ == "an" and hint == "consonant_hint"):
                result["strict_article_status"] = "spelling_hint_conflict_requires_review"
                reasons.append("strict_article_change_would_be_needed")
            else:
                result["strict_article_status"] = "no_spelling_hint_conflict"
        conventional = any({src.lemma_.lower(), dst.lemma_.lower()} <= set(family.split())
                           for family in FAMILIES[term_type])
        source_senses = set(_senses(src.lemma_.lower(), term_type))
        target_senses = set(_senses(dst.lemma_.lower(), term_type))
        shared = source_senses & target_senses
        result.update(strict_source_sense_count=len(source_senses),
                      strict_target_sense_count=len(target_senses), strict_shared_sense_count=len(shared))
        if conventional and shared:
            result["strict_wordnet_status"] = "conventional_synonym_family"
        elif shared and source_senses == shared and target_senses == shared:
            result["strict_wordnet_status"] = "all_senses_shared"
        else:
            result["strict_wordnet_status"] = "nonconventional_partial_sense_overlap"
            reasons.append("strict_nonconventional_sense_requires_review")
    result.update(strict_review_status="review_required" if reasons else "eligible_provisional",
                  strict_review_reason=" | ".join(reasons) if reasons else "strict_checks_passed_video_unverified")
    if reasons:
        result.update(text_validation_status="review_required", validation_status="review_required",
                      validation_reason=reasons[0])
    return result


def self_check():
    """Regression examples use only installed local spaCy and WordNet assets."""
    import spacy
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    text = "Phoebe picks up a drink from the bar."
    source_doc = nlp(text)

    def check(source, target, kind, witness, pos=""):
        start = text.index(source)
        return validate_candidate(source_doc, nlp(text[:start] + target + text[start + len(source):]),
                                  nlp(witness), term_type=kind, source_span_start=start,
                                  source_span_end=start + len(source), source_expression=source,
                                  val_expression=target, source_pos=pos, timestamp_status="resolved")

    drink = check("drink", "beverage", "object", "Ross holds a beverage.", "NN")
    assert drink["text_validation_status"] == "eligible", drink
    assert drink["source_word_count"] == 8 and not drink["source_length_filter_applied"]
    assert drink["visual_grounding_status"] == "not_verified"
    choices = check("picks", "choices", "object", "There are many choices.", "NNS")
    assert choices["text_validation_status"] == "rejected", choices
    blame = check("picks", "blame", "action", "They blame Ross.", "VBZ")
    assert blame["text_validation_status"] == "rejected", blame
    blames = check("picks", "blames", "action", "Ross blames Joey.", "VBZ")
    assert blames["validation_reason"] == "phrasal_verb_component_preserved", blames
    cake = check("bar", "cake", "object", "Ross holds a cake.", "NN")
    assert cake["text_validation_status"] != "eligible", cake
    original_drink = dict(drink)

    def review(text, source, target, initial=drink):
        start = text.index(source)
        return strict_post_review(nlp(text), nlp(text[:start] + target + text[start + len(source):]),
                                  source_span_start=start, source_span_end=start + len(source),
                                  val_expression=target, term_type="object", validation=initial)

    car = review("Ross drives a car.", "car", "auto")
    assert car["validation_reason"] == "strict_article_change_would_be_needed", car
    red_car = review("Ross drives a red car.", "car", "auto")
    assert red_car["text_validation_status"] == "eligible", red_car
    assert red_car["strict_article_status"] == "unchanged_intervening_word", red_car
    phone = review("Ross dials his phone.", "phone", "earpiece")
    assert phone["validation_reason"] == "strict_nonconventional_sense_requires_review", phone
    beverage = review(text, "drink", "beverage")
    assert beverage["text_validation_status"] == "eligible", beverage
    assert drink == original_drink and not beverage["semantic_preservation_certified"]
    print("PASS: context/morphology, length metadata, article and strict WordNet post-review checks")


if __name__ == "__main__":
    self_check()
