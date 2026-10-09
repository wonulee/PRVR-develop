"""Offline guards/context/resume check: python -m Rewriter.test_six_strategy."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .six_strategy import (RESUME_COMPATIBLE_SOURCES, Rejected, candidate_check,
                           ground_summary, prompt_json, rewrite_file, summary_check, validate_run_settings,
                           verdict_check)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + (["<s>", "</s>"] if add_special_tokens else [])


class Generator:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.tokenizer = self
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return "prompt"

    def __call__(self, prompt, **kwargs):
        value = next(self.responses)
        return [{"generated_text": value if isinstance(value, str) else json.dumps(value)}]


class BatchGenerator:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.tokenizer = self
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, prompts, **kwargs):
        outputs = next(self.responses)
        count = len(prompts) if isinstance(prompts, list) else 1
        assert len(outputs) == count
        self.calls.append((count, kwargs.get("batch_size")))
        rows = [[{"generated_text": json.dumps(value)}] for value in outputs]
        return rows if isinstance(prompts, list) else rows[0]


def rejects(function, reason):
    try:
        function()
    except Rejected as exc:
        assert str(exc) == reason, (str(exc), reason)
    else:
        raise AssertionError(reason)


def check():
    payload_with_whitespace = {"TARGET": {"query": "House talks about a case." + "\t " * 3000},
                               "grounding": [{"quote": "House talks about a case.\t "}],
                               "internal_whitespace": "a\t b", "flag": False}
    sanitized = json.loads(prompt_json(payload_with_whitespace))
    assert sanitized["TARGET"]["query"] == "House talks about a case."
    assert sanitized["grounding"][0]["quote"] == "House talks about a case."
    assert sanitized["internal_whitespace"] == "a\t b" and sanitized["flag"] is False
    assert payload_with_whitespace["TARGET"]["query"].endswith("\t ")
    tokenizer = Tokenizer()
    original = "Penny walks without shoes across the living room and places the cup on the table beside Sheldon."
    candidate = "Penny walks barefoot across the living room and places the cup on the table beside Sheldon."
    captions = {"v#enc#0": original}
    payload = {"query": candidate, "grounding": []}
    assert candidate_check(original, payload, tokenizer, captions, 29, 16, 22) == candidate
    rejects(lambda: candidate_check(original, payload, tokenizer, captions, 17, 16, 22), "token_budget")
    rejects(lambda: candidate_check(original, payload, tokenizer, captions, 29, 19, 22), "word_budget")
    paraphrase_original = "Penny walks without shoes across the living room, then places the cup on the table beside Sheldon."
    paraphrase = "Penny walks barefoot across the living room, then puts the cup on the table beside Sheldon."
    assert candidate_check(paraphrase_original, {"query": paraphrase, "grounding": [
        {"text": paraphrase_original, "caption_id": "v#enc#0", "quote": paraphrase_original}]},
        tokenizer, {"v#enc#0": paraphrase_original}, 29, 16, 22,
        strict_words=False) == paraphrase
    short_original = "He walks without shoes."
    short_payload = {"query": "He walks barefoot.", "grounding": []}
    assert candidate_check(short_original, short_payload, tokenizer, {}, 29, 16, 22,
                           strict_words=False) == "He walks barefoot."
    rejects(lambda: candidate_check(short_original, short_payload, tokenizer, {}, 29, 16, 22,
                                    strict_words=True), "word_budget")
    rejects(lambda: candidate_check(original, {"query": "word " * 23, "grounding": []},
                                    tokenizer, captions, 29, 16, 22, strict_words=False), "word_budget")
    summary = {"characters": [{"text": "Penny", "caption_id": "v#enc#0", "quote": "Penny walks"}],
               "objects": [], "settings": []}
    assert summary_check(summary, captions) == summary
    fabricated = {"characters": [], "objects": [{"text": "laptop", "caption_id": "v#enc#0",
                                                  "quote": "Penny walks"}], "settings": []}
    rejects(lambda: summary_check(fabricated, captions), "unsupported_summary_span")
    malformed = {"characters": [{"text": "Penny", "caption_id": "unknown", "quote": "invented quote"}],
                 "objects": [{"text": "laptop", "caption_id": "v#enc#0", "quote": "Penny walks"}],
                 "settings": "invalid list"}
    grounded, repairs = ground_summary(malformed, captions)
    assert grounded["characters"] == [{"text": "Penny", "caption_id": "v#enc#0", "quote": original}]
    assert grounded["objects"] == grounded["settings"] == []
    assert any(note.get("reason") == "unsupported_text" for note in repairs)
    assert ground_summary("invalid JSON response", captions)[0] == {"characters": [], "objects": [], "settings": []}
    verdict = dict(target_event_preserved=True, participant_roles_preserved=True,
                   negation_equivalent=True, temporal_relationships_preserved=True,
                   sibling_event_imported=False, unsupported_visual_details=[], reason="faithful")
    verdict_check(verdict)
    rejects(lambda: verdict_check(dict(verdict, target_event_preserved=False)), "target_event_preserved")
    rejects(lambda: verdict_check(dict(verdict, unsupported_visual_details=["frowns"])), "unsupported_visual_details")
    rejects(lambda: verdict_check(dict(verdict, participant_roles_preserved=1)), "invalid_verification_schema")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "captions.txt"
        second = "Penny crosses the living room without shoes and sets a cup on the table beside Sheldon."
        source.write_text(f"v#enc#0 {original}\nv#enc#1 {second}\nv#enc#2 Sheldon sits on the couch.\n")
        ids = root / "ids.txt"
        ids.write_text("v#enc#0\nv#enc#1\n")
        args = argparse.Namespace(caption_file=source, output=root / "run", target_ids=ids,
                                  limit=None, resume=False, max_tokens=29, word_min=16, word_max=22)
        generator = Generator(malformed, payload, verdict, payload)
        result = rewrite_file(args, generator, tokenizer, {"test": True, "protocol": "context-assisted/oracle"})
        assert result["total"] == 2 and result["accepted"] == 1 and result["fallback"] == 1
        records = [json.loads(line) for line in (args.output / "audit.jsonl").read_text().splitlines()]
        assert records[1]["reason"] == "target_collapse"
        cached = json.loads((args.output / "context_summary.jsonl").read_text())
        assert cached["raw_summary"] == malformed
        assert cached["summary"]["objects"] == [] and cached["grounding_repairs"]
        request = json.loads(generator.messages[1][1]["content"])
        assert "v#enc#2" in request["CLIP_CAPTIONS"]  # Target sampling preserves full sibling context.
        assert (args.output / "rewritten.caption.txt").read_text() == f"v#enc#0 {candidate}\n"
        assert (args.output / "effective.caption.txt").read_text() == f"v#enc#0 {candidate}\nv#enc#1 {second}\n"
        args.resume = True
        assert rewrite_file(args, Generator(), tokenizer, {"test": True, "protocol": "context-assisted/oracle"})["accepted"] == 1
        rejects(lambda: candidate_check(original, {"query": candidate, "grounding": [
            {"text": "Penny", "caption_id": "v#enc#0", "quote": "invented quote"}]},
            tokenizer, captions, 29, 16, 22), "unsupported_quote")
        args.output, args.resume, args.target_ids, args.limit = root / "empty_summary", False, None, 1
        invalid_summary = rewrite_file(args, Generator("invalid JSON", payload, verdict), tokenizer,
                                       {"test": True, "protocol": "context-assisted/oracle"})
        assert invalid_summary["accepted"] == 1  # A malformed shared summary never vetoes the target.
        cached = json.loads((args.output / "context_summary.jsonl").read_text())
        assert cached["raw_summary"] == "invalid JSON" and cached["reason"] == "invalid_json"
        assert all(not values for values in cached["summary"].values())
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "captions.txt"
        second = "Penny crosses the living room without shoes and sets a cup on the table beside Sheldon."
        source.write_text(f"v#enc#0 {original}\nv#enc#1 {second}\nw#enc#0 A dog runs.\n")
        args = argparse.Namespace(caption_file=source, output=root / "batched", target_ids=None,
                                  limit=None, resume=False, max_tokens=29, word_min=16,
                                  word_max=22, batch_size=3)
        empty_summary = {"characters": [], "objects": [], "settings": []}
        generator = BatchGenerator(
            [empty_summary, empty_summary],
            [payload, payload, {"query": "word " * 23, "grounding": []}],
            [{"query": "The dog runs.", "grounding": []}],
            [verdict, verdict, verdict])
        result = rewrite_file(args, generator, tokenizer, {"test": True, "batch_size": 3})
        assert result["accepted"] == 2 and result["fallback"] == 1
        assert generator.calls == [(2, 3), (3, 3), (1, None), (3, 3)]
        records = [json.loads(line) for line in (args.output / "audit.jsonl").read_text().splitlines()]
        assert [row["id"] for row in records] == ["v#enc#0", "v#enc#1", "w#enc#0"]
        assert records[1]["reason"] == "target_collapse"
        assert len(records[2]["attempts"]) == 2
        args.resume = True
        assert rewrite_file(args, BatchGenerator(), tokenizer,
                            {"test": True, "batch_size": 3})["accepted"] == 2
        # Simulate interruption after the first committed query, then resume with batch 1.
        audit_path = args.output / "audit.jsonl"
        first_record = audit_path.read_bytes().splitlines(keepends=True)[0]
        audit_path.write_bytes(first_record)
        legacy = {"test": True, "batch_size": 3,
                  "source_sha256": next(iter(RESUME_COMPATIBLE_SOURCES)),
                  "prompt_sha256": "fixed-prompt"}
        settings_path = args.output / "settings.json"
        settings_path.write_text(json.dumps(legacy))
        original_settings = settings_path.read_bytes()
        current = dict(legacy, batch_size=1, source_sha256="current-source")
        args.batch_size = 1
        generator = Generator(payload, {"query": "The dog runs.", "grounding": []}, verdict)
        result = rewrite_file(args, generator, tokenizer, current)
        assert result["accepted"] == 2 and result["batch_size"] == 1
        assert audit_path.read_bytes().startswith(first_record)
        assert settings_path.read_bytes() == original_settings
        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert [row["id"] for row in records] == ["v#enc#0", "v#enc#1", "w#enc#0"]
        history = json.loads((args.output / "resume_history.jsonl").read_text().splitlines()[-1])
        assert history["completed"] == 1 and history["pending"] == 2
        assert history["execution_settings"] == current
        for changed in (dict(current, prompt_sha256="different-prompt"),
                        dict(current, test=False)):
            try:
                validate_run_settings(args.output, changed, True)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed experiment must not resume")
        settings_path.write_text(json.dumps(dict(legacy, source_sha256="unknown-old-source")))
        try:
            validate_run_settings(args.output, current, True)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown old source must not resume")
    print("Negation, budgets, grounding, verification, batching and interrupted resume with a smaller batch passed.")


if __name__ == "__main__":
    check()
