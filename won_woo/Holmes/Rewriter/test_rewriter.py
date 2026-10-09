"""Offline checks: python -m Rewriter.test_rewriter."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .context_manager import build_video_context_db, get_sibling_queries
from .linguistic_rules import detect_linguistic_flags
from .mca_rewriter import (RewriteRejected, rewrite_batch_with_context, rewrite_file,
                           rewrite_with_context, validate_length)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + (["<s>", "</s>"] if add_special_tokens else [])


class Generator:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.tokenizer = self
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(list(messages))
        return "prompt"

    def __call__(self, prompt, **kwargs):
        return [{"generated_text": next(self.responses)}]


class BatchGenerator:
    def __init__(self, *batches):
        self.batches = iter(batches)
        self.tokenizer = self
        self.messages = []
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(list(messages))
        return "prompt"

    def __call__(self, prompts, **kwargs):
        assert isinstance(prompts, list)
        self.calls.append((len(prompts), kwargs["batch_size"]))
        responses = next(self.batches)
        assert len(responses) == len(prompts)
        return [[{"generated_text": response}] for response in responses]


def check():
    tokenizer = Tokenizer()
    assert validate_length("word " * 28, tokenizer)
    assert not validate_length("word " * 29, tokenizer)
    assert not validate_length("", tokenizer)
    original = "He walks."
    generator = Generator(json.dumps({"query": "word " * 29}), '{"query": "He is walking."}')
    assert rewrite_with_context(original, [], generator, tokenizer) == "He is walking."
    assert len(generator.messages) == 2
    cases = [
        ("He does not run.", '{"query": "He runs."}', "temporal_or_negation_change"),
        ("He doesn't run.", '{"query": "He runs."}', "temporal_or_negation_change"),
        ("He walks before eating.", '{"query": "He walks after eating."}', "temporal_or_negation_change"),
        (original, "not JSON", "invalid_json"),
        (original, '[]', "invalid_text"),
        (original, '{"query": null}', "invalid_text"),
        (original, '{"query": "He walks."}', "unchanged"),
        (original, json.dumps({"query": "He\rwalks slowly."}), "invalid_text"),
        (original, json.dumps({"query": "He\nwalks slowly."}), "invalid_text"),
        (original, json.dumps({"query": "word " * 29}), "token_budget"),
    ]
    for target, response, reason in cases:
        try:
            rewrite_with_context(target, [], Generator(response, response), tokenizer)
        except RewriteRejected as exc:
            assert exc.reason == reason, (exc.reason, reason)
        else:
            raise AssertionError(reason)
    assert all(detect_linguistic_flags("He isn't happy.").values())
    batched = BatchGenerator(
        ['{"query": "He is walking."}', json.dumps({"query": "word " * 29}), "not JSON"],
        ['{"query": "A dog is running."}'],
    )
    outcomes = rewrite_batch_with_context(
        [("He walks.", []), ("A dog runs.", []), ("A woman sits.", [])],
        batched, tokenizer, batch_size=3,
    )
    assert outcomes[0] == "He is walking." and outcomes[1] == "A dog is running."
    assert isinstance(outcomes[2], RewriteRejected) and outcomes[2].reason == "invalid_json"
    assert batched.calls == [(3, 3), (1, 3)]
    assert [len(messages) for messages in batched.messages] == [4, 4, 4, 6]
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "captions.txt"
        content = "v#enc#0 He walks.\nw#enc#0 A dog runs.\nv#enc#7 A woman sits.\n"
        source.write_text(content, encoding="utf-8")
        contexts = build_video_context_db(source)
        assert get_sibling_queries(contexts, "v", 0) == ["A woman sits."]
        assert get_sibling_queries(contexts, "w", 0) == []
        generator = Generator('{"query": "He is walking."}', "invalid JSON")
        summary = rewrite_file(source, root / "output", generator, tokenizer, limit=2)
        assert summary == dict(total=2, accepted=1, rejected=1)
        assert (root / "output/rewritten.caption.txt").read_text() == "v#enc#0 He is walking.\n"
        audit = [json.loads(line) for line in (root / "output/audit.jsonl").read_text().splitlines()]
        assert audit[0]["siblings"] == ["A woman sits."]  # Limit targets, not context.
        assert audit[1]["reason"] == "invalid_json" and not audit[1]["accepted"]
        assert source.read_text() == content
        batch_generator = BatchGenerator(
            ['{"query": "He is walking."}', "invalid JSON"],
            ['{"query": "A woman remains seated."}'],
        )
        batch_summary = rewrite_file(source, root / "batch", batch_generator, tokenizer,
                                     batch_size=2)
        assert batch_summary == dict(total=3, accepted=2, rejected=1)
        assert batch_generator.calls == [(2, 2), (1, 2)]
        assert (root / "batch/rewritten.caption.txt").read_text() == (
            "v#enc#0 He is walking.\nv#enc#7 A woman remains seated.\n"
        )
        batch_audit = [json.loads(line) for line in (root / "batch/audit.jsonl").read_text().splitlines()]
        assert [row["id"] for row in batch_audit] == ["v#enc#0", "w#enc#0", "v#enc#7"]
        (root / "batch/audit.jsonl").write_text(json.dumps(batch_audit[0]) + "\n")
        resumed = BatchGenerator(['{"query": "A dog is running."}',
                                  '{"query": "A woman remains seated."}'])
        assert rewrite_file(source, root / "batch", resumed, tokenizer,
                            batch_size=2, resume=True)["accepted"] == 3
        assert resumed.calls == [(2, 2)]
        assert [json.loads(line)["id"] for line in
                (root / "batch/audit.jsonl").read_text().splitlines()] == [
                    "v#enc#0", "w#enc#0", "v#enc#7"]
        rewrite_file(source, root / "batch", BatchGenerator(), tokenizer,
                     batch_size=2, resume=True)
        rewrite_file(source, root / "no-context", Generator('{"query": "He is walking."}'),
                     tokenizer, limit=1, use_siblings=False)
        assert json.loads((root / "no-context/audit.jsonl").read_text())["siblings"] == []
        try:
            rewrite_file(source, root / "output", generator, tokenizer)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output was overwritten")
        for invalid in ("v#enc#0 A.\nv#enc#0 B.\n", "v#enc#0#ar A.\n", "v#enc#01 A.\n", ""):
            source.write_text(invalid)
            try:
                build_video_context_db(source)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid caption file accepted")
    print("Context isolation, retries, token limits, meaning guards and output checks passed.")


if __name__ == "__main__":
    check()
