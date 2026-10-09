"""Run with the Holmes Python environment; checks enforced budgets and safe application."""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

def main():
    import spacy
    from select_diagram_synonyms import syntax_status

    nlp = spacy.load("en_core_web_md", disable=["ner"])
    for caption, expression, rejected in [
            ("Leonard gives Sheldon a lift to work.", "lift", True),
            ("Rachel steps into the lift.", "lift", False),
            ("Monica takes a drink from the cup.", "drink", True),
            ("Monica holds a drink.", "drink", False),
            ("A man is drinks from the bottle.", "drinks", True),
            ("Monica orders a drink.", "drink", True),
            ("Sheldon stops beside a bike.", "bike", True),
            ("Sheldon pedals a bike.", "bike", False),
            ("Ross walks into the coffee shop.", "shop", True)]:
        tokens = [token for token in nlp(caption) if token.text == expression]
        assert bool(syntax_status(tokens, "object", tokens[0].lemma_)) == rejected
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captions = [
            "Ross places a phone on the desk while Monica sits beside the open window.",
            "Ross puts a phone on the desk.",
            "Ross places a phone on the desk while Monica sits beside the open window.",
        ]
        original = "".join(f"clip#enc#{i} {caption}\n" for i, caption in enumerate(captions)).encode()
        path, pairs, out = root / "train.txt", root / "pairs.csv", root / "out"
        path.write_bytes(original)
        fields = ["caption_id", "term_type", "caption", "source_expression", "synonym_expression",
                  "synonym_rank", "meaning", "val_caption_id"]
        with pairs.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i, (caption, target) in enumerate(zip(captions, ["telephone", "telephone", "xyz" * 40])):
                writer.writerow(dict(caption_id=f"clip#enc#{i}", term_type="object", caption=caption,
                                     source_expression="phone", synonym_expression=target,
                                     synonym_rank=1, meaning="test", val_caption_id="val#enc#0"))
        result = subprocess.run([
            sys.executable, str(Path(__file__).with_name("apply_val_synonyms.py")),
            "--captions", str(path), "--pairs", str(pairs), "--output-dir", str(out),
            "--enforce-length-budget", "--apply"], check=True, capture_output=True, text=True)
        summary = json.loads(result.stdout)
        actual = path.read_bytes().splitlines(keepends=True)
        before = original.splitlines(keepends=True)
        assert actual[0] == before[0].replace(b"phone", b"telephone")
        assert actual[1:] == before[1:]
        assert (out / "tvrtrain.caption.original.txt").read_bytes() == original
        assert summary["changed_caption_count"] == 1
        assert summary["changed_captions_within_length_budget"] == 1
        assert summary["replacement_status_counts"]["length_budget_exceeded"] == 2
        assert [line.split()[0] for line in actual] == [line.split()[0] for line in before]
    print("PASS: wrong-sense/compound guards and word/token budgets; IDs and backup preserved")


if __name__ == "__main__":
    main()
