"""Run from Holmes: python tests/test_rewrite_report.py."""
from contextlib import redirect_stdout
from collections import Counter
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.inspect_rewrites import read_new, show, status


def check():
    first = dict(id="v#enc#0", original="He walks.", candidate="He is walking.",
                 accepted=True, reason=None, tokens=6)
    second = dict(id="v#enc#1", original="He eats then walks.", candidate="He eats and walks.",
                  accepted=False, reason="temporal_or_negation_change", tokens=7)
    record = json.dumps(second).encode()
    stream = io.BytesIO(json.dumps(first).encode() + b"\n" + record[:20])
    assert list(read_new(stream)) == [first]
    offset = stream.tell()
    stream.seek(0, 2)
    stream.write(record[20:] + b"\n")
    stream.seek(offset)
    assert list(read_new(stream)) == [second]
    assert list(read_new(stream)) == []
    output = io.StringIO()
    with redirect_stdout(output):
        show(Counter([status(first), status(second)]), [second], 10)
    rendered = output.getvalue()
    assert "2/10 (20.0%)" in rendered
    assert "채택률 50.0%" in rendered
    assert "['then'] → []" in rendered
    assert first["original"] not in rendered
    print("Partial audit records, live append, acceptance counts and exact rejection explanations passed.")


if __name__ == "__main__":
    check()
