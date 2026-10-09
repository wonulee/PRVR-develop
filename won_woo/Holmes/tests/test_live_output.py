"""Run from Holmes: python tests/test_live_output.py."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Rewriter.experiment import stage


def check():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "logs").mkdir()
        (root / "stages").mkdir()
        seen = root / "seen"
        output = root / "output"

        class Terminal(io.StringIO):
            def write(self, text):
                result = super().write(text)
                if "epoch 0: 50%\r" in self.getvalue():
                    seen.touch()
                return result

        # The child cannot finish until its newline-free update reaches the terminal.
        child = """
import os, sys, time
from pathlib import Path
seen, output = map(Path, sys.argv[1:])
os.write(2, b'epoch 0: 50%\\r')
deadline = time.monotonic() + 5
while not seen.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('Progress was buffered until process exit')
    time.sleep(0.02)
os.write(1, '학습 완료\\n'.encode())
output.touch()
"""
        terminal = Terminal()
        command = [sys.executable, "-u", "-c", child, str(seen), str(output)]
        with redirect_stdout(terminal):
            stage(root, "live", command, [output])
        expected = "epoch 0: 50%\r학습 완료\n".encode()
        assert (root / "logs/live.log").read_bytes() == expected
        assert "학습 완료" in terminal.getvalue()
        with redirect_stdout(terminal):
            stage(root, "live", command, [output])
        assert "[skip] live" in terminal.getvalue()
        assert (root / "logs/live.log").read_bytes() == expected
        with redirect_stdout(terminal):
            try:
                stage(root, "failed", [sys.executable, "-c",
                      "import sys; print('failure', file=sys.stderr); sys.exit(7)"], [])
            except subprocess.CalledProcessError as exc:
                assert exc.returncode == 7
            else:
                raise AssertionError("Child failure was swallowed")
        assert not (root / "stages/failed.json").exists()
        assert "failure" in terminal.getvalue()
    print("Live carriage-return output, log mirroring, stage reuse and failure propagation passed.")


if __name__ == "__main__":
    check()
