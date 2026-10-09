"""One small fixture checks the backup split and a portable staged link."""

import contextlib
import io
from pathlib import Path
import tempfile

from backup_workspace import GITHUB_LIMIT, hash_files, prepare, stage


def test_backup():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "source"
        root.mkdir()
        (root / "code.py").write_text("print('Holmes')\n")
        (root / "small.csv").write_text("score\n1\n")
        (root / "large.csv").write_bytes(b"x" * (GITHUB_LIMIT + 1))
        (root / "features.hdf5").write_bytes(b"small feature still belongs to Drive")
        (root / "blob").write_bytes(b"\0binary")
        (root / ".env").write_text("EXAMPLE=fixture\n")
        (root / "__pycache__").mkdir()
        (root / "__pycache__/cache.pyc").write_bytes(b"cache")
        (root / "selected").symlink_to(root / "features.hdf5")
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare(root)
            snapshot = stage(root, base / "output")
            hash_files(root, "github")
        destinations = {record["path"]: record["destination"] for record in manifest["files"]}
        assert destinations["code.py"] == destinations["small.csv"] == "github"
        assert destinations["large.csv"] == destinations["features.hdf5"] == destinations["blob"] == "drive"
        assert destinations[".env"] == destinations["__pycache__/cache.pyc"] == "skip"
        assert (snapshot / "code.py").read_text() == (root / "code.py").read_text()
        assert not (snapshot / "features.hdf5").exists()
        assert (snapshot / "selected").is_symlink()
        assert (snapshot / "selected").readlink() == Path("features.hdf5")
        assert (root / "selected").readlink().is_absolute()
        assert (snapshot / "backup/manifest.json").exists()
        try:
            stage(root, base / "output")
        except FileExistsError:
            pass
        else:
            raise AssertionError("Staging must refuse to overwrite an existing snapshot")
        (root / "code.py").write_text("changed\n")
        try:
            stage(root, base / "changed_output")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Staging must detect source changes after prepare")


if __name__ == "__main__":
    test_backup()
    print("Backup fixture passed")
