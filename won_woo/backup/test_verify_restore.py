"""Check complete restoration, changed content, missing files, and unsafe paths."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from verify_restore import verify


def expect_failure(root, manifest, fragment):
    try:
        verify(root, manifest)
    except (RuntimeError, ValueError) as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"Expected restore failure containing {fragment!r}")


def test_restore():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        code, feature = root / "code.py", root / "feature.hdf5"
        code.write_bytes(b"abc")
        feature.write_bytes(b"feature")
        link = root / "selected"
        link.symlink_to("feature.hdf5")
        records = [{"path": path.name, "destination": destination,
                    "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "mtime_ns": 0} for path, destination in ((code, "github"), (feature, "drive"))]
        data = {"files": records, "symlinks": [{"path": "selected", "portable_target": "feature.hdf5"}],
                "metadata_files": ["backup/manifest.json"]}
        manifest = root / "manifest.json"
        def save():
            manifest.write_text(json.dumps(data))
        save()
        os.utime(code, (100, 100))
        counts = verify(root, manifest)
        assert counts["github"] == {"files": 1, "bytes": 3}
        assert counts["drive"] == {"files": 1, "bytes": 7}
        assert counts["symlinks"] == 1
        code.write_bytes(b"xyz")
        expect_failure(root, manifest, "SHA256 mismatch")
        code.write_bytes(b"abc")
        del records[0]["sha256"]
        save()
        expect_failure(root, manifest, "Missing or invalid SHA256 digest")
        records[0]["sha256"] = hashlib.sha256(b"abc").hexdigest()
        feature.unlink()
        save()
        expect_failure(root, manifest, "Missing file")
        expect_failure(root, manifest, "Broken symlink target")
        feature.write_bytes(b"different size")
        expect_failure(root, manifest, "Size mismatch")
        feature.write_bytes(b"feature")
        link.unlink()
        link.symlink_to("./feature.hdf5")
        expect_failure(root, manifest, "Symlink target mismatch")
        link.unlink()
        link.symlink_to("code.py")
        expect_failure(root, manifest, "Symlink target mismatch")
        records[0]["path"] = "../outside"
        save()
        expect_failure(root, manifest, "Unsafe relative path")


if __name__ == "__main__":
    test_restore()
    print("Restore verification fixture passed")
