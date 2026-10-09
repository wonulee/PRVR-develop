"""Run with the Holmes Python environment: python backup/test_restore_links.py."""
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import h5py

from restore_links import repair


with TemporaryDirectory() as temporary:
    source = Path(temporary) / "source"
    restored = Path(temporary) / "restored"
    source.mkdir()
    (source / "features").mkdir()
    with h5py.File(source / "features/data.hdf5", "w") as handle:
        handle.create_dataset("value", data=[1, 2])
    with h5py.File(source / "index.hdf5", "w") as handle:
        handle.create_group("nested")
        handle["nested/value"] = h5py.ExternalLink(str(source / "features/data.hdf5"), "/value")
    (source / "absolute").symlink_to(source / "features/data.hdf5")
    (source / "relative").symlink_to("features/data.hdf5")
    original_hash = hashlib.sha256((source / "index.hdf5").read_bytes()).hexdigest()
    shutil.copytree(source, restored, symlinks=True)
    manifest = {"root": str(source),
                "files": [{"path": path, "destination": "drive"}
                          for path in ("index.hdf5", "features/data.hdf5")],
                "symlinks": [{"path": "absolute"}, {"path": "relative"}]}
    try:
        repair(source, manifest)
    except ValueError as error:
        assert "Refusing original workspace" in str(error)
    else:
        raise AssertionError("Original workspace protection failed")

    # A failed preflight must leave all links unchanged.
    (restored / "features/data.hdf5").unlink()
    try:
        repair(restored, manifest)
    except ValueError:
        assert Path(os.readlink(restored / "absolute")).is_absolute()
        assert hashlib.sha256((restored / "index.hdf5").read_bytes()).hexdigest() == original_hash
    else:
        raise AssertionError("Missing target was accepted")
    shutil.copyfile(source / "features/data.hdf5", restored / "features/data.hdf5")

    with h5py.File(restored / "index.hdf5", "r+") as handle:
        del handle["nested/value"]
        handle["nested/value"] = h5py.ExternalLink(str(Path(temporary) / "outside.hdf5"), "/value")
    outside_hash = hashlib.sha256((restored / "index.hdf5").read_bytes()).hexdigest()
    try:
        repair(restored, manifest)
    except ValueError as error:
        assert "outside original workspace" in str(error)
        assert Path(os.readlink(restored / "absolute")).is_absolute()
        assert hashlib.sha256((restored / "index.hdf5").read_bytes()).hexdigest() == outside_hash
    else:
        raise AssertionError("External workspace target was accepted")
    shutil.copyfile(source / "index.hdf5", restored / "index.hdf5")

    result = repair(restored, manifest)
    assert result["symlinks_repaired"] == 1
    assert result["external_links_repaired"] == 1
    assert os.readlink(restored / "relative") == "features/data.hdf5"
    assert (restored / "absolute").resolve() == restored / "features/data.hdf5"
    with h5py.File(restored / "index.hdf5", "r") as handle:
        link = handle.get("nested/value", getlink=True)
        assert link.filename == "features/data.hdf5" and link.path == "/value"
        assert handle["nested/value"].shape == (2,)
    assert hashlib.sha256((source / "index.hdf5").read_bytes()).hexdigest() == original_hash
    assert repair(restored, manifest)["external_links_repaired"] == 0

print("PASS: source protection, failed preflight, external target rejection, relative links, nested HDF5 links, repeat restore")
